import enum
import logging

import attr
import torch
from typing import List, Dict, Optional, Tuple

import wandb
from torch.utils.data import Dataset
import random
import copy

from tqdm import tqdm

from core.train import train_model, compute_predictions
from core.train_params import TrainingParams, is_nn_module
from dataset.cifar_100 import get_dataloader
from utils.model_utils import iid_sharding, non_iid_sharding
from utils.numpy_utils import numpy_random_seed


class ShardingType(enum.Enum):
    IID = enum.auto()
    NON_IID = enum.auto()


def federated_averaging(
    global_model: torch.nn.Module, client_models: List[torch.nn.Module]
) -> torch.nn.Module:
    """
    Perform Federated Averaging on the given list of client models to update the global model.

    The function averages the weights of the provided client models and updates the global model.

    Args:
    - global_model (torch.nn.Module): The global model that will be updated.
    - client_models (List[torch.nn.Module]): A list of client models whose weights will be averaged.

    Returns:
    - torch.nn.Module: The updated global model with averaged weights.

    Example:
    >>> updated_global_model = federated_averaging(global_model, client_models)
    """
    global_dict = global_model.state_dict()

    for key in global_dict:
        # Compute the average of the corresponding parameter across all client models
        global_dict[key] = torch.mean(
            torch.stack(
                [
                    client_model.state_dict()[key].float()
                    for client_model in client_models
                ],
                dim=0,
            ),
            dim=0,
        )

    # Load the averaged weights into the global model
    global_model.load_state_dict(global_dict)

    return global_model


@attr.s(kw_only=True)
class FederatedAveraging:
    global_model: torch.nn.Module = attr.ib(validator=is_nn_module)
    trainset: Dataset = attr.ib(validator=attr.validators.instance_of(Dataset))
    valset: Dataset = attr.ib(validator=attr.validators.instance_of(Dataset))
    client_training_params: TrainingParams = attr.ib(
        validator=attr.validators.instance_of(TrainingParams)
    )
    sharding_type: ShardingType = attr.ib(
        validator=attr.validators.instance_of(ShardingType)
    )
    _num_clients: int = attr.ib(
        validator=attr.validators.ge(0), default=100
    )  # number of clients
    _client_fraction: float = attr.ib(
        default=0.1
    )  # percentage of clients to select for training
    _rounds: int = attr.ib(
        default=10
    )  # the number of times the central server communicates with the client devices
    _batch_size: int = attr.ib(default=32)
    _num_classes: Optional[int] = attr.ib(
        default=None
    )  # num classes for non iid sharing
    _seed: int = attr.ib(default=42)
    _training_session_name: Optional[str] = attr.ib(
        validator=attr.validators.optional(attr.validators.instance_of(str)),
        default=None,
    )
    """
    Class to perform Federated Averaging training across multiple clients.

    Args:
        global_model (torch.nn.Module): The global model to be updated via federated learning.
        trainset (Dataset): The training dataset to be distributed among clients.
        valset (Dataset): The validation dataset to be used for local and global evaluation.
        client_training_params (TrainingParams): Parameters used for local client training.
        sharding_type (ShardingType): Strategy to shard dataset among clients (IID or NON_IID).
        _num_clients (int): Total number of simulated clients. Default is 100.
        _client_fraction (float): Fraction of clients selected in each round. Default is 0.1.
        _rounds (int): Number of communication rounds between server and clients. Default is 10.
        _batch_size (int): Batch size for training and validation. Default is 32.
        _num_classes (Optional[int]): Number of classes for non-IID sharding. Required if NON_IID.
        _seed (int): Base seed for reproducibility. Default is 42.
        _training_session_name (str): Base name for training and WandB logging.
    Methods:
        train: perform training using federated learning algorithm. 
    """

    def __attrs_post_init__(self):
        if self.sharding_type == ShardingType.NON_IID:
            assert (
                self._num_classes is not None
            ), f"num_classes cannot be None for {self.sharding_type.name}"

    def _get_client_data(
        self, seed: int
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        if self.sharding_type == ShardingType.IID:
            train_client_data = iid_sharding(
                self.trainset, self._num_clients, seed=seed
            )
            val_client_data = iid_sharding(self.valset, self._num_clients, seed=seed)
        elif self.sharding_type == ShardingType.NON_IID:
            train_client_data = non_iid_sharding(
                self.trainset, self._num_clients, self._num_classes, seed=seed
            )
            val_client_data = non_iid_sharding(
                self.valset, self._num_clients, self._num_classes, seed=seed
            )
        else:
            raise NotImplementedError()
        return train_client_data, val_client_data

    @property
    def _session_name(self) -> str:
        optimizer_params = self.client_training_params.optimizer_params
        _training_session_name = self._training_session_name or "fl"
        return (
            _training_session_name
            + f"_mom_{optimizer_params.get('momentum'):.2f}_decay_{optimizer_params.get('weight_decay'):.3f}_lr"
            + f"_{self.client_training_params.learning_rate:.2f}_{self.sharding_type.name}"
        )

    def _evaluate(self) -> Tuple[float, float, float, float]:
        full_train_loader = get_dataloader(
            dataset=self.trainset, batch_size=self._batch_size, shuffle=True
        )
        full_val_loader = get_dataloader(
            dataset=self.valset, batch_size=self._batch_size, shuffle=False
        )
        _, _, train_loss, train_accuracy = compute_predictions(
            model=self.global_model,
            dataloader=full_train_loader,
            loss_function=self.client_training_params.loss_function,
        )
        _, _, val_loss, val_accuracy = compute_predictions(
            model=self.global_model,
            dataloader=full_val_loader,
            loss_function=self.client_training_params.loss_function,
        )
        return (
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
        )

    def train(self) -> None:
        """
        Run the Federated Averaging training process for the specified number of rounds.
        """

        wandb.init(
            project="fl",
            name=self._session_name,
            config={
                "epochs": self.client_training_params.epochs,
                "batch_size": self._batch_size,
                "learning_rate": self.client_training_params.learning_rate,
                "architecture": self.client_training_params.model.__class__.__name__,
                "optimizer_class": self.client_training_params.optimizer_class.__name__,
                "loss_function": self.client_training_params.loss_function.__class__.__name__,
                "sharding_type": self.sharding_type.name,
                **(self.client_training_params.optimizer_params or {}),
            },
        )
        for r in tqdm(range(self._rounds)):

            logging.info(f"Training round {r} of {self._rounds}:")
            # Select a subset of clients (client_fraction * num_clients)
            seed = self._seed + r  # update seed, for reproducibility in the loop
            with numpy_random_seed(seed=seed):
                selected_clients = random.sample(
                    range(self._num_clients),
                    int(self._client_fraction * self._num_clients),
                )
            logging.info(
                f"Selected {len(selected_clients)} clients: {selected_clients}"
            )
            client_models = []

            train_client_data_idx, val_client_data_idx = self._get_client_data(
                seed=seed
            )
            # Train the selected clients locally (simulated by local training process)
            for client_id in tqdm(selected_clients):
                local_model = copy.deepcopy(self.global_model)
                training_params = attr.evolve(
                    self.client_training_params, model=local_model
                )
                train_loader = get_dataloader(
                    self.trainset,
                    train_client_data_idx[client_id],
                    batch_size=self._batch_size,
                    shuffle=True,
                )
                val_loader = get_dataloader(
                    self.valset,
                    val_client_data_idx[client_id],
                    batch_size=self._batch_size,
                    shuffle=False,
                )
                res_dict = train_model(
                    training_params=training_params,
                    train_loader=train_loader,
                    client_id=client_id,
                    val_loader=val_loader,
                    wandb_log=False,
                    wandb_save=False,
                )
                logging.info(
                    f"Best accuracy client {client_id} in round {r}: {res_dict['best_accuracy']}"
                )
                client_models.append(res_dict["model"])

            # Perform Federated Averaging to update the global model
            self.global_model = federated_averaging(self.global_model, client_models)

            # Evaluate the global model
            train_loss, train_accuracy, val_loss, val_accuracy = self._evaluate()
            wandb.log(
                {
                    "Round": r,
                    "Train Loss": train_loss,
                    "Train Accuracy": train_accuracy,
                    "Validation Loss": val_loss,
                    "Validation Accuracy": val_accuracy,
                }
            )

            model_name = f"global_model_round_{r}.pth"
            torch.save(self.global_model.state_dict(), model_name)
            wandb.save(model_name)

        wandb.finish()
        logging.info("FL Training completed.")
