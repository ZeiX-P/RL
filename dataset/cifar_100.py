from typing import List, Tuple, Optional

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split

from utils.model_utils import iid_sharding, non_iid_sharding


def get_train_transform():
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(224),  # resize to 224 x 224 (required by ViT)
            transforms.ToTensor(),
            # Imagenet normalization
            # DINO model has learned features from ImageNet, so during fine-tuning on CIFAR-100,
            # the model will expect inputs to be normalized in the same way as during pretraining.
            # TODO check this is correct
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform


def get_test_tranform():
    transform = transforms.Compose(
        [
            transforms.Resize(224),  # resize to 224 x 224 (required by ViT)
            transforms.ToTensor(),
            # Imagenet normalization
            # DINO model has learned features from ImageNet, so during fine-tuning on CIFAR-100,
            # the model will expect inputs to be normalized in the same way as during pretraining.
            # TODO check this is correct
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform


def get_cifar_100_datasets() -> Tuple[Dataset, Dataset]:
    train_transform = get_train_transform()
    test_transform = get_test_tranform()
    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=test_transform
    )
    return trainset, testset


def get_cifar_100_train_valset_datasets(
    dataset: Dataset, seed: int = 42
) -> Tuple[Dataset, Dataset]:
    data_size = len(dataset)  # type: ignore
    train_size = int(0.8 * data_size)  # 80pc train, 20pc validation
    val_size = data_size - train_size
    trainset, valset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )
    return trainset, valset


def get_dataloader(
    dataset: Dataset,
    indices: Optional[List[int]] = None,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
) -> DataLoader:
    """
    Return a DataLoader for a given dataset, optionally using a subset of the data.

    Args:
        dataset (Dataset): The dataset to be loaded.
        indices (List[int], optional): A list of indices to create a subset of the dataset. If None, the entire dataset is used. Defaults to None.
        batch_size (int, optional): The number of samples per batch. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the data at the beginning of each epoch. Defaults to True.
            NB Should be set to False for test dataset and to True for train/val datasets.

    Returns:
        DataLoader: A DataLoader for the given dataset, potentially using a subset of the data.
    """
    batch_size = batch_size or 32  # if batch_size is None, use 32
    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices=indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def get_cifar_dataloaders(batch_size=None):
    trainset, _ = get_cifar_100_datasets()
    trainset, valset = get_cifar_100_train_valset_datasets(trainset)
    train_dataloader, val_dataloader = get_dataloader(
        trainset, batch_size=batch_size
    ), get_dataloader(valset, batch_size=batch_size)
    return train_dataloader, val_dataloader


if __name__ == "__main__":

    K = 10  # Number of clients
    num_classes = 2  # Number of classes per client (non iid)

    train_set, _ = get_cifar_100_datasets()
    trainset, valset = get_cifar_100_train_valset_datasets(train_set)

    iid_client_data = iid_sharding(trainset, K)
    non_iid_client_data = non_iid_sharding(trainset, K, num_classes)

    client_0_iid_loader = get_dataloader(trainset, iid_client_data[0], shuffle=True)
    client_0_non_iid_data = get_dataloader(
        trainset, non_iid_client_data[0], shuffle=True
    )
    val_loader = DataLoader(valset, batch_size=32, shuffle=False)
