from typing import List

import pytest
from unittest.mock import MagicMock
from pathlib import Path
import os
from torch.utils.data import Dataset

from utils.model_utils import save_checkpoint, iid_sharding, non_iid_sharding, non_iid_dirichlet


class SimpleDataset(Dataset):
    def __init__(self, data: List[int], labels: List[int]):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


@pytest.mark.parametrize("is_colab", [False])  # only test for local machine
def test_save_checkpoint_local(model_and_optimizer, is_colab):
    model, optimizer = model_and_optimizer

    # Mock os.path.exists to simulate the local environment (is_colab = False)
    mock_exists = MagicMock()
    mock_exists.return_value = (
        False  # Simulate local directory (it doesn't exist in Colab)
    )

    # Simulate saving to the checkpoint directory
    checkpoint_dir = "./checkpoints/"
    checkpoint_path = Path(checkpoint_dir) / "model_checkpoint.pth"

    # Mock the behavior of os.path.exists
    os.path.exists = mock_exists

    save_checkpoint(model, optimizer, epoch=1, loss=0.25)

    # Assert that the checkpoint file is saved to the correct directory
    assert checkpoint_path.exists(), f"Checkpoint was not saved to {checkpoint_path}"

    # Cleanup: Remove the checkpoint file after the test
    checkpoint_path.unlink()  # This deletes the file


def test_iid_sharding(binary_dataset):
    num_clients = 2
    shard_data = iid_sharding(binary_dataset, num_clients, seed=42)

    # Ensure correct number of clients
    assert (
        len(shard_data) == num_clients
    ), "Number of clients in the sharded data is incorrect"

    # Ensure each client has 5 data points
    for client_id, indices in shard_data.items():
        assert len(indices) == 5, f"Client {client_id} does not have 5 data points"

    # Ensure no data points are repeated and all are accounted for
    all_indices = sum(list(shard_data.values()), [])
    assert len(set(all_indices)) == len(
        binary_dataset
    ), "Some data points are missing or duplicated in the sharded data"


def test_non_iid_sharding(ternary_dataset):
    num_clients = 2
    num_classes = 2  # Each client will get 2 classes
    shard_data = non_iid_sharding(ternary_dataset, num_clients, num_classes, seed=42)

    # Ensure correct number of clients
    assert (
        len(shard_data) == num_clients
    ), "Number of clients in the sharded data is incorrect"

    # Ensure each client has data from 2 classes
    for client_id, indices in shard_data.items():
        unique_labels = set(ternary_dataset[idx][1] for idx in indices)
        assert (
            len(unique_labels) == 2
        ), f"Client {client_id} does not have data from exactly 2 classes"

    # Ensure no data points are repeated and all are accounted for
    all_indices = sum(list(shard_data.values()), [])
    assert len(set(all_indices)) == len(
        ternary_dataset
    ), "Some data points are missing or duplicated in the sharded data"

def test_non_idd_dirichlet_sharding(ternary_dataset):
    
    num_clients = 5
    num_classes = 3 # Matches ternary_dataset fixture
    alpha = 0.5
    seed = 123

    client_data = non_iid_dirichlet(
        dataset=ternary_dataset,
        num_clients=num_clients,
        num_classes=num_classes,
        alpha=alpha,
        seed=seed
    )

    # 1. Check structure: Correct number of clients
    assert len(client_data) == num_clients, \
        f"Expected {num_clients} clients, but got {len(client_data)}"

    # 2. Check structure: Values are lists of integers (indices)
    all_indices_collected = []
    for client_id, indices in client_data.items():
        assert isinstance(client_id, int), "Client ID is not an integer"
        assert isinstance(indices, list), f"Data for client {client_id} is not a list"
        if indices: # If list is not empty
            assert all(isinstance(idx, int) for idx in indices), \
                f"Indices for client {client_id} are not all integers"
        all_indices_collected.extend(indices)

    # 3. Check completeness: All dataset indices are assigned exactly once
    assert len(all_indices_collected) == len(ternary_dataset), \
        f"Total number of assigned indices ({len(all_indices_collected)}) does not match dataset size ({len(ternary_dataset)})"
    assert len(set(all_indices_collected)) == len(all_indices_collected), \
        "Duplicate indices found across clients"
    assert set(all_indices_collected) == set(range(len(ternary_dataset))), \
        "The set of assigned indices does not match the set of expected indices [0, ..., N-1]"

def test_iid_reproducibility(binary_dataset):
    num_clients = 2
    # Generate two shardings with the same seed
    shard_data_1 = iid_sharding(binary_dataset, num_clients, seed=42)
    shard_data_2 = iid_sharding(binary_dataset, num_clients, seed=42)

    # Assert that both shardings are the same (reproducibility)
    assert (
        shard_data_1 == shard_data_2
    ), "Sharding results are not reproducible with the same seed"


def test_non_iid_reproducibility(ternary_dataset):
    num_clients = 2
    num_classes = 2  # Each client will get 2 classes
    # Generate two shardings with the same seed
    shard_data_1 = non_iid_sharding(ternary_dataset, num_clients, num_classes, seed=42)
    shard_data_2 = non_iid_sharding(ternary_dataset, num_clients, num_classes, seed=42)

    # Assert that both shardings are the same (reproducibility)
    assert (
        shard_data_1 == shard_data_2
    ), "Sharding results are not reproducible with the same seed"


def test_non_idd_dirichlet_reproducibility(ternary_dataset):
    """Tests if the function produces identical results with the same seed."""
    num_clients = 4
    num_classes = 3
    alpha = 0.1 # Use a smaller alpha for potentially higher non-IID
    seed = 42

    # Run first time
    client_data_1 = non_iid_dirichlet(
        dataset=ternary_dataset,
        num_clients=num_clients,
        num_classes=num_classes,
        alpha=alpha,
        seed=seed
    )

    # Run second time with identical parameters
    client_data_2 = non_iid_dirichlet(
        dataset=ternary_dataset,
        num_clients=num_clients,
        num_classes=num_classes,
        alpha=alpha,
        seed=seed
    )

    # Assert that the resulting dictionaries are identical
    # This works because the final shuffle within each client list is also seeded.
    assert client_data_1 == client_data_2, \
        "non_iid_dirichlet results are not reproducible with the same seed"