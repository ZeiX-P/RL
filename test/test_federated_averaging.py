import torch
from core.federated_averaging import federated_averaging
import pytest
from models.cnn import CustomNet
from test.conftest import simple_cnn, SimpleCNN
import torch.nn as nn

@pytest.fixture
def global_model(simple_cnn):
    """Fixture to create a global model for testing"""
    # Initialize with zeros for easy testing
    for param in simple_cnn.parameters():
        nn.init.zeros_(param)
    return simple_cnn

@pytest.fixture
def client_models(simple_cnn):
    """Fixture to create client models with known weights"""
    models = []
    # Create 3 client models
    for i in range(3):
        model = SimpleCNN() # Create a new instance for each client
        # Initialize with specific values for testing
        for param in model.parameters():
            nn.init.constant_(param, i + 1)  # Values 1, 2, 3
        models.append(model)
    return models

def test_federated_averaging_updates_global_model(global_model, client_models):
    """Test that federated averaging properly updates the global model"""
    # Before averaging, check that global model has all zeros
    for param in global_model.parameters():
        assert torch.all(param == 0)

    # Perform federated averaging
    updated_model = federated_averaging(global_model, client_models)

    # After averaging, all parameters should be equal to the mean value (2)
    # Client models had values 1, 2, 3, so average is 2
    for param in updated_model.parameters():
        assert torch.allclose(param, torch.tensor(2.0))

    # Check that the returned model is the same object as the input global model
    assert updated_model is global_model

def test_federated_averaging(simple_cnn):
    """Test that federated averaging updates the global model parameters"""
    # Initialize a global model and client models
    global_model = simple_cnn
    client_models = [SimpleCNN(), SimpleCNN(), SimpleCNN()]  # Create new instances

    # Save the initial state of global model parameters
    initial_global_params = {
        name: param.clone() for name, param in global_model.named_parameters()
    }

    # Perform Federated Averaging
    updated_global_model = federated_averaging(global_model, client_models)

    # Check that the global model parameters have been updated
    for (name, initial_param), (name2, updated_param) in zip(
        initial_global_params.items(), updated_global_model.named_parameters()
    ):
        assert initial_param.shape == updated_param.shape, f"Shape mismatch for {name}"
        assert not torch.equal(
            initial_param, updated_param
        ), f"Parameters for {name} should be different after averaging"

    # Ensure no parameters are NaN or infinite after the averaging process
    for param in updated_global_model.parameters():
        assert torch.all(
            torch.isfinite(param)
        ), "Parameters should not contain NaNs or Infs after averaging"

def test_federated_averaging_different_parameter_values(global_model, simple_cnn):
    """Test federated averaging with different parameter values"""
    # Create client models with different weights
    client_models = []
    values = [1.0, 2.0, 3.0, 4.0]  # Mean = 2.5

    for val in values:
        model = SimpleCNN()
        for param in model.parameters():
            nn.init.constant_(param, val)
        client_models.append(model)

    # Perform federated averaging
    updated_model = federated_averaging(global_model, client_models)

    # Check that parameters equal the mean of all client models
    for param in updated_model.parameters():
        assert torch.allclose(param, torch.tensor(2.5))