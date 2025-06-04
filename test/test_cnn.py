import numpy as np
import torch
import pytest
from models.cnn import CustomNet

def test_custom_net():
    torch.random.manual_seed(42)
    model = CustomNet()
    out = model(torch.randn(32, 3, 224, 224))
    out_mean = out.mean()
    assert np.allclose(out_mean.detach().numpy(), -0.01339262)


@pytest.fixture
def sample_input():
    """Fixture for sample input tensor"""
    return torch.randn(4, 3, 224, 224)

def test_forward_pass(custom_cnn_model, sample_input):
    """Test forward pass works with expected shape"""
    output = custom_cnn_model(sample_input)
    
    # Check output shape (batch_size=4, num_classes=200)
    assert output.shape == (4, 200)

def test_train_eval_modes(custom_cnn_model, sample_input):
    """Test model behaves differently in train vs eval mode"""
    # Set to training mode
    custom_cnn_model.train()
    out1 = custom_cnn_model(sample_input)
    
    # Set to evaluation mode
    custom_cnn_model.eval()
    out2 = custom_cnn_model(sample_input)
    
    # Outputs should differ due to dropout
    assert not torch.allclose(out1, out2)

def test_model_with_different_input_size(custom_cnn_model):
    """Test model with different input sizes"""
    # Smaller input size
    small_input = torch.randn(2, 3, 160, 160)
    output = custom_cnn_model(small_input)
    
    # Should still output correct shape
    assert output.shape == (2, 200)

def test_gradient_flow(custom_cnn_model, sample_input):
    """Test that gradients properly flow through the model"""
    # Clear any existing gradients
    custom_cnn_model.zero_grad()
    
    # Forward pass
    output = custom_cnn_model(sample_input)
    
    # Create a simple loss and backpropagate
    loss = output.sum()
    loss.backward()
    
    # Check if any parameter has non-zero gradients
    for param in custom_cnn_model.parameters():
        if param.grad is not None and torch.sum(param.grad.abs()) > 0:
            # Found at least one parameter with gradient - test passes
            return
    
    # If we reach here, no gradients were found
    pytest.fail("No gradients were computed during backpropagation")