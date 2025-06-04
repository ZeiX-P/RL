import pytest
from torch import nn

from core.train_params import TrainingParams
from optim.ssgd import SparseSGDM


def test_invalid_optimizer_class():
    model = nn.Linear(10, 5)
    loss_function = nn.MSELoss()

    with pytest.raises(TypeError):
        TrainingParams(
            training_name="test_training",
            epochs=10,
            learning_rate=0.001,
            model=model,
            loss_function=loss_function,
            optimizer_class=nn.Linear,  # type: ignore # Invalid optimizer class
            optimizer_params={},
        )


def test_train_params_sparsesgd():
    model = nn.Linear(10, 5)
    loss_function = nn.MSELoss()

    with pytest.raises(ValueError):
        TrainingParams(
            training_name="",
            epochs=0,
            learning_rate=1e-3,
            model=model,
            loss_function=loss_function,
            optimizer_class=SparseSGDM,  # type: ignore
        )
    with pytest.raises(ValueError):
        TrainingParams(
            training_name="",
            epochs=0,
            learning_rate=1e-3,
            model=model,
            loss_function=loss_function,
            optimizer_class=SparseSGDM,  # type: ignore
            optimizer_params={"named_params": {}},
        )
    params = TrainingParams(
        training_name="",
        epochs=0,
        learning_rate=1e-3,
        model=model,
        loss_function=loss_function,
        optimizer_class=SparseSGDM,  # type: ignore
        optimizer_params={"named_params": {}, "grad_mask": {}},
    )
    assert params.epochs == 0  # instantiation succeeds
