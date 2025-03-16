"""

    Prompt into chatGPT: uploaded files + " using pytest implement unit tests for my projects "

"""


import pytest
from unittest.mock import MagicMock, patch
import torch
from imports import get_device, DataLoader, ModelTrainer
from torchvision.models import resnet50

# 1. Test get_device function
def test_get_device():
    with patch("torch.cuda.is_available", return_value=True):
        assert str(get_device()) == "cuda", "Expected 'cuda' when GPU is available."
    with patch("torch.cuda.is_available", return_value=False):
        assert str(get_device()) == "cpu", "Expected 'cpu' when GPU is unavailable."

# 2. Test DataLoader functionality
def test_data_loader():
    data_loader = DataLoader()
    
    # Mock CIFAR10 dataset loading to avoid actual file downloads
    with patch("torchvision.datasets.CIFAR10") as mock_cifar10:
        # Mock training and test datasets
        mock_cifar10.side_effect = [
            [(torch.rand(3, 32, 32), torch.tensor(0))] * 100,  # Training data
            [(torch.rand(3, 32, 32), torch.tensor(1))] * 50,   # Test data
        ]
        
        train_loader, test_loader = data_loader.load_data()
        
        # Validate data loader outputs
        for inputs, labels in train_loader:
            assert inputs.shape == (64, 3, 32, 32), "Train batch data shape is incorrect."
            assert labels.shape == (64,), "Train batch labels shape is incorrect."
            break  # Only check the first batch for simplicity
        
        for inputs, labels in test_loader:
            assert inputs.shape == (64, 3, 32, 32), "Test batch data shape is incorrect."
            assert labels.shape == (64,), "Test batch labels shape is incorrect."
            break

# 3. Test ModelTrainer's training process
def test_model_trainer_training():
    # Initialize mock data loader
    mock_data_loader = MagicMock()
    mock_train_loader = [
        (torch.rand(64, 3, 32, 32), torch.randint(0, 10, (64,))) for _ in range(2)
    ]
    mock_test_loader = [
        (torch.rand(64, 3, 32, 32), torch.randint(0, 10, (64,))) for _ in range(1)
    ]
    mock_data_loader.load_data.return_value = (mock_train_loader, mock_test_loader)

    # Mock model
    model = resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 10)

    # Create ModelTrainer instance
    trainer = ModelTrainer(
        data_loader=mock_data_loader,
        model=model,
        device="cpu",
        learning_rate=0.01,
        num_epochs=1
    )

    # Mock logger and tensorboard
    trainer.logger.log_info = MagicMock()
    trainer.writer.add_scalar = MagicMock()

    # Train the model
    trainer.train()

    # Check if model parameters were updated
    initial_params = [p.clone() for p in model.parameters()]
    trainer.train()
    updated_params = [p for p in model.parameters()]
    
    assert not all(
        torch.equal(i, u) for i, u in zip(initial_params, updated_params)
    ), "Model parameters were not updated during training."

    # Verify logging
    trainer.logger.log_info.assert_called()
    trainer.writer.add_scalar.assert_called()
