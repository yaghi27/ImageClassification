import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision.datasets import CIFAR10
import Logger

class DataLoader:
    def __init__(self, fine_tuning = False):

        # Data Augmentation for training data
        if fine_tuning == False:
            self.train_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),  # Randomly crop to 32x32 with padding
                transforms.RandomHorizontalFlip(),    # Random horizontal flip
                transforms.RandomRotation(15),       # Random rotation
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
            ])
        else:
            self.train_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),  # Random crop with padding
                transforms.RandomHorizontalFlip(),    # Random horizontal flip
                transforms.RandomRotation(15),       # Random rotation
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
                transforms.RandomGrayscale(p=0.1),   # Random grayscale conversion
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Random perspective
                transforms.ToTensor(),               # Convert to tensor (must be before RandomErasing)
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0), # Apply erasing
            ])


        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
        ])


        self.logger = Logger.Logger('DataLoader')
        

    def load_data(self):

        self.logger.log_info("Loading data...")

        # Load data from CIFAR10 dataset directly from PyTorch
        train_dataset = CIFAR10(
            root='./data', train=True, download=True, transform=self.train_transforms
        )
        test_dataset = CIFAR10(
            root='./data', train=False, download=True, transform=self.test_transforms
        )

        train_loader = TorchDataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = TorchDataLoader(test_dataset, batch_size=64, shuffle=False)

        return train_loader, test_loader
