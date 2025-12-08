# src/cv/dataset.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(train_dir, test_dir, batch_size=64, img_size=(48, 48), num_workers=2):
    """
    Returns train & test DataLoaders and emotion class names.

    Args:
        train_dir (str): Path to train dataset
        test_dir (str): Path to test dataset
        batch_size (int): Batch size for training/testing
        img_size (tuple): Resize dimensions (H, W)
        num_workers (int): Number of background workers for data loading

    Returns:
        train_loader, test_loader, class_names (list[str])
    """

    train_transform = transforms.Compose([
        transforms.Grayscale(),                     # ensure 1 channel
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),     # augmentation
        transforms.RandomRotation(10),              # ±10 degrees rotation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # normalize to [-1,1]
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    class_names = train_dataset.classes

    return train_loader, test_loader, class_names
