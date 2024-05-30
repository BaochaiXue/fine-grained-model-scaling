import os
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn import Module
from torch.optim import Optimizer
from torch_pruning import DependencyGraph
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


def load_data(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    transform: transforms.Compose = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset: datasets.CIFAR10 = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader: DataLoader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset: datasets.CIFAR10 = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader: DataLoader = DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2
    )

    return trainloader, testloader
