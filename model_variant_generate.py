import torch
import torch.nn as nn
import torchvision
import torch_pruning as tp
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn import CrossEntropyLoss
from typing import Tuple, List
import numpy as np
import os
import sys


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
    )  # train=True: Loads the training set of the CIFAR-10 dataset.

    testset: datasets.CIFAR10 = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader: DataLoader = DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2
    )

    return trainloader, testloader


def initialize_model(model_name: str, device: torch.device) -> Module:
    if model_name == "vgg16":
        model: Module = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device)
    elif model_name == "resnet18":
        model: Module = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(
            device
        )
    elif model_name == "mobilenet_v3_large":
        model: Module = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        ).to(device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model


def build_dependency_graph(
    model: Module, example_inputs: torch.Tensor
) -> tp.DependencyGraph:
    DG: tp.DependencyGraph = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=example_inputs)
    return DG


def prune_model_with_depgraph(
    DG: tp.DependencyGraph, model: Module, pruning_factor: float, device: torch.device
) -> None:
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune_idxs = list(range(0, int(module.out_channels * pruning_factor)))
            group = DG.get_pruning_group(
                module, tp.prune_conv_out_channels, idxs=prune_idxs
            )
            if DG.check_pruning_group(group):
                group.prune()


def train_and_prune(
    model: Module,
    trainloader: DataLoader,
    criterion: CrossEntropyLoss,
    optimizer: Optimizer,
    device: torch.device,
    iterative_steps: int,
    pruning_factor_total: float,
) -> None:
    pruning_factor: float = 1 - np.power(1 - pruning_factor_total, 1 / iterative_steps)
    print(f"Pruning Factor: {pruning_factor}")
    sample_inputs: torch.Tensor
    sample_inputs, _ = next(iter(trainloader))
    sample_inputs = sample_inputs.to(device)
    DG: tp.DependencyGraph = build_dependency_graph(model, sample_inputs)
    for step in range(iterative_steps):
        model.train()
        inputs: torch.Tensor
        targets: torch.Tensor
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss: torch.Tensor = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        prune_model_with_depgraph(DG, model, pruning_factor, device)
        macs, params = tp.utils.count_ops_and_params(model, inputs)
        print(f"Step {step+1}: MACs = {macs}, Params = {params}")
        sample_inputs: torch.Tensor
        sample_inputs, _ = next(iter(trainloader))
        sample_inputs = sample_inputs.to(device)
        DG: tp.DependencyGraph = build_dependency_graph(model, sample_inputs)
    print("Finished Training and Pruning the model")


def fine_tune_model(
    model: Module,
    trainloader: DataLoader,
    criterion: CrossEntropyLoss,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int = 5,
) -> None:
    for epoch in range(epochs):
        model.train()
        running_loss: float = 0.0
        inputs: torch.Tensor
        targets: torch.Tensor
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss: torch.Tensor = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}")
    print("Finished Fine-tuning the model")


def evaluate(model: Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct: int = 0
    total: int = 0
    with torch.no_grad():
        inputs: torch.Tensor
        targets: torch.Tensor
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs: torch.Tensor = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def model_saving(model: Module, model_name: str, pruning_factor: float) -> None:
    if not os.path.exists("model_variants"):
        os.makedirs("model_variants")

    if not os.path.exists(f"model_variants/{model_name}"):
        os.makedirs(f"model_variants/{model_name}")

    # if pruning_factor is 0, then it is the original model
    if pruning_factor == 0:
        torch.save(model, f"model_variants/{model_name}/{model_name}_original.pth")
    else:
        torch.save(
            model,
            f"model_variants/{model_name}/{model_name}_pruned_{pruning_factor}.pth",
        )


class CandidateModel:
    def __init__(self, model: Module, size: int, prune_rate: float, name: str):
        self.model = model
        self.size = size
        self.prune_rate = prune_rate
        self.name = name


def models_read(
    path: str, model_name: str, device: torch.device
) -> List[CandidateModel]:
    models: List[CandidateModel] = []
    for file in os.listdir(path):
        if file.endswith(".pth"):
            model = torch.load(os.path.join(path, file))
            size = os.path.getsize(os.path.join(path, file))  # size in bytes
            # if it is the original model
            if "original" in file:
                prune_rate = 0
            else:
                prune_rate = float(file.split("_")[-1].split(".")[0])
            models.append(CandidateModel(model, size, prune_rate, model_name))

    return models


def model_read(path: str, model_name: str, device: torch.device) -> CandidateModel:
    model = torch.load(path)
    size = os.path.getsize(path)
    if "original" in path:
        prune_rate = 0
    else:
        prune_rate = float(path.split("_")[-1].split(".")[0])
    return CandidateModel(model, size, prune_rate, model_name)


def main(model_name: str, pruning_factor: float):
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader: DataLoader
    testloader: DataLoader
    trainloader, testloader = load_data()
    model: Module = initialize_model(model_name, device)
    criterion: CrossEntropyLoss = CrossEntropyLoss()
    # we use self-adaptive learning rate
    optimizer: Optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    if not np.isclose(pruning_factor, 0):
        train_and_prune(
            model,
            trainloader,
            criterion,
            optimizer,
            device,
            iterative_steps=int(
                5 * (pruning_factor / 0.05)
            ),  # 5 steps for each 0.05 pruning factor
            pruning_factor_total=pruning_factor,
        )
    fine_tune_model(model, trainloader, criterion, optimizer, device, epochs=100)
    accuracy: float = evaluate(model, testloader, device)
    print(
        f"Pruned Model Accuracy for {model_name} with pruning factor {pruning_factor}: {accuracy}%"
    )
    model_saving(model, model_name, pruning_factor)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args: List[str] = sys.argv
    if len(args) != 3:
        raise ValueError("Please provide model name and pruning factor")
    model_name: str = args[1]
    pruning_factor: float = float(args[2])
    main(model_name, pruning_factor)
