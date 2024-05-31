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


def load_data(
    batch_size: int = 128, vit_16_using: bool = False
) -> Tuple[DataLoader, DataLoader]:
    if vit_16_using:
        transform: transforms.Compose = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    else:
        transform: transforms.Compose = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
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


def initialize_model(
    model_name: str, device: torch.device, pretrained: bool = True
) -> Module:
    if model_name == "vgg16":
        if pretrained:
            model: Module = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        else:
            model: Module = models.vgg16(weights=None)
    elif model_name == "resnet18":
        if pretrained:
            model: Module = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            model: Module = models.resnet18(weights=None)
    elif model_name == "mobilenet_v3_large":
        if pretrained:
            model: Module = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.DEFAULT
            )
        else:
            model: Module = models.mobilenet_v3_large(weights=None)
    elif model_name == "vit_b_16":
        if pretrained:
            model: Module = models.vit_b_16(
                weights=models.ViT_B_16_Weights.IMAGENET1K_V1
            )
        else:
            model: Module = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, 10)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Move the model to the specified device
    model.to(device)

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
    last_layer_name: str = "heads.head"
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name != last_layer_name:
            prune_idxs = list(range(0, int(module.out_channels * pruning_factor)))
            group = DG.get_pruning_group(
                module, tp.prune_conv_out_channels, idxs=prune_idxs
            )
            if DG.check_pruning_group(group):
                group.prune()
        elif isinstance(module, nn.Linear) and name != last_layer_name:
            prune_idxs = list(range(0, int(module.out_features * pruning_factor)))
            group = DG.get_pruning_group(
                module, tp.prune_linear_out_channels, idxs=prune_idxs
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
    # Create the base directory if it does not exist
    base_dir: str = "model_variants"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created directory: {base_dir}")

    # Create the model-specific directory if it does not exist
    model_dir: str = os.path.join(base_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")

    # Determine the filename based on the pruning factor
    if pruning_factor == 0:
        file_name: str = f"{model_name}_original.pth"
    else:
        file_name: str = f"{model_name}_pruned_{pruning_factor}.pth"

    # Full path to save the model
    save_path: str = os.path.join(model_dir, file_name)

    # Save the model
    try:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")
    except Exception as e:
        print(f"Error saving the model: {e}")


def main(model_name: str, pruning_factor: float, epochs: int, iterations: int):
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader: DataLoader
    testloader: DataLoader
    trainloader, testloader = load_data(
        batch_size=128, vit_16_using="vit" in model_name
    )
    model: Module = initialize_model(model_name, device)
    criterion: CrossEntropyLoss = CrossEntropyLoss()
    if "ViT" in model_name:
        optimizer: Optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    else:
        optimizer: Optimizer = torch.optim.SGD(
            model.parameters(), lr=0.001, momentum=0.9
        )
    if not np.isclose(pruning_factor, 0):
        train_and_prune(
            model,
            trainloader,
            criterion,
            optimizer,
            device,
            iterative_steps=iterations,
            pruning_factor_total=pruning_factor,
        )
    fine_tune_model(model, trainloader, criterion, optimizer, device, epochs=epochs)
    accuracy: float = evaluate(model, testloader, device)
    print(
        f"Pruned Model Accuracy for {model_name} with pruning factor {pruning_factor}: {accuracy}%"
    )
    model_saving(model, model_name, pruning_factor)
    del model
    torch.cuda.empty_cache()
    sys.stdout.flush()


if __name__ == "__main__":
    args: List[str] = sys.argv
    if len(args) != 5:
        raise ValueError(
            "Please provide the model name, pruning factor, epochs, and iterations."
        )
    model_name: str = args[1]
    pruning_factor: float = float(args[2])
    epochs: int = int(args[3])
    iterations: int = int(args[4])
    main(model_name, pruning_factor, epochs, iterations)
