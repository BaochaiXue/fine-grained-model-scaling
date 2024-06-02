import os
import sys
from typing import Any, Dict, Tuple, Callable, List
import torch
import torch.nn as nn
import torch_pruning as tp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights,
    vgg16,
    VGG16_Weights,
    resnet50,
    ResNet50_Weights,
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)
from torchvision.models.vision_transformer import VisionTransformer
import numpy as np
import torch.optim as optim


def load_data(batch_size: int, vit_16_using: bool) -> Tuple[DataLoader, DataLoader]:
    transform: Callable[[Any], Any] = (
        transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                ),
            ]
        )
        if vit_16_using
        else transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                ),
            ]
        )
    )

    trainset: datasets.CIFAR10 = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader: DataLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset: datasets.CIFAR10 = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader: DataLoader = DataLoader(
        testset, batch_size=batch_size * 2, shuffle=False
    )

    return trainloader, testloader


def initialize_model(model_name: str, pretrain: bool) -> nn.Module:
    weights: str
    model: nn.Module
    if model_name == "vit_b_16":
        weights = ViT_B_16_Weights.DEFAULT if pretrain else None
        model = vit_b_16(weights=weights)
    elif model_name == "vgg16":
        weights = VGG16_Weights.DEFAULT if pretrain else None
        model = vgg16(weights=weights)
    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrain else None
        model = resnet50(weights=weights)
    elif model_name == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrain else None
        model = mobilenet_v3_large(weights=weights)
    else:
        raise ValueError("Model not supported.")

    return model


def prune_model(
    model: nn.Module,
    example_inputs: torch.Tensor,
    output_transform: Callable[[Any], Any],
    model_name: str,
    pruning_ratio: float = 0.5,
    iterations: int = 1,
) -> None:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ori_size: int = tp.utils.count_params(model)
    model.to(device).eval()
    example_inputs = example_inputs.to(device)
    ignored_layers: List[nn.Module] = []
    for p in model.parameters():
        p.requires_grad_(True)  # Setting all parameters to be trainable
    # Ignoring unprunable modules based on model type
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m)
    channel_groups: Dict[nn.Module, int] = {}
    if isinstance(model, VisionTransformer):
        for m in model.modules():
            if isinstance(m, nn.MultiheadAttention):
                channel_groups[m] = m.num_heads

    importance = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        iterative_steps=iterations,
        pruning_ratio=pruning_ratio,
        global_pruning=True,
        round_to=None,
        ignored_layers=ignored_layers,
        channel_groups=channel_groups,
    )
    print("==============Before pruning=================")
    print("Model Name: {}".format(model_name))

    layer_channel_cfg: Dict[nn.Module, int] = {}
    for module in model.modules():
        if module not in pruner.ignored_layers:
            # print(module)
            if isinstance(module, nn.Conv2d):
                layer_channel_cfg[module] = module.out_channels
            elif isinstance(module, nn.Linear):
                layer_channel_cfg[module] = module.out_features

    pruner.step()
    if isinstance(
        model, VisionTransformer
    ):  # Torchvision relies on the hidden_dim variable for forwarding, so we have to modify this varaible after pruning
        model.hidden_dim = model.conv_proj.out_channels
        print(model.class_token.shape, model.encoder.pos_embedding.shape)
    # Pruning process

    print("==============After pruning=================")
    # Testing
    with torch.no_grad():
        if isinstance(example_inputs, dict):
            out = model(**example_inputs)
        else:
            out = model(example_inputs)
        if output_transform:
            out = output_transform(out)
        print("{} Pruning: ".format(model_name))
        params_after_prune: int = tp.utils.count_params(model)
        print("  Params: %s => %s" % (ori_size, params_after_prune))

        if isinstance(out, (dict, list, tuple)):
            print("  Output:")
            for o in tp.utils.flatten_as_list(out):
                print(o.shape)
        else:
            print("  Output:", out.shape)
        print("------------------------------------------------------\n")
    sys.stdout.flush()
    torch.cuda.empty_cache()


def model_saving(model: nn.Module, model_name: str, pruning_factor: float) -> None:
    base_dir: str = "model_variants"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created directory: {base_dir}")
    model_dir: str = os.path.join(base_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")
    if pruning_factor == 0:
        file_name: str = f"{model_name}_original.pth"
    else:
        file_name: str = f"{model_name}_pruned_{pruning_factor}.pth"
    save_path: str = os.path.join(model_dir, file_name)
    try:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")
    except Exception as e:
        print(f"Error saving the model: {e}")


def train(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Optimizer,
    epochs: int,
) -> None:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    model.to(device)
    for epoch in range(epochs):
        for i, data in enumerate(trainloader, 0):
            inputs: torch.Tensor
            labels: torch.Tensor
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs: torch.Tensor
            outputs = model(inputs)
            loss: torch.Tensor = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # print statistics
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
    print("Finished Training")
    # clean up
    torch.cuda.empty_cache()


def test(model: nn.Module, testloader: DataLoader) -> None:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    correct: int = 0
    total: int = 0
    with torch.no_grad():
        data: Tuple[torch.Tensor, torch.Tensor]
        for data in testloader:
            images: torch.Tensor
            labels: torch.Tensor
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs: torch.Tensor = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on the test images: {100 * correct / total}%")
    # clean up
    torch.cuda.empty_cache()


def main(model_name: str, pruning_ratio: float, epochs: int, iterations: int):
    trainloader: DataLoader
    testloader: DataLoader
    trainloader, testloader = load_data(128, "vit" in model_name)
    model = initialize_model(model_name, True)
    # randomly sample one as the example input
    example_inputs: torch.Tensor
    example_inputs, _ = next(iter(trainloader))
    if not np.isclose(pruning_ratio, 0):
        prune_model(model, example_inputs, None, model_name, pruning_ratio, iterations)
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: optim.Optimizer
    if isinstance(model, VisionTransformer):
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train(model, trainloader, criterion, optimizer, epochs)
    test(model, testloader)
    model_saving(model, model_name, pruning_ratio)


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
    sys.stdout.flush()
