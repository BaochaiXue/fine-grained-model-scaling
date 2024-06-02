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
import time
import sys
from model_variant_generate import initialize_model, load_data
from torchvision.models import VisionTransformer


class CandidateModel:
    def __init__(self, model: Module, size: int, prune_rate: float, name: str):
        self.model = model
        self.size = size
        self.prune_rate = prune_rate
        self.name = name


def model_read(path: str, model_name: str, device: torch.device) -> CandidateModel:
    model_state_dict: torch.load = torch.load(path)
    model: Module = initialize_model(model_name, False)
    model.load_state_dict(model_state_dict)
    # the size of model in bytes
    size: int = os.path.getsize(path)
    if "original" in path:
        prune_rate: float = 0.0
    else:
        prune_rate: float = float(path.split("_")[-1].split(".")[0])
    return CandidateModel(model, size, prune_rate, model_name)


def evaluate_and_measure_time(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    correct: int = 0
    total: int = 0
    start_time: float = time.time()

    with torch.no_grad():
        inputs: torch.Tensor
        targets: torch.Tensor
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs: torch.Tensor = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    end_time: float = time.time()
    inference_time: float = end_time - start_time
    accuracy: float = 100.0 * correct / total
    # clean up
    torch.cuda.empty_cache()
    return accuracy, inference_time


def main() -> None:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args: List[str] = sys.argv
    if len(args) != 4:
        raise ValueError(
            "Please provide the model name, pruning factor, epochs, and iterations."
        )
    batch_size: int = int(args[1])
    path: str = args[2]
    model_name: str = args[3]
    candidateModel: CandidateModel = model_read(path, model_name, device)
    model: Module = candidateModel.model
    model.to(device)
    testloader: DataLoader
    _, testloader = load_data(batch_size, isinstance(model, VisionTransformer))
    accuracy: float
    inference_time: float
    accuracy, inference_time = evaluate_and_measure_time(model, testloader, device)
    # decribe the model, from the model name and prune rate
    print(f"Model: {candidateModel.name}, Prune Rate: {candidateModel.prune_rate}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Inference Time: {inference_time:.4f} seconds")
    print(f"Model Size: {candidateModel.size} bytes")
    # print this information to the json file
    if not os.path.exists("model_info.json"):
        os.makedirs("model_info.json", exist_ok=True)
    with open("model_info.json", "w") as f:
        f.write(
            f'{{"model": "{candidateModel.name}", "prune_rate": {candidateModel.prune_rate}, "accuracy": {accuracy}, "inference_time": {inference_time}, "model_size": {candidateModel.size}}}'
        )


if __name__ == "__main__":
    main()
