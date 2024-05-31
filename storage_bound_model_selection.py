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
import subprocess
import json
from typing import Any, Dict


def find_model_files(directory: str, extension: str = ".pth") -> List[str]:
    model_files: List[str] = []
    root: str
    files: List[str]
    file: str
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                model_files.append(os.path.join(root, file))
    return model_files


def run_model_test(batch_size: int, directory: str, model_name: str) -> Dict[str, Any]:
    """
    Run the model test script as a subprocess and extract the information from the resulting JSON file.

    Parameters:
    - batch_size (int): The number of samples per batch to load.
    - directory (str): The directory containing the model files.
    - model_name (str): The name of the model architecture to initialize.

    Returns:
    - Dict[str, Any]: A dictionary containing the model information.
    """
    model_files: List[str] = find_model_files(directory)

    if not model_files:
        raise FileNotFoundError(
            f"No model files with the specified extension were found in {directory}"
        )

    # Select the first found model file for testing
    model_path: str = model_files[0]

    # Define the command to run the subprocess
    command = [
        "python",
        "python_model_test.py",
        str(batch_size),
        model_path,
        model_name,
    ]

    # Use 'shell=True' for Windows to handle command execution correctly
    shell = os.name == "nt"

    # Run the command as a subprocess
    subprocess.run(command, check=True, shell=shell)

    # Read the resulting JSON file
    with open("model_info.json", "r") as json_file:
        model_info: Dict[str, Any] = json.load(json_file)

    return model_info


# Example usage
if __name__ == "__main__":
    batch_size: int = 64
    directory: str = os.path.join(os.getcwd(), "model_variants", "mobilenet_v3_large")
    model_name: str = "mobilenet_v3_large"

    model_info: Dict[str, Any] = run_model_test(batch_size, directory, model_name)
    print(model_info)
