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
import csv

base_directory: str = os.path.join(os.getcwd(), "model_variants")
models_name: List[str] = ["vit_b_16", "resnet50", "vgg16", "mobilenet_v3_large"]
test_batch_size: int = 256


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


def run_model_test(
    batch_size: int, directory: str, model_name: str
) -> List[Dict[str, Any]]:
    model_infos: List[Dict[str, Any]] = []

    try:

        model_files: List[str] = find_model_files(directory)

        if not model_files:
            raise FileNotFoundError(
                f"No model files with the specified extension were found in {directory}"
            )
        model_path: str
        for model_path in model_files:
            # Define the command to run the subprocess
            command = [
                "python",
                "python_model_test.py",
                str(batch_size),
                model_path,
                model_name,
            ]

            # Use 'shell=True' for Windows to handle command execution correctly
            shell: bool = os.name == "nt"

            # Run the command as a subprocess
            result: subprocess.CompletedProcess = subprocess.run(
                command, check=True, shell=shell
            )
            print(f"Output:\n{result.stdout}")
            if result.returncode == 0:
                print(f"Subprocess for {model_path} ended successfully.")
                with open("model_info.json", "r") as json_file:
                    model_info: Dict[str, Any] = json.load(json_file)
                model_infos.append(model_info)
            else:
                print(f"Subprocess for {model_path} ended with errors.")
                model_infos.append(None)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        model_infos.append(None)

    except subprocess.CalledProcessError as e:
        print(f"Subprocess error: {e}")
        model_infos.append(None)

    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}")
        model_infos.append(None)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        model_infos.append(None)

    return model_infos


def save_to_csv(
    all_model_infos: Dict[str, List[Dict[str, Any]]], output_file: str
) -> None:
    fieldnames = ["model", "prune_rate", "accuracy", "inference_time", "model_size"]
    with open(output_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, infos in all_model_infos.items():
            info: Dict[str, Any]
            for info in infos:
                if info:
                    writer.writerow(info)


if __name__ == "__main__":

    all_model_infos: Dict[str, List[Dict[str, Any]]] = {}
    model_name: str
    for model_name in models_name:
        directory: str = os.path.join(base_directory, model_name)
        model_infos: List[Dict[str, Any]] = run_model_test(
            test_batch_size, directory, model_name
        )
        all_model_infos[model_name] = model_infos
    output_file: str = "model_information.csv"
    save_to_csv(all_model_infos, output_file)
    for model_name, infos in all_model_infos.items():
        print(f"Model: {model_name}")
        for idx, info in enumerate(infos):
            print(f"Model {idx + 1} info: {info}")
