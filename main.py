import subprocess
import sys
import os
from typing import List, Tuple
from typing import Iterator, Any, Dict
import json

epochs: int = 150
transformer_epochs: int = 80
iterations_in_pruning: int = 20
batch_size: int = 256
transformer_batch_size: int = 64
model_gen_script: str = "model_variant_generate.py"
S: float = 1000
K: int = 5


def float_range(start: float, end: float, step: float) -> Iterator[float]:
    if step == 0:
        raise ValueError("Step cannot be zero.")
    if (step > 0 and start >= end) or (step < 0 and start <= end):
        return  # Exits if the range is not feasible with the given step
    while (step > 0 and start < end) or (step < 0 and start > end):
        yield start
        start += step


def read_config_from_json(
    json_path: str = "config.json",
) -> Tuple[List[str], List[float], str]:
    try:
        with open(json_path, "r") as file:
            data: Dict[str, Any] = json.load(file)
            models_name: List[str] = data.get("models", [])
            prune_factors_range: List[float] = data.get("prune_factors", [])
            if len(prune_factors_range) != 3:
                raise ValueError(
                    "Pruning factors should contain three values: start, end, and step."
                )
            pruning_factors: List[float] = list(
                float_range(*prune_factors_range)
                if prune_factors_range
                else [0.05, 1.0, 0.05]
            )
            pruning_factors = list(map(lambda x: round(x, 2), pruning_factors))
            if not models_name:
                raise ValueError("No models found in the JSON file.")

            return models_name, pruning_factors, data.get("dataset", "CIFAR10")
    except Exception as e:
        print(f"An error occurred while reading the JSON file: {str(e)}")
        sys.exit(1)


models_name: List[str]
pruning_factors: List[float]
models_name, pruning_factors, _ = read_config_from_json("config.json")


def call_generate_model(path: str, *args) -> None:
    try:
        python_exec: str = "python3" if sys.executable.endswith("python3") else "python"
        command: List[str] = [python_exec, path, *args]
        result: subprocess.CompletedProcess = subprocess.run(
            command, capture_output=True, text=True, check=True
        )
        if result.returncode == 0:
            print("Subprocess ended successfully.")
        else:
            print("Subprocess ended with errors.")
        print(f"Output:\n{result.stdout}")
        print(f"Error (if any):\n{result.stderr}")

    except subprocess.CalledProcessError as e:
        # Handle exceptions if the subprocess returns a non-zero exit status
        print(f"Script failed with error:\n{e.stderr}")
        print(f"Return code: {e.returncode}")
    except Exception as e:
        # Handle unexpected exceptions
        print(f"An unexpected error occurred: {str(e)}")


def call_storage_bound_model_selection(S: float, K: int) -> None:
    try:
        # Call the storage_bound_model_selection.py script with S and K as arguments
        result = subprocess.run(
            ["python", "storage_bound_model_selection.py", str(S), str(K)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("Subprocess Output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("An error occurred while running the script:")
        print(e.stderr)


if __name__ == "__main__":
    model_name: str
    pruning_ratio: float
    for model_name in models_name:
        call_generate_model(
            model_gen_script,
            model_name,
            "0.0",
            str(epochs) if "vit" not in model_name else str(transformer_epochs),
            str(iterations_in_pruning),
            (
                str(batch_size)
                if "vit" not in model_name
                else str(transformer_batch_size)
            ),
        )
        for pruning_ratio in pruning_factors:
            print(f"Pruning Factor: {pruning_ratio}")
            call_generate_model(
                model_gen_script,
                model_name,
                str(pruning_ratio * iterations_in_pruning),
                (
                    str(round(epochs * (1 + pruning_ratio)))
                    if "vit" not in model_name
                    else str(round(transformer_epochs * (1 + pruning_ratio)))
                ),
                str(iterations_in_pruning),
                (
                    str(batch_size)
                    if "vit" not in model_name
                    else str(transformer_batch_size)
                ),
            )
    call_storage_bound_model_selection(S, K)
    print("Model selection process completed.")
