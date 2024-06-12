import subprocess
import sys
import os
from typing import List, Tuple
from typing import Iterator

epochs: int = 100
transformer_epochs: int = 20
iterations_in_pruning: int = 20
batch_size: int = 256
transformer_batch_size: int = 64
models_name: List[str] = ["vit_b_16", "resnet50", "vgg16", "mobilenet_v3_large"]
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


pruning_factors: List[float] = list(float_range(0.05, 1.0, 0.2))
pruning_factors = list(map(lambda x: round(x, 2), pruning_factors))


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
