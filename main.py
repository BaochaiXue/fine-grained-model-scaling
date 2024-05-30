import subprocess
import sys
import os
from typing import List, Tuple
from typing import Iterator

epochs: int = 1
iterations_in_pruning: int = 1
models_name: List[str] = ["vgg16", "resnet18", "mobilenet_v3_large"]
model_gen_script: str = "model_variant_generate.py"


def float_range(start: float, end: float, step: float) -> Iterator[float]:
    if step == 0:
        raise ValueError("Step cannot be zero.")
    if (step > 0 and start >= end) or (step < 0 and start <= end):
        return  # Exits if the range is not feasible with the given step
    while (step > 0 and start < end) or (step < 0 and start > end):
        yield start
        start += step


pruning_factors: List[float] = list(float_range(0.05, 1.0, 0.05))
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


if __name__ == "__main__":
    model_name: str
    pruning_factor: float
    for model_name in models_name:
        call_generate_model(
            model_gen_script, model_name, "0.0", str(epochs), str(iterations_in_pruning)
        )
        for pruning_factor in pruning_factors:
            print(f"Pruning Factor: {pruning_factor}")
            call_generate_model(
                model_gen_script,
                model_name,
                str(pruning_factor),
                str(epochs),
                str(iterations_in_pruning),
            )