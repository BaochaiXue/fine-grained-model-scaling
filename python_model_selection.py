import pandas as pd
import typing
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import math
from deap import base, creator, tools, algorithms
import os
import shutil
import argparse

creator.create(
    "FitnessMulti", base.Fitness, weights=(-1.0, 20000.0, 2000.0)
)  # Minimize storage, maximize inference time difference and accuracy
creator.create("Individual", list, fitness=creator.FitnessMulti)


def select_models(group: pd.DataFrame, K: int, S: float) -> pd.DataFrame:
    def evaluate(
        individual: List[int], models_df: pd.DataFrame
    ) -> Tuple[float, float, float]:
        selected_models: pd.DataFrame = models_df.iloc[individual]
        total_storage: float = selected_models["model_size"].sum()
        min_inference_diff: float = np.min(
            np.diff(sorted(selected_models["inference_time"]))
        )
        accuracy_sum: float = selected_models["accuracy"].sum()
        # penalize for every model that has not an accuracy of at least 0.5
        for i in range(K):
            if selected_models.iloc[i]["accuracy"] < 0.5:
                accuracy_sum *= selected_models.iloc[i]["accuracy"]
        # bonus for inference time difference
        max_inference_diff: float = np.max(
            np.diff(sorted(selected_models["inference_time"]))
        )
        diff_ratio: float = min_inference_diff / max_inference_diff
        min_inference_diff *= 1 + diff_ratio
        return total_storage, min_inference_diff, accuracy_sum

    def feasible(individual: List[int], models_df: pd.DataFrame, S: float) -> bool:
        selected_models: pd.DataFrame = models_df.iloc[individual]
        total_storage: float = selected_models["model_size"].sum()
        return total_storage <= S and len(set(individual)) == len(individual)

    toolbox: base.Toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(len(group)), K)

    toolbox.register(
        "individual", tools.initIterate, creator.Individual, toolbox.indices
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, models_df=group)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    # Define the constraint
    feasibility_func = lambda ind: feasible(ind, group, S)
    toolbox.decorate(
        "evaluate",
        tools.DeltaPenalty(
            feasibility_func, (float("inf"), -float("inf"), -float("inf"))
        ),
    )

    population: List[creator.Individual] = toolbox.population(n=1000)
    algorithms.eaMuPlusLambda(
        population,
        toolbox,
        mu=1000,
        lambda_=2000,
        cxpb=0.7,
        mutpb=0.3,
        ngen=1000,
        verbose=False,
    )
    ind: creator.Individual
    population.sort(key=lambda ind: ind.fitness.values)

    best_individual: creator.Individual = population[0]
    best_models: pd.DataFrame = group.iloc[best_individual[:K]]

    return best_models


def copy_model(model_name: str, pruning_factor: float, selected_dir: str) -> None:
    base_dir: str = "model_variants"
    model_dir: str = os.path.join(base_dir, model_name)

    if pruning_factor == 0:
        file_name: str = f"{model_name}_original.pth"
    else:
        file_name: str = f"{model_name}_pruned_{pruning_factor}.pth"

    source_path: str = os.path.join(model_dir, file_name)
    target_model_dir: str = os.path.join(selected_dir, model_name)
    selected_save_path: str = os.path.join(target_model_dir, file_name)

    try:
        if not os.path.exists(source_path):
            print(f"Source model does not exist: {source_path}")
            return

        if not os.path.exists(target_model_dir):
            os.makedirs(target_model_dir)
            print(f"Created directory: {target_model_dir}")

        shutil.copyfile(source_path, selected_save_path)
        print(f"Model copied to: {selected_save_path}")

    except Exception as e:
        print(f"Error copying the model: {e}")


def main(S: float, K: int) -> None:
    model_info: pd.DataFrame = pd.read_csv("model_information.csv")
    model_info.set_index(["model", "prune_rate"], inplace=True)
    model_info.sort_values(by=["model", "inference_time"], inplace=True)
    model_info["model_size"] = model_info["model_size"] / 1024 / 1024
    model_groups: typing.Dict[str, pd.DataFrame] = dict(
        tuple(model_info.groupby("model"))
    )

    selected_models_list: List[pd.DataFrame] = []

    for model, group in model_groups.items():
        selected_models: pd.DataFrame = select_models(
            group,
            K,
            S,
        )
        selected_models_list.append(selected_models)

    all_selected_models: pd.DataFrame = pd.concat(selected_models_list)

    print("Selected Models:")
    print(all_selected_models)

    selected_dir: str = "selected_models"
    for idx, row in all_selected_models.iterrows():
        model_name: str = idx[0]
        pruning_factor: float = idx[1]
        copy_model(model_name, pruning_factor, selected_dir)


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run model selection process."
    )
    parser.add_argument("--S", type=float, required=True, help="Storage constraint")
    parser.add_argument(
        "--K", type=int, required=True, help="Number of models to select"
    )
    args: argparse.Namespace = parser.parse_args()
    main(S=args.S, K=args.K)
