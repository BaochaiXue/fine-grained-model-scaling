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
    "FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0)
)  # Minimize the third value
creator.create("Individual", list, fitness=creator.FitnessMulti)


def select_models(group: pd.DataFrame, K: int, S: float) -> pd.DataFrame:
    storage_min: float = group["model_size"].min()
    storage_max: float = group["model_size"].max()
    inference_min: float = group["inference_time"].min()
    inference_max: float = group["inference_time"].max()
    accuracy_min: float = group["accuracy"].min()
    accuracy_max: float = group["accuracy"].max()

    # Normalize the data using MinMax scaling
    group["model_size"] = (group["model_size"] - storage_min) / (
        storage_max - storage_min
    )
    group["inference_time"] = (group["inference_time"] - inference_min) / (
        inference_max - inference_min
    )
    group["accuracy"] = (group["accuracy"] - accuracy_min) / (
        accuracy_max - accuracy_min
    )

    def evaluate(
        individual: List[int], models_df: pd.DataFrame
    ) -> Tuple[float, float, float]:
        selected_models: pd.DataFrame = models_df.iloc[individual]
        total_storage: float = selected_models["model_size"].sum()
        min_inference_diff: float = np.diff(
            sorted(selected_models["inference_time"])
        ).min()
        accuracy_sum: float = selected_models["accuracy"].sum()
        return (min_inference_diff, accuracy_sum, total_storage)

    def feasible(individual: List[int], models_df: pd.DataFrame, S: float) -> bool:
        normalized_S: float = (S - storage_min) / (storage_max - storage_min)
        selected_models: pd.DataFrame = models_df.iloc[individual]
        total_storage: float = selected_models["model_size"].sum()
        if selected_models["accuracy"].min() < (15 - accuracy_min) / (
            accuracy_max - accuracy_min
        ):
            return False
        return total_storage <= normalized_S and len(set(individual)) == len(individual)

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

    feasibility_func: typing.Callable = lambda ind: feasible(ind, group, S)
    toolbox.decorate(
        "evaluate",
        tools.DeltaPenalty(
            feasibility_func, (-float("inf"), -float("inf"), float("inf"))
        ),
    )

    population: List[creator.Individual] = toolbox.population(n=1000)
    hof: tools.HallOfFame = tools.HallOfFame(1)
    algorithms.eaMuPlusLambda(
        population,
        toolbox,
        mu=100,
        lambda_=200,
        cxpb=0.7,
        mutpb=0.3,
        ngen=100,
        verbose=False,
        halloffame=hof,
    )

    best_individual: List[int] = hof[0]

    print(f"Best individual: {best_individual}")

    # Denormalize the data
    group["model_size"] = (
        group["model_size"] * (storage_max - storage_min) + storage_min
    )
    group["inference_time"] = (
        group["inference_time"] * (inference_max - inference_min) + inference_min
    )
    group["accuracy"] = group["accuracy"] * (accuracy_max - accuracy_min) + accuracy_min

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
    model_info["model_size"] = model_info["model_size"] / 1024 / 1024
    model_groups: typing.Dict[str, pd.DataFrame] = dict(
        tuple(model_info.groupby("model"))
    )

    selected_models_list: List[pd.DataFrame] = []

    for model, group in model_groups.items():
        selected_models: pd.DataFrame = select_models(group, K, S)
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
