import os
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.data.download import load_config as load_data_config
from src.data.download import load_price_data
from src.data.features import (
    fit_transform_train_features,
    split_by_date,
    transform_split_features,
)
from src.models.train_ppo import DATA_CONFIG_PATH, TRAIN_CONFIG_PATH, load_train_config


MULTI_SEED_CONFIG_PATH = os.path.join("configs", "multi_seed.yaml")


def softmax(x: np.ndarray) -> np.ndarray:
    # convert raw model outputs into a proper allocation vector
    # this mirrors the environment logic and guarantees valid nonnegative weights that sum to 1
    x = np.asarray(x, dtype=np.float64)

    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    weights = exp_x / np.sum(exp_x)

    return weights.astype(np.float64)


def load_multi_seed_config(config_path: str = MULTI_SEED_CONFIG_PATH) -> dict:
    # load the seed list so we know which independently trained models belong to the ensemble
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "seeds" not in cfg or not cfg["seeds"]:
        raise ValueError("multi-seed config must contain a non-empty 'seeds' list")

    return cfg


def build_model_paths(
    train_config_path: str = TRAIN_CONFIG_PATH,
    multi_seed_config_path: str = MULTI_SEED_CONFIG_PATH,
) -> dict[int, str]:
    # map each seed to its saved PPO artifact path
    train_cfg = load_train_config(train_config_path)
    multi_cfg = load_multi_seed_config(multi_seed_config_path)

    model_dir = train_cfg["artifacts"]["model_dir"]

    paths = {}
    for seed in multi_cfg["seeds"]:
        paths[seed] = os.path.join(model_dir, f"ppo_seed{seed}.zip")

    return paths


def load_ensemble_models(
    model_paths: dict[int, str],
    device: str = "cpu",
) -> dict[int, Any]:
    # load each independently trained PPO model
    # CPU is the safest choice for MLP inference in SB3
    models = {}

    for seed, path in model_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"model for seed {seed} not found at {path}")

        models[seed] = PPO.load(path, device=device)

    return models


def prepare_inference_splits(
    data_config_path: str = DATA_CONFIG_PATH,
    train_config_path: str = TRAIN_CONFIG_PATH,
) -> dict:
    # rebuild the same train/val/test splits and train-only scaling pipeline used in training
    # this keeps ensemble inference aligned with the experimental protocol
    data_cfg = load_data_config(data_config_path)
    train_cfg = load_train_config(train_config_path)

    prices = load_price_data(
        data_dir=data_cfg["data_dir"],
        output_file=data_cfg["output_file"],
    )

    train_prices, val_prices, test_prices = split_by_date(
        prices=prices,
        train_end=data_cfg["train_end"],
        val_end=data_cfg["val_end"],
    )

    scaler_path = os.path.join(
        data_cfg["artifacts_dir"],
        train_cfg["artifacts"]["scaler_name"],
    )

    # fit scaler on train only, then reuse it on val/test
    train_features, scaler = fit_transform_train_features(
        train_prices=train_prices,
        scaler_output_path=scaler_path,
    )
    val_features = transform_split_features(val_prices, scaler)
    test_features = transform_split_features(test_prices, scaler)

    train_prices = train_prices.loc[train_features.index]
    val_prices = val_prices.loc[val_features.index]
    test_prices = test_prices.loc[test_features.index]

    return {
        "train_prices": train_prices,
        "val_prices": val_prices,
        "test_prices": test_prices,
        "train_features": train_features,
        "val_features": val_features,
        "test_features": test_features,
    }


def predict_raw_action(model: Any, observation: np.ndarray) -> np.ndarray:
    # ask one PPO policy for its deterministic raw action vector
    raw_action, _ = model.predict(observation, deterministic=True)
    raw_action = np.asarray(raw_action, dtype=np.float64).reshape(-1)
    return raw_action


def predict_allocation(model: Any, observation: np.ndarray) -> np.ndarray:
    # convert a PPO model's raw action into valid portfolio weights
    raw_action = predict_raw_action(model, observation)
    return softmax(raw_action)


def compute_disagreement(member_allocations: np.ndarray) -> float:
    # disagreement is the average L2 distance from each member to the ensemble mean allocation
    # higher disagreement means lower consensus and therefore lower confidence
    member_allocations = np.asarray(member_allocations, dtype=np.float64)

    if member_allocations.ndim != 2:
        raise ValueError("member_allocations must be a 2D array")

    ensemble_mean = member_allocations.mean(axis=0)
    distances = np.linalg.norm(member_allocations - ensemble_mean, axis=1)

    return float(distances.mean())


def disagreement_to_confidence(
    disagreement: float,
    alpha: float = 10.0,
) -> float:
    # map disagreement into [0, 1] using a smooth exponential decay
    # lower disagreement -> confidence closer to 1
    if disagreement < 0:
        raise ValueError("disagreement must be nonnegative")

    confidence = float(np.exp(-alpha * disagreement))
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return confidence


def ensemble_inference(
    models: dict[int, Any],
    observation: np.ndarray,
    confidence_alpha: float = 10.0,
) -> dict:
    # run all ensemble members on the same state, then aggregate allocations and uncertainty
    if not models:
        raise ValueError("models dictionary is empty")

    member_allocations = []
    member_raw_actions = {}
    member_weights = {}

    for seed, model in models.items():
        raw_action = predict_raw_action(model, observation)
        allocation = softmax(raw_action)

        member_raw_actions[seed] = raw_action
        member_weights[seed] = allocation
        member_allocations.append(allocation)

    member_allocations = np.asarray(member_allocations, dtype=np.float64)

    ensemble_weights = member_allocations.mean(axis=0)
    disagreement = compute_disagreement(member_allocations)
    confidence = disagreement_to_confidence(
        disagreement=disagreement,
        alpha=confidence_alpha,
    )

    return {
        "ensemble_weights": ensemble_weights,
        "member_weights": member_weights,
        "member_raw_actions": member_raw_actions,
        "disagreement": disagreement,
        "confidence": confidence,
    }


def infer_on_split_row(
    models: dict[int, Any],
    features: pd.DataFrame,
    row_idx: int = -1,
    previous_weights: np.ndarray | None = None,
    confidence_alpha: float = 10.0,
) -> dict:
    # build one full observation row for ensemble inference:
    # market features + previous risky weights + previous cash weight
    if features.empty:
        raise ValueError("features dataframe is empty")

    feature_row = features.iloc[row_idx].to_numpy(dtype=np.float64)

    # default portfolio context = all cash
    if previous_weights is None:
        n_assets = 5
        previous_weights = np.zeros(n_assets + 1, dtype=np.float64)
        previous_weights[-1] = 1.0

    previous_weights = np.asarray(previous_weights, dtype=np.float64)

    observation = np.concatenate([feature_row, previous_weights])

    result = ensemble_inference(
        models=models,
        observation=observation.astype(np.float32),
        confidence_alpha=confidence_alpha,
    )

    result["observation"] = observation
    result["timestamp"] = features.index[row_idx]

    return result


def pretty_allocation_dict(
    weights: np.ndarray,
    asset_names: list[str],
) -> dict[str, float]:
    # convert allocation vector into a readable mapping including cash
    weights = np.asarray(weights, dtype=np.float64)

    if len(weights) != len(asset_names) + 1:
        raise ValueError("weights length must equal len(asset_names) + 1")

    labels = asset_names + ["CASH"]
    return {label: float(weight) for label, weight in zip(labels, weights)}