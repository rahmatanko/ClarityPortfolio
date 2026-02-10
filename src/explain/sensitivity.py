import numpy as np
import pandas as pd

from src.data.features import get_feature_groups
from src.models.ensemble import ensemble_inference


def validate_observation_and_features(
    observation: np.ndarray,
    features: pd.DataFrame,
    row_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    # make sure the observation really matches the chosen feature row
    # this prevents subtle bugs where we explain the wrong state
    if features.empty:
        raise ValueError("features dataframe is empty")

    observation = np.asarray(observation, dtype=np.float64).reshape(-1)

    feature_row = features.iloc[row_idx].to_numpy(dtype=np.float64)

    if len(observation) < len(feature_row):
        raise ValueError("observation is shorter than feature row")

    observation_feature_part = observation[: len(feature_row)]

    if not np.allclose(observation_feature_part, feature_row, atol=1e-8):
        raise ValueError("observation feature segment does not match selected feature row")

    return observation, feature_row


def build_feature_group_indices(
    features: pd.DataFrame,
    asset_names: list[str],
) -> dict[str, list[int]]:
    # map human-readable feature groups to column indices in the feature matrix
    # this lets us perturb whole groups like volatility or long-term returns together
    expected_groups = get_feature_groups(asset_names)

    col_to_idx = {col: idx for idx, col in enumerate(features.columns)}

    grouped_indices = {}
    for group_name, cols in expected_groups.items():
        missing = [col for col in cols if col not in col_to_idx]
        if missing:
            raise ValueError(f"missing expected feature columns for group '{group_name}': {missing}")

        grouped_indices[group_name] = [col_to_idx[col] for col in cols]

    return grouped_indices


def perturb_feature_group(
    observation: np.ndarray,
    feature_indices: list[int],
    delta: float,
) -> np.ndarray:
    # create a local perturbation by shifting a selected feature group
    # we only perturb the feature part of the observation, not portfolio context
    perturbed = np.asarray(observation, dtype=np.float64).copy()

    for idx in feature_indices:
        perturbed[idx] += delta

    return perturbed.astype(np.float32)


def allocation_shift_l1(
    base_weights: np.ndarray,
    perturbed_weights: np.ndarray,
) -> float:
    # simple, interpretable effect size:
    # how much total portfolio mass moved because of the perturbation
    base_weights = np.asarray(base_weights, dtype=np.float64)
    perturbed_weights = np.asarray(perturbed_weights, dtype=np.float64)

    if base_weights.shape != perturbed_weights.shape:
        raise ValueError("weight vectors must have the same shape")

    return float(np.sum(np.abs(perturbed_weights - base_weights)))


def allocation_shift_by_asset(
    base_weights: np.ndarray,
    perturbed_weights: np.ndarray,
    asset_names: list[str],
) -> dict[str, float]:
    # report signed per-asset change so later explanations can say
    # which allocations rose or fell when a feature group was perturbed
    base_weights = np.asarray(base_weights, dtype=np.float64)
    perturbed_weights = np.asarray(perturbed_weights, dtype=np.float64)

    labels = asset_names + ["CASH"]

    if len(base_weights) != len(labels):
        raise ValueError("weight vector length does not match asset_names + CASH")

    return {
        label: float(perturbed_weights[i] - base_weights[i])
        for i, label in enumerate(labels)
    }


def sensitivity_analysis(
    models: dict,
    observation: np.ndarray,
    features: pd.DataFrame,
    asset_names: list[str],
    row_idx: int = -1,
    perturbation_delta: float = 0.1,
    confidence_alpha: float = 10.0,
) -> dict:
    # run group-wise local sensitivity analysis around one observation
    # this is the core policy-linked explanation mechanism
    if not models:
        raise ValueError("models dictionary is empty")

    observation, _ = validate_observation_and_features(observation, features, row_idx)
    group_indices = build_feature_group_indices(features, asset_names)

    base_result = ensemble_inference(
        models=models,
        observation=observation.astype(np.float32),
        confidence_alpha=confidence_alpha,
    )

    base_weights = base_result["ensemble_weights"]

    group_effects = []

    for group_name, indices in group_indices.items():
        perturbed_obs = perturb_feature_group(
            observation=observation,
            feature_indices=indices,
            delta=perturbation_delta,
        )

        perturbed_result = ensemble_inference(
            models=models,
            observation=perturbed_obs,
            confidence_alpha=confidence_alpha,
        )

        perturbed_weights = perturbed_result["ensemble_weights"]

        effect_size = allocation_shift_l1(
            base_weights=base_weights,
            perturbed_weights=perturbed_weights,
        )

        per_asset_shift = allocation_shift_by_asset(
            base_weights=base_weights,
            perturbed_weights=perturbed_weights,
            asset_names=asset_names,
        )

        group_effects.append(
            {
                "group": group_name,
                "effect_size": effect_size,
                "per_asset_shift": per_asset_shift,
                "perturbed_confidence": float(perturbed_result["confidence"]),
                "perturbed_disagreement": float(perturbed_result["disagreement"]),
            }
        )

    ranked_effects = sorted(
        group_effects,
        key=lambda x: x["effect_size"],
        reverse=True,
    )

    return {
        "timestamp": features.index[row_idx],
        "base_weights": base_weights,
        "base_confidence": float(base_result["confidence"]),
        "base_disagreement": float(base_result["disagreement"]),
        "ranked_group_effects": ranked_effects,
    }


def top_k_sensitivity_groups(
    sensitivity_result: dict,
    k: int = 3,
) -> list[dict]:
    # helper to pull out the most important feature groups
    ranked = sensitivity_result["ranked_group_effects"]

    if k <= 0:
        raise ValueError("k must be positive")

    return ranked[:k]


def summarize_group_direction(
    per_asset_shift: dict[str, float],
    threshold: float = 1e-4,
) -> dict[str, list[str]]:
    # convert raw signed shifts into a compact directional summary
    increased = []
    decreased = []

    for asset, delta in per_asset_shift.items():
        if delta > threshold:
            increased.append(asset)
        elif delta < -threshold:
            decreased.append(asset)

    return {
        "increased": increased,
        "decreased": decreased,
    }


def build_explanation_payload(
    sensitivity_result: dict,
    top_k: int = 3,
    direction_threshold: float = 1e-4,
) -> dict:
    # create a clean explanation object
    top_groups = top_k_sensitivity_groups(sensitivity_result, k=top_k)

    summary_groups = []
    for item in top_groups:
        direction = summarize_group_direction(
            per_asset_shift=item["per_asset_shift"],
            threshold=direction_threshold,
        )

        summary_groups.append(
            {
                "group": item["group"],
                "effect_size": float(item["effect_size"]),
                "increased_assets": direction["increased"],
                "decreased_assets": direction["decreased"],
            }
        )

    return {
        "timestamp": sensitivity_result["timestamp"],
        "base_confidence": float(sensitivity_result["base_confidence"]),
        "base_disagreement": float(sensitivity_result["base_disagreement"]),
        "top_groups": summary_groups,
    }