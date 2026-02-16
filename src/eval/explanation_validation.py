import json
import os
from typing import Any

import pandas as pd

from src.data.download import load_config as load_data_config
from src.data.download import load_price_data
from src.data.features import (
    build_feature_frame,
    fit_standard_scaler,
    transform_with_scaler,
)
from src.explain.sensitivity import build_explanation_payload, sensitivity_analysis
from src.models.ensemble import (
    build_model_paths,
    load_ensemble_models,
    pretty_allocation_dict,
)

OUTPUT_DIR = os.path.join("data", "artifacts", "runs")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "explanation_regime_validation.csv")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "explanation_regime_validation.json")


REGIMES = [
    {"label": "COVID Crash", "date": "2020-03-20"},
    {"label": "2022 Uncertainty", "date": "2022-06-17"},
    {"label": "2023 Recovery", "date": "2023-07-14"},
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def prepare_full_feature_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    # build one full historical feature matrix for regime-based explanation checks
    data_cfg = load_data_config("configs/data.yaml")
    prices = load_price_data(
        data_dir=data_cfg["data_dir"],
        output_file=data_cfg["output_file"],
    )

    raw_features = build_feature_frame(prices)
    aligned_prices = prices.loc[raw_features.index]

    # fit scaler on the earliest training portion only for consistency with methodology
    train_prices = prices.loc[prices.index < pd.Timestamp(data_cfg["train_end"])].copy()
    train_raw_features = build_feature_frame(train_prices)
    scaler = fit_standard_scaler(train_raw_features)

    scaled_features = transform_with_scaler(raw_features, scaler)

    return aligned_prices, scaled_features


def build_observation(
    features: pd.DataFrame,
    row_idx: int,
    n_assets: int,
) -> Any:
    feature_row = features.iloc[row_idx].to_numpy(dtype=float)

    previous_weights = [0.0] * (n_assets + 1)
    previous_weights[-1] = 1.0

    return list(feature_row) + previous_weights


def nearest_row_index(features: pd.DataFrame, target_date: str) -> int:
    ts = pd.Timestamp(target_date)
    idx = features.index.get_indexer([ts], method="nearest")[0]
    return int(idx)


def validate_explanations_across_regimes(
    confidence_alpha: float = 10.0,
    perturbation_delta: float = 0.1,
    top_k: int = 3,
) -> dict:
    prices, features = prepare_full_feature_data()
    asset_names = list(prices.columns)

    model_paths = build_model_paths()
    models = load_ensemble_models(model_paths, device="cpu")

    rows = []

    for regime in REGIMES:
        row_idx = nearest_row_index(features, regime["date"])
        timestamp = features.index[row_idx]

        observation = build_observation(features, row_idx, n_assets=len(asset_names))

        from src.models.ensemble import ensemble_inference
        ensemble_result = ensemble_inference(
            models=models,
            observation=observation,
            confidence_alpha=confidence_alpha,
        )

        sensitivity_result = sensitivity_analysis(
            models=models,
            observation=observation,
            features=features,
            asset_names=asset_names,
            row_idx=row_idx,
            perturbation_delta=perturbation_delta,
            confidence_alpha=confidence_alpha,
        )

        payload = build_explanation_payload(sensitivity_result, top_k=top_k)
        allocation = pretty_allocation_dict(
            ensemble_result["ensemble_weights"],
            asset_names=asset_names,
        )

        top_allocs = sorted(allocation.items(), key=lambda x: x[1], reverse=True)[:3]
        top_groups = [g["group"] for g in payload["top_groups"]]

        rows.append(
            {
                "regime": regime["label"],
                "requested_date": regime["date"],
                "actual_timestamp": str(timestamp.date()),
                "confidence": float(ensemble_result["confidence"]),
                "disagreement": float(ensemble_result["disagreement"]),
                "top_group_1": top_groups[0] if len(top_groups) > 0 else None,
                "top_group_2": top_groups[1] if len(top_groups) > 1 else None,
                "top_group_3": top_groups[2] if len(top_groups) > 2 else None,
                "top_alloc_1": top_allocs[0][0],
                "top_alloc_1_weight": float(top_allocs[0][1]),
                "top_alloc_2": top_allocs[1][0],
                "top_alloc_2_weight": float(top_allocs[1][1]),
                "top_alloc_3": top_allocs[2][0],
                "top_alloc_3_weight": float(top_allocs[2][1]),
                "allocation": allocation,
                "payload": payload,
            }
        )

    df = pd.DataFrame(rows)

    ensure_dir(OUTPUT_DIR)
    df.drop(columns=["allocation", "payload"]).to_csv(OUTPUT_CSV, index=False)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)

    return {
        "table": df,
        "csv_path": OUTPUT_CSV,
        "json_path": OUTPUT_JSON,
    }


if __name__ == "__main__":
    result = validate_explanations_across_regimes()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\n[INFO] explanation regime validation:")
    print(result["table"][[
        "regime",
        "actual_timestamp",
        "confidence",
        "disagreement",
        "top_group_1",
        "top_group_2",
        "top_group_3",
        "top_alloc_1",
        "top_alloc_1_weight",
        "top_alloc_2",
        "top_alloc_2_weight",
        "top_alloc_3",
        "top_alloc_3_weight",
    ]])

    print("\n[INFO] saved artifacts:")
    print(result["csv_path"])
    print(result["json_path"])