import numpy as np
import pandas as pd

from src.data.features import build_feature_frame, fit_standard_scaler, transform_with_scaler
from src.explain.sensitivity import (
    allocation_shift_by_asset,
    allocation_shift_l1,
    build_explanation_payload,
    build_feature_group_indices,
    perturb_feature_group,
    sensitivity_analysis,
    summarize_group_direction,
    top_k_sensitivity_groups,
    validate_observation_and_features,
)


def make_sample_prices(n_rows: int = 80) -> pd.DataFrame:
    # deterministic fake price data so tests stay fast and predictable
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="D")

    aapl = np.linspace(100, 150, n_rows)
    msft = np.linspace(200, 260, n_rows) + 2 * np.sin(np.arange(n_rows))
    nvda = np.linspace(50, 120, n_rows) + 3 * np.cos(np.arange(n_rows))
    googl = np.linspace(90, 140, n_rows)
    tsla = np.linspace(40, 100, n_rows) + 5 * np.sin(np.arange(n_rows) / 2)

    return pd.DataFrame(
        {
            "AAPL": aapl,
            "MSFT": msft,
            "NVDA": nvda,
            "GOOGL": googl,
            "TSLA": tsla,
        },
        index=dates,
    )


def make_scaled_features(prices: pd.DataFrame) -> pd.DataFrame:
    features = build_feature_frame(prices)
    scaler = fit_standard_scaler(features)
    return transform_with_scaler(features, scaler)


class DummyModel:
    # deterministic stub model returning a preset raw action
    def __init__(self, raw_action):
        self.raw_action = np.asarray(raw_action, dtype=np.float32)

    def predict(self, observation, deterministic=True):
        return self.raw_action, None


def make_observation(features: pd.DataFrame, row_idx: int = 0) -> np.ndarray:
    # observation = feature row + previous portfolio context
    feature_row = features.iloc[row_idx].to_numpy(dtype=np.float64)

    previous_weights = np.zeros(6, dtype=np.float64)
    previous_weights[-1] = 1.0

    obs = np.concatenate([feature_row, previous_weights])
    return obs.astype(np.float32)


def test_validate_observation_and_features_accepts_matching_observation():
    prices = make_sample_prices()
    features = make_scaled_features(prices)
    obs = make_observation(features, row_idx=0)

    validated_obs, feature_row = validate_observation_and_features(obs, features, row_idx=0)

    assert validated_obs.shape[0] == len(obs)
    assert feature_row.shape[0] == features.shape[1]


def test_build_feature_group_indices_returns_expected_groups():
    prices = make_sample_prices()
    features = make_scaled_features(prices)

    groups = build_feature_group_indices(features, list(prices.columns))

    assert set(groups.keys()) == {
        "short_term_returns",
        "medium_term_returns",
        "long_term_returns",
        "volatility",
        "rsi",
    }

    assert len(groups["volatility"]) == 5
    assert len(groups["rsi"]) == 5


def test_perturb_feature_group_changes_only_selected_indices():
    obs = np.zeros(31, dtype=np.float32)
    perturbed = perturb_feature_group(obs, [1, 3, 5], delta=0.2)

    assert np.isclose(perturbed[1], 0.2)
    assert np.isclose(perturbed[3], 0.2)
    assert np.isclose(perturbed[5], 0.2)
    assert np.isclose(perturbed[0], 0.0)
    assert np.isclose(perturbed[2], 0.0)


def test_allocation_shift_l1_matches_manual_value():
    base = np.array([0.2, 0.3, 0.1, 0.1, 0.1, 0.2])
    perturbed = np.array([0.1, 0.4, 0.1, 0.1, 0.15, 0.15])

    shift = allocation_shift_l1(base, perturbed)

    expected = np.sum(np.abs(perturbed - base))
    assert np.isclose(shift, expected)


def test_allocation_shift_by_asset_returns_signed_mapping():
    base = np.array([0.2, 0.3, 0.1, 0.1, 0.1, 0.2])
    perturbed = np.array([0.1, 0.4, 0.1, 0.1, 0.15, 0.15])
    assets = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]

    shifts = allocation_shift_by_asset(base, perturbed, assets)

    assert set(shifts.keys()) == {"AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "CASH"}
    assert np.isclose(shifts["AAPL"], -0.1)
    assert np.isclose(shifts["MSFT"], 0.1)


def test_summarize_group_direction_separates_increases_and_decreases():
    direction = summarize_group_direction(
        {
            "AAPL": 0.02,
            "MSFT": -0.01,
            "NVDA": 0.0,
            "GOOGL": 0.000001,
            "TSLA": -0.03,
            "CASH": 0.015,
        },
        threshold=1e-4,
    )

    assert "AAPL" in direction["increased"]
    assert "CASH" in direction["increased"]
    assert "MSFT" in direction["decreased"]
    assert "TSLA" in direction["decreased"]


def test_sensitivity_analysis_returns_ranked_group_effects():
    prices = make_sample_prices()
    features = make_scaled_features(prices)
    obs = make_observation(features, row_idx=0)

    models = {
        42: DummyModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        43: DummyModel([0.2, 0.1, -0.1, 0.0, 0.3, -0.2]),
        44: DummyModel([-0.2, 0.4, 0.0, -0.1, 0.1, 0.2]),
    }

    result = sensitivity_analysis(
        models=models,
        observation=obs,
        features=features,
        asset_names=list(prices.columns),
        row_idx=0,
        perturbation_delta=0.1,
        confidence_alpha=10.0,
    )

    assert "base_weights" in result
    assert "base_confidence" in result
    assert "ranked_group_effects" in result
    assert len(result["ranked_group_effects"]) == 5

    effect_sizes = [item["effect_size"] for item in result["ranked_group_effects"]]
    assert effect_sizes == sorted(effect_sizes, reverse=True)


def test_top_k_sensitivity_groups_returns_requested_count():
    dummy_result = {
        "ranked_group_effects": [
            {"group": "a", "effect_size": 0.5},
            {"group": "b", "effect_size": 0.4},
            {"group": "c", "effect_size": 0.3},
        ]
    }

    top = top_k_sensitivity_groups(dummy_result, k=2)

    assert len(top) == 2
    assert top[0]["group"] == "a"
    assert top[1]["group"] == "b"


def test_build_explanation_payload_returns_compact_summary():
    dummy_result = {
        "timestamp": pd.Timestamp("2022-12-30"),
        "base_confidence": 0.2,
        "base_disagreement": 0.3,
        "ranked_group_effects": [
            {
                "group": "volatility",
                "effect_size": 0.5,
                "per_asset_shift": {
                    "AAPL": -0.02,
                    "MSFT": -0.01,
                    "NVDA": -0.03,
                    "GOOGL": 0.0,
                    "TSLA": -0.01,
                    "CASH": 0.07,
                },
            },
            {
                "group": "long_term_returns",
                "effect_size": 0.3,
                "per_asset_shift": {
                    "AAPL": 0.02,
                    "MSFT": 0.01,
                    "NVDA": 0.03,
                    "GOOGL": 0.0,
                    "TSLA": 0.01,
                    "CASH": -0.07,
                },
            },
        ],
    }

    payload = build_explanation_payload(dummy_result, top_k=2)

    assert payload["timestamp"] == pd.Timestamp("2022-12-30")
    assert np.isclose(payload["base_confidence"], 0.2)
    assert len(payload["top_groups"]) == 2
    assert payload["top_groups"][0]["group"] == "volatility"