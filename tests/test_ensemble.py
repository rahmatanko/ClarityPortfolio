import numpy as np
import pandas as pd

from src.data.features import build_feature_frame, fit_standard_scaler, transform_with_scaler
from src.models.ensemble import (
    compute_disagreement,
    disagreement_to_confidence,
    ensemble_inference,
    infer_on_split_row,
    pretty_allocation_dict,
    softmax,
)


def make_sample_prices(n_rows: int = 80) -> pd.DataFrame:
    # deterministic fake price data so ensemble tests stay stable
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
    # tiny deterministic model stub returning a preset raw action vector
    def __init__(self, raw_action):
        self.raw_action = np.asarray(raw_action, dtype=np.float32)

    def predict(self, observation, deterministic=True):
        return self.raw_action, None


def test_softmax_returns_valid_allocation():
    raw = np.array([0.1, -0.3, 1.2, 0.0, 0.5, -0.2], dtype=np.float32)
    weights = softmax(raw)

    assert weights.shape == (6,)
    assert np.all(weights >= 0.0)
    assert np.isclose(weights.sum(), 1.0)


def test_compute_disagreement_zero_when_members_identical():
    allocations = np.array(
        [
            [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
        ]
    )

    disagreement = compute_disagreement(allocations)

    assert np.isclose(disagreement, 0.0)


def test_confidence_decreases_as_disagreement_increases():
    low = disagreement_to_confidence(0.01, alpha=10.0)
    high = disagreement_to_confidence(0.20, alpha=10.0)

    assert low > high
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0


def test_ensemble_inference_returns_expected_keys():
    models = {
        42: DummyModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        43: DummyModel([0.2, 0.1, -0.1, 0.0, 0.3, -0.2]),
        44: DummyModel([-0.2, 0.4, 0.0, -0.1, 0.1, 0.2]),
    }

    obs = np.zeros(31, dtype=np.float32)

    result = ensemble_inference(models, obs, confidence_alpha=10.0)

    assert "ensemble_weights" in result
    assert "member_weights" in result
    assert "member_raw_actions" in result
    assert "disagreement" in result
    assert "confidence" in result

    assert result["ensemble_weights"].shape == (6,)
    assert np.isclose(result["ensemble_weights"].sum(), 1.0)
    assert 0.0 <= result["confidence"] <= 1.0


def test_infer_on_split_row_builds_full_observation():
    prices = make_sample_prices()
    features = make_scaled_features(prices)

    models = {
        42: DummyModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        43: DummyModel([0.1, 0.2, -0.1, 0.0, 0.3, -0.2]),
        44: DummyModel([0.3, -0.1, 0.1, 0.0, -0.2, 0.2]),
    }

    result = infer_on_split_row(models=models, features=features, row_idx=0)

    assert "observation" in result
    assert "timestamp" in result
    assert result["observation"].shape[0] == features.shape[1] + 6
    assert result["timestamp"] == features.index[0]


def test_pretty_allocation_dict_labels_cash():
    weights = np.array([0.1, 0.2, 0.15, 0.25, 0.1, 0.2])
    asset_names = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]

    pretty = pretty_allocation_dict(weights, asset_names)

    assert set(pretty.keys()) == {"AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "CASH"}
    assert np.isclose(sum(pretty.values()), 1.0)