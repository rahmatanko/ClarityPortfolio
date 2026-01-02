import os

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    build_feature_frame,
    compute_log_returns,
    compute_rsi,
    fit_standard_scaler,
    fit_transform_train_features,
    get_feature_column_order,
    get_feature_groups,
    load_scaler,
    reorder_feature_columns,
    save_scaler,
    split_by_date,
    transform_with_scaler,
)


def make_sample_prices(n_rows: int = 80) -> pd.DataFrame:
    # make a deterministic fake price dataset so tests are stable and not dependent on the internet
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


def test_compute_log_returns_shape():
    prices = make_sample_prices()
    returns = compute_log_returns(prices)

    assert isinstance(returns, pd.DataFrame)
    assert returns.shape == prices.shape
    assert returns.index.equals(prices.index)


def test_compute_rsi_shape_and_range():
    prices = make_sample_prices()
    rsi = compute_rsi(prices)

    assert isinstance(rsi, pd.DataFrame)
    assert rsi.shape == prices.shape
    assert ((rsi >= 0) & (rsi <= 100)).all().all()


def test_build_feature_frame_expected_columns():
    prices = make_sample_prices()
    features = build_feature_frame(prices)

    expected_cols = get_feature_column_order(list(prices.columns))
    assert list(features.columns) == expected_cols
    assert not features.empty
    assert not features.isnull().values.any()


def test_build_feature_frame_drops_warmup_rows():
    prices = make_sample_prices()
    features = build_feature_frame(prices)

    # with max rolling window = 20 and returns needing one previous day,
    # the engineered feature matrix should be shorter than the raw price matrix
    assert len(features) < len(prices)


def test_reorder_feature_columns_restores_canonical_order():
    prices = make_sample_prices()
    features = build_feature_frame(prices)

    shuffled = features.sample(axis=1, frac=1, random_state=42)
    restored = reorder_feature_columns(shuffled, list(prices.columns))

    assert list(restored.columns) == get_feature_column_order(list(prices.columns))


def test_fit_standard_scaler_and_transform():
    prices = make_sample_prices()
    features = build_feature_frame(prices)

    scaler = fit_standard_scaler(features)
    scaled = transform_with_scaler(features, scaler)

    # mean should be close to 0 after scaling
    assert np.allclose(scaled.mean().values, 0, atol=1e-6)

    stds = scaled.std()

    # check each feature individually so we can debug easily if something breaks
    for col, std in stds.items():
        assert np.isclose(std, 1, atol=1e-5) or np.isclose(std, 0, atol=1e-8), f"unexpected std for {col}: {std}"


def test_transform_with_scaler_rejects_column_mismatch():
    prices = make_sample_prices()
    features = build_feature_frame(prices)

    scaler = fit_standard_scaler(features)

    bad = features.copy().rename(columns={features.columns[0]: "BROKEN_COL"})

    with pytest.raises(ValueError, match="feature columns do not match scaler columns"):
        transform_with_scaler(bad, scaler)


def test_save_and_load_scaler_roundtrip(tmp_path):
    prices = make_sample_prices()
    features = build_feature_frame(prices)

    scaler = fit_standard_scaler(features)

    output_path = os.path.join(tmp_path, "scaler.json")
    save_scaler(scaler, output_path)

    loaded = load_scaler(output_path)

    assert scaler["columns"] == loaded["columns"]
    assert scaler["mean"].keys() == loaded["mean"].keys()
    assert scaler["std"].keys() == loaded["std"].keys()


def test_split_by_date_chronological():
    prices = make_sample_prices(n_rows=120)

    train, val, test = split_by_date(
        prices,
        train_end="2020-03-15",
        val_end="2020-04-15",
    )

    assert not train.empty
    assert not val.empty
    assert not test.empty

    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()


def test_fit_transform_train_features_returns_scaled_frame():
    prices = make_sample_prices()
    scaled, scaler = fit_transform_train_features(prices)

    assert not scaled.empty
    assert scaled.shape[1] == 25  # 5 assets * (3 returns + 1 vol + 1 rsi)
    assert len(scaler["columns"]) == 25
    assert np.allclose(scaled.mean().values, 0, atol=1e-6)


def test_get_feature_groups_structure():
    assets = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]
    groups = get_feature_groups(assets)

    assert set(groups.keys()) == {
        "short_term_returns",
        "medium_term_returns",
        "long_term_returns",
        "volatility",
        "rsi",
    }

    assert len(groups["short_term_returns"]) == 5
    assert len(groups["volatility"]) == 5