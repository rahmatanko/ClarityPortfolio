import numpy as np
import pandas as pd

from src.data.features import build_feature_frame, fit_standard_scaler, transform_with_scaler
from src.eval.baselines import (
    align_prices_and_features,
    build_constant_weight_schedule,
    build_momentum_weight_schedule,
    cash_only_strategy,
    compute_next_step_returns,
    equal_weight_strategy,
    evaluate_baseline,
    momentum_strategy_from_features,
    run_weight_schedule_backtest,
)


def make_sample_prices(n_rows: int = 80) -> pd.DataFrame:
    # deterministic fake prices so tests stay stable and debugging stays sane
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
    scaled = transform_with_scaler(features, scaler)
    return scaled


def test_compute_next_step_returns_shape():
    prices = make_sample_prices()
    returns = compute_next_step_returns(prices)

    assert returns.shape[0] == len(prices) - 1
    assert returns.shape[1] == prices.shape[1]


def test_equal_weight_strategy_sums_to_one():
    weights = equal_weight_strategy(n_assets=5, include_cash=True)

    assert weights.shape == (6,)
    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)


def test_cash_only_strategy_is_all_cash():
    weights = cash_only_strategy(n_assets=5)

    assert weights.shape == (6,)
    assert np.allclose(weights[:-1], 0.0)
    assert np.isclose(weights[-1], 1.0)


def test_momentum_strategy_outputs_valid_weights():
    prices = make_sample_prices()
    features = make_scaled_features(prices)

    row = features.iloc[0]
    weights = momentum_strategy_from_features(row, asset_names=list(prices.columns))

    assert weights.shape == (6,)
    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)


def test_build_constant_weight_schedule_shape():
    prices = make_sample_prices()
    index = prices.index[:10]
    asset_names = list(prices.columns)
    weights = equal_weight_strategy(len(asset_names), include_cash=True)

    schedule = build_constant_weight_schedule(index, asset_names, weights)

    assert schedule.shape == (10, 6)
    assert list(schedule.columns) == asset_names + ["CASH"]


def test_build_momentum_weight_schedule_shape():
    prices = make_sample_prices()
    features = make_scaled_features(prices)

    schedule = build_momentum_weight_schedule(features.iloc[:10], list(prices.columns))

    assert schedule.shape == (10, 6)
    assert list(schedule.columns) == list(prices.columns) + ["CASH"]


def test_run_weight_schedule_backtest_returns_expected_columns():
    prices = make_sample_prices()
    features = make_scaled_features(prices)
    asset_names = list(prices.columns)

    schedule = build_constant_weight_schedule(
        features.index[:-1],
        asset_names,
        equal_weight_strategy(len(asset_names), include_cash=True),
    )

    backtest = run_weight_schedule_backtest(prices.loc[features.index], schedule)

    assert list(backtest.columns) == [
        "gross_return",
        "turnover",
        "transaction_cost",
        "net_return",
    ]


def test_align_prices_and_features_returns_matching_indices():
    prices = make_sample_prices()
    features = make_scaled_features(prices)

    aligned_prices, aligned_features = align_prices_and_features(prices, features)

    assert aligned_prices.index.equals(aligned_features.index)


def test_evaluate_equal_weight_baseline_runs():
    prices = make_sample_prices()
    features = make_scaled_features(prices)

    backtest, summary = evaluate_baseline(
        prices=prices,
        features=features,
        strategy_name="equal_weight",
    )

    assert not backtest.empty
    assert summary["strategy_name"] == "equal_weight"
    assert "cumulative_return" in summary
    assert "sortino_ratio" in summary


def test_evaluate_cash_only_baseline_runs():
    prices = make_sample_prices()
    features = make_scaled_features(prices)

    backtest, summary = evaluate_baseline(
        prices=prices,
        features=features,
        strategy_name="cash_only",
    )

    assert not backtest.empty
    assert summary["strategy_name"] == "cash_only"


def test_evaluate_momentum_baseline_runs():
    prices = make_sample_prices()
    features = make_scaled_features(prices)

    backtest, summary = evaluate_baseline(
        prices=prices,
        features=features,
        strategy_name="momentum",
    )

    assert not backtest.empty
    assert summary["strategy_name"] == "momentum"


def test_backtest_first_step_has_positive_transaction_cost_when_rebalancing_from_cash():
    prices = make_sample_prices()
    features = make_scaled_features(prices)
    asset_names = list(prices.columns)

    schedule = build_constant_weight_schedule(
        features.index[:-1],
        asset_names,
        equal_weight_strategy(len(asset_names), include_cash=True),
    )

    backtest = run_weight_schedule_backtest(
        prices=prices.loc[features.index],
        weights_schedule=schedule,
        transaction_cost=0.01,
    )

    assert backtest.iloc[0]["turnover"] > 0.0
    assert backtest.iloc[0]["transaction_cost"] > 0.0