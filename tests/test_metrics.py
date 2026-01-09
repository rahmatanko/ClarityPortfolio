import numpy as np
import pandas as pd

from src.eval.metrics import (
    annualized_return,
    annualized_volatility,
    average_transaction_cost,
    average_turnover,
    cumulative_return,
    drawdown_series,
    equity_curve,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summarize_backtest,
    total_transaction_cost,
    total_turnover,
)


def test_cumulative_return_matches_manual_compounding():
    returns = pd.Series([0.01, -0.02, 0.03])
    expected = (1.01 * 0.98 * 1.03) - 1.0

    actual = cumulative_return(returns)

    assert np.isclose(actual, expected)


def test_equity_curve_starts_with_first_compounded_step():
    returns = pd.Series([0.10, -0.05, 0.02])

    curve = equity_curve(returns)

    expected = pd.Series([1.10, 1.10 * 0.95, 1.10 * 0.95 * 1.02], name="equity_curve")
    assert np.allclose(curve.values, expected.values)


def test_drawdown_series_is_non_positive():
    returns = pd.Series([0.10, -0.20, 0.05, -0.10])

    dd = drawdown_series(returns)

    assert (dd <= 0.0).all()


def test_max_drawdown_matches_expected_value():
    returns = pd.Series([0.10, -0.20, 0.05])

    # equity curve: 1.10 -> 0.88 -> 0.924
    # running peak stays 1.10, worst drawdown = 0.88 / 1.10 - 1 = -0.20
    actual = max_drawdown(returns)

    assert np.isclose(actual, -0.20)


def test_annualized_volatility_zero_for_constant_returns():
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])

    vol = annualized_volatility(returns)

    assert np.isclose(vol, 0.0)


def test_sharpe_ratio_zero_when_volatility_zero():
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])

    sharpe = sharpe_ratio(returns)

    assert np.isclose(sharpe, 0.0)


def test_sortino_ratio_zero_when_no_downside_deviation():
    returns = pd.Series([0.01, 0.02, 0.015, 0.03])

    sortino = sortino_ratio(returns)

    assert np.isclose(sortino, 0.0)


def test_turnover_metrics_match_manual_values():
    turnovers = pd.Series([0.1, 0.3, 0.2])

    assert np.isclose(average_turnover(turnovers), 0.2)
    assert np.isclose(total_turnover(turnovers), 0.6)


def test_transaction_cost_metrics_match_manual_values():
    costs = pd.Series([0.001, 0.002, 0.0005])

    assert np.isclose(average_transaction_cost(costs), (0.001 + 0.002 + 0.0005) / 3)
    assert np.isclose(total_transaction_cost(costs), 0.0035)


def test_summarize_backtest_contains_core_keys():
    returns = pd.Series([0.01, -0.02, 0.015, 0.005])
    turnovers = pd.Series([0.2, 0.1, 0.15, 0.05])
    costs = pd.Series([0.001, 0.0005, 0.0007, 0.0003])

    summary = summarize_backtest(
        net_returns=returns,
        turnovers=turnovers,
        transaction_costs=costs,
    )

    expected_keys = {
        "n_steps",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "average_turnover",
        "total_turnover",
        "average_transaction_cost",
        "total_transaction_cost",
    }

    assert expected_keys.issubset(summary.keys())


def test_annualized_return_is_finite_for_valid_returns():
    returns = pd.Series([0.01, -0.005, 0.012, 0.0, 0.004])

    ann_return = annualized_return(returns)

    assert np.isfinite(ann_return)


def test_metrics_accept_numpy_arrays():
    returns = np.array([0.01, -0.02, 0.03])

    assert np.isfinite(cumulative_return(returns))
    assert np.isfinite(annualized_return(returns))
    assert np.isfinite(max_drawdown(returns))