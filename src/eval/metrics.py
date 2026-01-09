import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def _to_series(values, name: str) -> pd.Series:
    # normalize inputs so every metric function can work with either
    # numpy arrays, python lists, or pandas series without weird edge cases
    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(values, name=name, dtype=float)

    if series.empty:
        raise ValueError(f"{name} is empty")

    if series.isnull().any():
        raise ValueError(f"{name} contains NaN values")

    return series.astype(float)


def cumulative_return(returns) -> float:
    # cumulative return compounds all per-step returns over the backtest horizon
    # e.g. [0.01, -0.02, 0.03] -> (1.01 * 0.98 * 1.03) - 1
    r = _to_series(returns, "returns")
    return float((1.0 + r).prod() - 1.0)


def annualized_return(returns, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    # annualize the compounded growth rate so results are easier to compare across runs
    r = _to_series(returns, "returns")

    total_growth = float((1.0 + r).prod())
    n_periods = len(r)

    if n_periods == 0:
        raise ValueError("returns is empty")

    ann_return = total_growth ** (periods_per_year / n_periods) - 1.0
    return float(ann_return)


def annualized_volatility(returns, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    # standard volatility metric scaled to annual units
    r = _to_series(returns, "returns")

    if len(r) < 2:
        return 0.0

    vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    return float(vol)


def sharpe_ratio(
    returns,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    # sharpe uses symmetric volatility penalty
    # for this project we mainly care about sortino, but sharpe is still useful for reference
    r = _to_series(returns, "returns")

    per_period_rf = risk_free_rate / periods_per_year
    excess = r - per_period_rf

    vol = annualized_volatility(excess, periods_per_year=periods_per_year)
    if np.isclose(vol, 0.0):
        return 0.0

    ann_excess_return = annualized_return(excess, periods_per_year=periods_per_year)
    return float(ann_excess_return / vol)


def sortino_ratio(
    returns,
    target_return: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    # sortino only penalizes downside volatility, which matches the project's risk-aware framing better
    r = _to_series(returns, "returns")

    per_period_target = target_return / periods_per_year
    downside = np.minimum(r - per_period_target, 0.0)

    # downside deviation is based only on negative excess returns
    downside_std = pd.Series(downside).std(ddof=1)

    if pd.isna(downside_std) or np.isclose(downside_std, 0.0):
        return 0.0

    annual_downside_dev = float(downside_std * np.sqrt(periods_per_year))
    ann_excess_return = annualized_return(
        r - per_period_target,
        periods_per_year=periods_per_year,
    )

    return float(ann_excess_return / annual_downside_dev)


def equity_curve(returns) -> pd.Series:
    # convert per-step returns into a normalized portfolio value path starting at 1.0
    r = _to_series(returns, "returns")
    curve = (1.0 + r).cumprod()
    curve.name = "equity_curve"
    return curve


def drawdown_series(returns) -> pd.Series:
    # drawdown at each point = distance from the running peak of the equity curve
    curve = equity_curve(returns)
    running_peak = curve.cummax()
    drawdowns = curve / running_peak - 1.0
    drawdowns.name = "drawdown"
    return drawdowns


def max_drawdown(returns) -> float:
    # maximum drawdown is one of the most important practical risk metrics for retail portfolios
    dd = drawdown_series(returns)
    return float(dd.min())


def average_turnover(turnovers) -> float:
    # average turnover tells us how aggressively the strategy is rebalancing
    t = _to_series(turnovers, "turnovers")
    return float(t.mean())


def total_turnover(turnovers) -> float:
    # useful for understanding total trading intensity over the full backtest
    t = _to_series(turnovers, "turnovers")
    return float(t.sum())


def total_transaction_cost(costs) -> float:
    # aggregate all transaction costs over the backtest horizon
    c = _to_series(costs, "costs")
    return float(c.sum())


def average_transaction_cost(costs) -> float:
    # average friction paid per step
    c = _to_series(costs, "costs")
    return float(c.mean())


def summarize_backtest(
    net_returns,
    turnovers=None,
    transaction_costs=None,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    # produce one compact summary dict that can go straight into tables, logs, or report outputs
    r = _to_series(net_returns, "net_returns")

    summary = {
        "n_steps": int(len(r)),
        "cumulative_return": cumulative_return(r),
        "annualized_return": annualized_return(r, periods_per_year=periods_per_year),
        "annualized_volatility": annualized_volatility(r, periods_per_year=periods_per_year),
        "sharpe_ratio": sharpe_ratio(
            r,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        ),
        "sortino_ratio": sortino_ratio(
            r,
            target_return=target_return,
            periods_per_year=periods_per_year,
        ),
        "max_drawdown": max_drawdown(r),
    }

    if turnovers is not None:
        t = _to_series(turnovers, "turnovers")
        summary["average_turnover"] = average_turnover(t)
        summary["total_turnover"] = total_turnover(t)

    if transaction_costs is not None:
        c = _to_series(transaction_costs, "costs")
        summary["average_transaction_cost"] = average_transaction_cost(c)
        summary["total_transaction_cost"] = total_transaction_cost(c)

    return summary