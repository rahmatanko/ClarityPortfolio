import numpy as np
import pandas as pd

from src.eval.metrics import summarize_backtest


def align_prices_and_features(
    prices: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # baselines must run on exactly the same timestamps as the RL environment
    # otherwise comparisons would be unfair or flat-out wrong
    if prices.empty:
        raise ValueError("prices dataframe is empty")

    if features.empty:
        raise ValueError("features dataframe is empty")

    prices = prices.sort_index()
    features = features.sort_index()

    common_index = prices.index.intersection(features.index)

    if len(common_index) < 2:
        raise ValueError("prices and features do not have enough overlapping dates")

    return prices.loc[common_index].copy(), features.loc[common_index].copy()


def softmax(x: np.ndarray) -> np.ndarray:
    # keep a local softmax helper so momentum baseline can also output valid weights if needed
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    weights = exp_x / np.sum(exp_x)
    return weights.astype(np.float64)


def compute_next_step_returns(prices: pd.DataFrame) -> pd.DataFrame:
    # per-step risky asset returns from t to t+1
    # these are the returns the portfolio experiences after choosing weights at time t
    if prices.empty:
        raise ValueError("prices dataframe is empty")

    prices = prices.sort_index()

    if (prices <= 0).any().any():
        raise ValueError("prices must be strictly positive")

    returns = prices.shift(-1) / prices - 1.0
    returns = returns.iloc[:-1].copy()

    if returns.empty:
        raise ValueError("not enough price rows to compute next-step returns")

    return returns


def equal_weight_strategy(
    n_assets: int,
    include_cash: bool = True,
) -> np.ndarray:
    # simplest diversified baseline:
    # either spread equally across risky assets + cash, or across risky assets only
    if n_assets <= 0:
        raise ValueError("n_assets must be positive")

    dim = n_assets + 1 if include_cash else n_assets
    weights = np.ones(dim, dtype=np.float64) / dim
    return weights


def cash_only_strategy(
    n_assets: int,
) -> np.ndarray:
    # fully defensive baseline: everything in cash
    if n_assets <= 0:
        raise ValueError("n_assets must be positive")

    weights = np.zeros(n_assets + 1, dtype=np.float64)
    weights[-1] = 1.0
    return weights


def momentum_strategy_from_features(
    feature_row: pd.Series,
    asset_names: list[str],
    return_suffix: str = "_ret_20",
    temperature: float = 25.0,
) -> np.ndarray:
    # simple interpretable baseline:
    # use long-horizon returns as a momentum signal
    # then softmax them into valid weights over risky assets + cash
    if feature_row.empty:
        raise ValueError("feature_row is empty")

    if len(asset_names) == 0:
        raise ValueError("asset_names is empty")

    scores = []

    for asset in asset_names:
        col = f"{asset}{return_suffix}"
        if col not in feature_row.index:
            raise ValueError(f"missing momentum feature column: {col}")
        scores.append(float(feature_row[col]))

    scores = np.asarray(scores, dtype=np.float64)

    # add a neutral cash score of 0
    # if all risky signals are weak/negative, cash can still receive meaningful weight
    full_scores = np.concatenate([scores * temperature, np.array([0.0])])

    weights = softmax(full_scores)
    return weights


def run_weight_schedule_backtest(
    prices: pd.DataFrame,
    weights_schedule: pd.DataFrame,
    transaction_cost: float = 0.001,
    cash_return: float = 0.0,
) -> pd.DataFrame:
    # generic backtest engine for any strategy that gives us a weight vector at each time step
    # this keeps baseline evaluation aligned with the same mechanics used in the RL environment
    if prices.empty:
        raise ValueError("prices dataframe is empty")

    if weights_schedule.empty:
        raise ValueError("weights_schedule dataframe is empty")

    prices = prices.sort_index()
    weights_schedule = weights_schedule.sort_index()

    risky_returns = compute_next_step_returns(prices)

    # weights are chosen at time t and applied to returns from t -> t+1
    common_index = risky_returns.index.intersection(weights_schedule.index)
    if len(common_index) == 0:
        raise ValueError("weights schedule and prices do not overlap on valid backtest dates")

    risky_returns = risky_returns.loc[common_index]
    weights_schedule = weights_schedule.loc[common_index]

    expected_cols = list(prices.columns) + ["CASH"]
    if list(weights_schedule.columns) != expected_cols:
        raise ValueError("weights_schedule columns do not match expected asset + CASH order")

    previous_weights = np.zeros(len(expected_cols), dtype=np.float64)
    previous_weights[-1] = 1.0

    records = []

    for date in common_index:
        weights = weights_schedule.loc[date].to_numpy(dtype=np.float64)

        if np.any(weights < 0):
            raise ValueError("weights_schedule contains negative weights")

        if not np.isclose(weights.sum(), 1.0):
            raise ValueError("weights for each row must sum to 1")

        full_returns = np.concatenate(
            [risky_returns.loc[date].to_numpy(dtype=np.float64), np.array([cash_return])]
        )

        gross_return = float(np.dot(weights, full_returns))
        turnover = float(np.sum(np.abs(weights - previous_weights)))
        cost = float(transaction_cost * turnover)
        net_return = gross_return - cost

        records.append(
            {
                "date": date,
                "gross_return": gross_return,
                "turnover": turnover,
                "transaction_cost": cost,
                "net_return": net_return,
            }
        )

        previous_weights = weights.copy()

    result = pd.DataFrame(records).set_index("date")
    return result


def build_constant_weight_schedule(
    index: pd.Index,
    asset_names: list[str],
    weights: np.ndarray,
) -> pd.DataFrame:
    # helper for equal-weight and cash-only strategies
    expected_dim = len(asset_names) + 1
    weights = np.asarray(weights, dtype=np.float64)

    if weights.shape != (expected_dim,):
        raise ValueError(f"weights must have shape {(expected_dim,)}, got {weights.shape}")

    if np.any(weights < 0):
        raise ValueError("weights must be nonnegative")

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must sum to 1")

    cols = asset_names + ["CASH"]
    schedule = pd.DataFrame(
        np.tile(weights, (len(index), 1)),
        index=index,
        columns=cols,
    )

    return schedule


def build_momentum_weight_schedule(
    features: pd.DataFrame,
    asset_names: list[str],
    return_suffix: str = "_ret_20",
    temperature: float = 25.0,
) -> pd.DataFrame:
    # build one weight vector per timestamp using a simple momentum heuristic
    if features.empty:
        raise ValueError("features dataframe is empty")

    rows = []

    for date, row in features.iterrows():
        weights = momentum_strategy_from_features(
            feature_row=row,
            asset_names=asset_names,
            return_suffix=return_suffix,
            temperature=temperature,
        )

        rows.append(weights)

    schedule = pd.DataFrame(
        rows,
        index=features.index,
        columns=asset_names + ["CASH"],
    )

    return schedule


def evaluate_baseline(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    strategy_name: str,
    transaction_cost: float = 0.001,
    cash_return: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    # one convenience wrapper so later scripts can evaluate baselines with minimal boilerplate
    prices, features = align_prices_and_features(prices, features)
    asset_names = list(prices.columns)

    if strategy_name == "equal_weight":
        weights = equal_weight_strategy(len(asset_names), include_cash=True)
        schedule = build_constant_weight_schedule(features.index[:-1], asset_names, weights)

    elif strategy_name == "cash_only":
        weights = cash_only_strategy(len(asset_names))
        schedule = build_constant_weight_schedule(features.index[:-1], asset_names, weights)

    elif strategy_name == "momentum":
        schedule = build_momentum_weight_schedule(features.iloc[:-1], asset_names)

    else:
        raise ValueError(f"unknown strategy_name: {strategy_name}")

    backtest = run_weight_schedule_backtest(
        prices=prices,
        weights_schedule=schedule,
        transaction_cost=transaction_cost,
        cash_return=cash_return,
    )

    summary = summarize_backtest(
        net_returns=backtest["net_return"],
        turnovers=backtest["turnover"],
        transaction_costs=backtest["transaction_cost"],
    )

    summary["strategy_name"] = strategy_name

    return backtest, summary