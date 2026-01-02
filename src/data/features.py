import json
import os

import numpy as np
import pandas as pd


# all rolling windows we use in the project
RETURN_WINDOWS = [5, 10, 20]
VOL_WINDOW = 20
RSI_WINDOW = 14


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    # log returns are more stable than raw percentage change for modelling
    # we also keep the original column names so everything stays easy to trace later
    if prices.empty:
        raise ValueError("prices dataframe is empty")

    returns = np.log(prices / prices.shift(1))
    return returns


def compute_rsi(prices: pd.DataFrame, window: int = RSI_WINDOW) -> pd.DataFrame:
    # RSI is one of the interpretable momentum indicators we said we'd use in the report
    # this implementation is simple, transparent, and good enough for the project
    if prices.empty:
        raise ValueError("prices dataframe is empty")

    delta = prices.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # if avg_loss is 0, RSI should go to 100 in a very strong uptrend
    # we handle that cleanly instead of letting divide-by-zero create weird junk
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # no movement case can leave NaNs around, so we use 50 as the neutral fallback
    rsi = rsi.fillna(50.0)

    return rsi


def build_feature_frame(prices: pd.DataFrame) -> pd.DataFrame:
    # this is the main feature engineering entry point
    # it creates a clean dataframe where each row corresponds to one timestamp
    # and each column is one interpretable feature for one asset
    if prices.empty:
        raise ValueError("prices dataframe is empty")

    prices = prices.sort_index()

    returns = compute_log_returns(prices)
    rsi = compute_rsi(prices, window=RSI_WINDOW)

    feature_blocks = []

    # rolling mean returns over multiple horizons
    for window in RETURN_WINDOWS:
        rolling_ret = returns.rolling(window=window, min_periods=window).mean()
        rolling_ret = rolling_ret.add_suffix(f"_ret_{window}")
        feature_blocks.append(rolling_ret)

    # 20-day rolling volatility
    vol = returns.rolling(window=VOL_WINDOW, min_periods=VOL_WINDOW).std()
    vol = vol.add_suffix(f"_vol_{VOL_WINDOW}")
    feature_blocks.append(vol)

    # RSI scaled to [0, 1] so it's numerically nicer alongside returns/volatility
    rsi_scaled = (rsi / 100.0).add_suffix(f"_rsi_{RSI_WINDOW}")
    feature_blocks.append(rsi_scaled)

    features = pd.concat(feature_blocks, axis=1)

    # drop warmup rows where rolling windows are incomplete
    # we would rather lose a few early rows than silently train on broken features
    features = features.dropna().sort_index()

    if features.empty:
        raise ValueError("feature generation produced an empty dataframe")

    return features


def get_feature_column_order(asset_names: list[str]) -> list[str]:
    # we define the canonical order once so train/val/test all match perfectly
    cols = []

    for window in RETURN_WINDOWS:
        for asset in asset_names:
            cols.append(f"{asset}_ret_{window}")

    for asset in asset_names:
        cols.append(f"{asset}_vol_{VOL_WINDOW}")

    for asset in asset_names:
        cols.append(f"{asset}_rsi_{RSI_WINDOW}")

    return cols


def reorder_feature_columns(features: pd.DataFrame, asset_names: list[str]) -> pd.DataFrame:
    # make sure column ordering is deterministic
    # this matters a lot once we start feeding states into PPO models
    expected = get_feature_column_order(asset_names)

    missing = [col for col in expected if col not in features.columns]
    if missing:
        raise ValueError(f"missing expected feature columns: {missing}")

    return features[expected].copy()


def fit_standard_scaler(train_features: pd.DataFrame) -> dict:
    # scaler must be fit on training data only
    # otherwise we'd leak information from validation/test into the model
    if train_features.empty:
        raise ValueError("train_features dataframe is empty")

    means = train_features.mean(axis=0)
    stds = train_features.std(axis=0)

    # if a feature is constant, std becomes 0 and would break normalization
    # replacing it with 1 keeps the feature centered without exploding anything
    stds = stds.replace(0, 1.0)

    scaler = {
        "columns": list(train_features.columns),
        "mean": means.to_dict(),
        "std": stds.to_dict(),
    }

    return scaler


def transform_with_scaler(features: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    # apply previously fitted train-only normalization stats
    # and make sure column alignment is exact before doing anything
    if features.empty:
        raise ValueError("features dataframe is empty")

    expected_cols = scaler["columns"]
    actual_cols = list(features.columns)

    if actual_cols != expected_cols:
        raise ValueError("feature columns do not match scaler columns")

    mean_series = pd.Series(scaler["mean"])
    std_series = pd.Series(scaler["std"])

    scaled = (features - mean_series) / std_series
    return scaled


def save_scaler(scaler: dict, output_path: str) -> None:
    # save scaler stats so val/test/inference all use the exact same normalization
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scaler, f, indent=2)


def load_scaler(input_path: str) -> dict:
    # load previously saved scaler stats
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"scaler file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        scaler = json.load(f)

    required_keys = {"columns", "mean", "std"}
    if not required_keys.issubset(scaler.keys()):
        raise ValueError("scaler file is missing required keys")

    return scaler


def split_by_date(
    prices: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # strict chronological split: train first, then validation, then test
    # this is non-negotiable for time series if we want believable evaluation
    if prices.empty:
        raise ValueError("prices dataframe is empty")

    prices = prices.sort_index()

    train = prices.loc[prices.index < pd.Timestamp(train_end)].copy()
    val = prices.loc[(prices.index >= pd.Timestamp(train_end)) & (prices.index < pd.Timestamp(val_end))].copy()
    test = prices.loc[prices.index >= pd.Timestamp(val_end)].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError("one or more chronological splits are empty")

    if not (train.index.max() < val.index.min() and val.index.max() < test.index.min()):
        raise ValueError("time split ordering is invalid")

    return train, val, test


def fit_transform_train_features(
    train_prices: pd.DataFrame,
    scaler_output_path: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    # helper for the training split: build features, fit scaler, transform features
    train_features = build_feature_frame(train_prices)
    train_features = reorder_feature_columns(train_features, list(train_prices.columns))

    scaler = fit_standard_scaler(train_features)
    train_scaled = transform_with_scaler(train_features, scaler)

    if scaler_output_path is not None:
        save_scaler(scaler, scaler_output_path)

    return train_scaled, scaler


def transform_split_features(
    prices: pd.DataFrame,
    scaler: dict,
) -> pd.DataFrame:
    # helper for validation/test/inference: build features using only that split's own past rows
    # then apply train-fitted normalization
    features = build_feature_frame(prices)
    asset_names = [col.split("_")[0] for col in scaler["columns"][: len(prices.columns)]]
    features = reorder_feature_columns(features, asset_names)
    features = transform_with_scaler(features, scaler)

    return features


def get_feature_groups(asset_names: list[str]) -> dict[str, list[str]]:
    # these groups will be useful later for perturbation-based explanations
    # grouping features now saves us from messy ad-hoc logic later
    groups = {
        "short_term_returns": [f"{asset}_ret_5" for asset in asset_names],
        "medium_term_returns": [f"{asset}_ret_10" for asset in asset_names],
        "long_term_returns": [f"{asset}_ret_20" for asset in asset_names],
        "volatility": [f"{asset}_vol_20" for asset in asset_names],
        "rsi": [f"{asset}_rsi_14" for asset in asset_names],
    }
    return groups