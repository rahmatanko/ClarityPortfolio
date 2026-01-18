import json
import os

import pandas as pd
import yaml
from stable_baselines3 import PPO

from src.data.download import load_config as load_data_config
from src.data.download import load_price_data
from src.data.features import (
    fit_transform_train_features,
    split_by_date,
    transform_split_features,
)
from src.env.portfolio_env import PortfolioEnv
from src.eval.baselines import evaluate_baseline
from src.eval.metrics import summarize_backtest
from src.models.train_ppo import load_train_config, rollout_policy


DATA_CONFIG_PATH = os.path.join("configs", "data.yaml")
TRAIN_CONFIG_PATH = os.path.join("configs", "train.yaml")


def ensure_dir(path: str) -> None:
    # helper so artifact saving never fails because folders are missing
    os.makedirs(path, exist_ok=True)


def prepare_validation_data(
    data_config_path: str = DATA_CONFIG_PATH,
    train_config_path: str = TRAIN_CONFIG_PATH,
) -> dict:
    # rebuild train/val/test exactly the same way training did
    # this is important so PPO and baselines are compared on identical data
    data_cfg = load_data_config(data_config_path)
    train_cfg = load_train_config(train_config_path)

    prices = load_price_data(
        data_dir=data_cfg["data_dir"],
        output_file=data_cfg["output_file"],
    )

    train_prices, val_prices, test_prices = split_by_date(
        prices=prices,
        train_end=data_cfg["train_end"],
        val_end=data_cfg["val_end"],
    )

    artifacts_dir = data_cfg["artifacts_dir"]
    scaler_name = train_cfg["artifacts"]["scaler_name"]
    scaler_path = os.path.join(artifacts_dir, scaler_name)

    # fit scaler on train only, then transform val/test
    train_features, scaler = fit_transform_train_features(
        train_prices=train_prices,
        scaler_output_path=scaler_path,
    )
    val_features = transform_split_features(val_prices, scaler)
    test_features = transform_split_features(test_prices, scaler)

    # align prices to engineered feature timestamps
    train_prices = train_prices.loc[train_features.index]
    val_prices = val_prices.loc[val_features.index]
    test_prices = test_prices.loc[test_features.index]

    return {
        "train_prices": train_prices,
        "val_prices": val_prices,
        "test_prices": test_prices,
        "train_features": train_features,
        "val_features": val_features,
        "test_features": test_features,
        "scaler_path": scaler_path,
    }


def build_compare_paths(train_cfg: dict) -> dict:
    # collect all artifact locations for comparison outputs
    run_dir = train_cfg["artifacts"]["run_dir"]
    model_dir = train_cfg["artifacts"]["model_dir"]

    ensure_dir(run_dir)
    ensure_dir(model_dir)

    model_name = train_cfg["artifacts"]["model_name"]

    return {
        "model_path": os.path.join(model_dir, f"{model_name}.zip"),
        "ppo_validation_backtest_path": os.path.join(run_dir, f"{model_name}_ppo_validation.csv"),
        "equal_weight_backtest_path": os.path.join(run_dir, f"{model_name}_equal_weight_validation.csv"),
        "cash_only_backtest_path": os.path.join(run_dir, f"{model_name}_cash_only_validation.csv"),
        "momentum_backtest_path": os.path.join(run_dir, f"{model_name}_momentum_validation.csv"),
        "comparison_table_path": os.path.join(run_dir, f"{model_name}_validation_comparison.csv"),
        "comparison_summary_path": os.path.join(run_dir, f"{model_name}_validation_comparison.json"),
    }


def validate_saved_ppo(
    model_path: str,
    val_prices: pd.DataFrame,
    val_features: pd.DataFrame,
    env_cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    # load a trained PPO model and evaluate it deterministically on validation
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"trained PPO model not found at {model_path}")

    model = PPO.load(model_path)

    env = PortfolioEnv(
        prices=val_prices,
        features=val_features,
        transaction_cost=env_cfg["transaction_cost"],
        downside_penalty_lambda=env_cfg["downside_penalty_lambda"],
        cash_return=env_cfg["cash_return"],
    )

    backtest = rollout_policy(model, env)

    summary = summarize_backtest(
        net_returns=backtest["net_return"],
        turnovers=backtest["turnover"],
        transaction_costs=backtest["transaction_cost"],
    )
    summary["strategy_name"] = "ppo"

    return backtest, summary


def evaluate_all_validation_strategies(
    data_config_path: str = DATA_CONFIG_PATH,
    train_config_path: str = TRAIN_CONFIG_PATH,
) -> dict:
    # main comparison entry point:
    # rebuild validation data, evaluate PPO and all baselines, and combine everything
    train_cfg = load_train_config(train_config_path)
    env_cfg = train_cfg["env"]
    datasets = prepare_validation_data(
        data_config_path=data_config_path,
        train_config_path=train_config_path,
    )
    paths = build_compare_paths(train_cfg)

    val_prices = datasets["val_prices"]
    val_features = datasets["val_features"]

    # evaluate PPO on validation
    ppo_backtest, ppo_summary = validate_saved_ppo(
        model_path=paths["model_path"],
        val_prices=val_prices,
        val_features=val_features,
        env_cfg=env_cfg,
    )

    # evaluate baselines on the exact same validation split
    eq_backtest, eq_summary = evaluate_baseline(
        prices=val_prices,
        features=val_features,
        strategy_name="equal_weight",
        transaction_cost=env_cfg["transaction_cost"],
        cash_return=env_cfg["cash_return"],
    )

    cash_backtest, cash_summary = evaluate_baseline(
        prices=val_prices,
        features=val_features,
        strategy_name="cash_only",
        transaction_cost=env_cfg["transaction_cost"],
        cash_return=env_cfg["cash_return"],
    )

    mom_backtest, mom_summary = evaluate_baseline(
        prices=val_prices,
        features=val_features,
        strategy_name="momentum",
        transaction_cost=env_cfg["transaction_cost"],
        cash_return=env_cfg["cash_return"],
    )

    summaries = [ppo_summary, eq_summary, cash_summary, mom_summary]
    comparison_df = pd.DataFrame(summaries).set_index("strategy_name").sort_index()

    return {
        "ppo_backtest": ppo_backtest,
        "equal_weight_backtest": eq_backtest,
        "cash_only_backtest": cash_backtest,
        "momentum_backtest": mom_backtest,
        "comparison_df": comparison_df,
        "summaries": summaries,
        "paths": paths,
    }


def save_validation_comparison(result: dict) -> None:
    # persist all backtests and the final comparison table for reporting and later analysis
    result["ppo_backtest"].to_csv(result["paths"]["ppo_validation_backtest_path"])
    result["equal_weight_backtest"].to_csv(result["paths"]["equal_weight_backtest_path"])
    result["cash_only_backtest"].to_csv(result["paths"]["cash_only_backtest_path"])
    result["momentum_backtest"].to_csv(result["paths"]["momentum_backtest_path"])
    result["comparison_df"].to_csv(result["paths"]["comparison_table_path"])

    with open(result["paths"]["comparison_summary_path"], "w", encoding="utf-8") as f:
        json.dump(result["summaries"], f, indent=2, default=str)


def run_validation_comparison(
    data_config_path: str = DATA_CONFIG_PATH,
    train_config_path: str = TRAIN_CONFIG_PATH,
) -> dict:
    # full entry point: evaluate all strategies and save outputs
    result = evaluate_all_validation_strategies(
        data_config_path=data_config_path,
        train_config_path=train_config_path,
    )
    save_validation_comparison(result)
    return result


if __name__ == "__main__":
    result = run_validation_comparison()

    print("\n[INFO] validation comparison:")
    print(result["comparison_df"])

    print("\n[INFO] saved artifacts:")
    for key, value in result["paths"].items():
        print(f"{key}: {value}")