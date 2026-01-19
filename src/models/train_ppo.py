import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch as th
import yaml
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.data.download import load_config as load_data_config
from src.data.download import load_price_data
from src.data.features import (
    fit_transform_train_features,
    split_by_date,
    transform_split_features,
)
from src.env.portfolio_env import PortfolioEnv
from src.eval.metrics import summarize_backtest


TRAIN_CONFIG_PATH = os.path.join("configs", "train.yaml")
DATA_CONFIG_PATH = os.path.join("configs", "data.yaml")


def load_train_config(config_path: str = TRAIN_CONFIG_PATH) -> dict:
    # load training settings from yaml so experiments are reproducible
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required_top_keys = {"seed", "env", "ppo", "policy_kwargs", "artifacts"}
    missing = required_top_keys - set(cfg.keys())
    if missing:
        raise KeyError(f"train config missing required keys: {sorted(missing)}")

    return cfg

def clone_config(cfg: dict) -> dict:
    # deep-copy config through json-safe roundtrip so seed-specific edits don't mutate the original
    return json.loads(json.dumps(cfg))


def apply_seed_to_config(cfg: dict, seed: int) -> dict:
    # make a seed-specific config so each run saves to distinct artifact names
    cfg = clone_config(cfg)
    cfg["seed"] = seed

    artifacts = cfg["artifacts"]
    artifacts["model_name"] = f"ppo_seed{seed}"
    artifacts["scaler_name"] = f"feature_scaler_seed{seed}.json"
    artifacts["summary_name"] = f"training_summary_seed{seed}.json"
    artifacts["validation_backtest_name"] = f"validation_backtest_seed{seed}.csv"

    return cfg


def set_global_seed(seed: int) -> None:
    # set all seeds we reasonably can so training runs are more reproducible
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def resolve_activation_fn(name: str):
    # map a simple config string to an actual torch activation class
    # this keeps yaml readable while still letting us construct the policy cleanly
    activation_map = {
        "ReLU": th.nn.ReLU,
        "Tanh": th.nn.Tanh,
        "ELU": th.nn.ELU,
        "LeakyReLU": th.nn.LeakyReLU,
    }

    if name not in activation_map:
        raise ValueError(f"unsupported activation function: {name}")

    return activation_map[name]


def build_policy_kwargs(cfg: dict) -> dict:
    # convert yaml-friendly policy config into stable-baselines format
    policy_cfg = cfg["policy_kwargs"]

    return {
        "activation_fn": resolve_activation_fn(policy_cfg["activation_fn"]),
        "net_arch": policy_cfg["net_arch"],
    }


def ensure_dir(path: str) -> None:
    # helper so artifact writing never fails because a directory doesn't exist yet
    os.makedirs(path, exist_ok=True)


def prepare_datasets(
    data_config_path: str = DATA_CONFIG_PATH,
    train_config_path: str = TRAIN_CONFIG_PATH,
    train_cfg_override: dict | None = None,
) -> dict:
    # full data prep for train/validation/test:
    # load prices, split chronologically, build train-only scaler, transform val/test
    data_cfg = load_data_config(data_config_path)
    train_cfg = train_cfg_override if train_cfg_override is not None else load_train_config(train_config_path)

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

    train_features, scaler = fit_transform_train_features(
        train_prices=train_prices,
        scaler_output_path=scaler_path,
    )
    val_features = transform_split_features(val_prices, scaler)
    test_features = transform_split_features(test_prices, scaler)

    # align prices to the exact feature timestamps so env mechanics are clean
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
        "scaler": scaler,
    }


def make_portfolio_env(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    env_cfg: dict,
):
    # create one environment instance with explicit mechanics from config
    env = PortfolioEnv(
        prices=prices,
        features=features,
        transaction_cost=env_cfg["transaction_cost"],
        downside_penalty_lambda=env_cfg["downside_penalty_lambda"],
        cash_return=env_cfg["cash_return"],
    )

    # episode stats wrapper makes debugging and training logs nicer
    env = RecordEpisodeStatistics(env)
    return env


def make_vec_env(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    env_cfg: dict,
) -> DummyVecEnv:
    # stable-baselines expects a vectorized env interface
    return DummyVecEnv(
        [lambda: make_portfolio_env(prices=prices, features=features, env_cfg=env_cfg)]
    )


def rollout_policy(model, env: PortfolioEnv) -> pd.DataFrame:
    # deterministic validation rollout:
    # we run one full episode and record the portfolio behaviour step by step
    obs, info = env.reset()

    done = False
    records = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)

        records.append(
            {
                "date": step_info["date"],
                "gross_return": step_info["gross_return"],
                "transaction_cost": step_info["transaction_cost"],
                "net_return": step_info["net_return"],
                "reward": step_info["reward"],
                "turnover": step_info["turnover"],
            }
        )

        done = terminated or truncated

    backtest = pd.DataFrame(records).set_index("date")
    return backtest


def build_run_paths(train_cfg: dict) -> dict:
    # collect artifact paths in one place so later code stays readable
    model_dir = train_cfg["artifacts"]["model_dir"]
    run_dir = train_cfg["artifacts"]["run_dir"]

    ensure_dir(model_dir)
    ensure_dir(run_dir)

    model_name = train_cfg["artifacts"]["model_name"]
    summary_name = train_cfg["artifacts"]["summary_name"]
    validation_name = train_cfg["artifacts"]["validation_backtest_name"]

    return {
        "model_path": os.path.join(model_dir, f"{model_name}.zip"),
        "summary_path": os.path.join(run_dir, summary_name),
        "validation_backtest_path": os.path.join(run_dir, validation_name),
    }


def train_single_ppo(
    train_prices: pd.DataFrame,
    train_features: pd.DataFrame,
    cfg: dict,
):
    # train one PPO policy on the train split only
    env_cfg = cfg["env"]
    ppo_cfg = cfg["ppo"]

    vec_env = make_vec_env(
        prices=train_prices,
        features=train_features,
        env_cfg=env_cfg,
    )

    policy_kwargs = build_policy_kwargs(cfg)

    model = PPO(
        policy=ppo_cfg["policy"],
        env=vec_env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=ppo_cfg["device"],
        seed=cfg["seed"],
    )

    model.learn(total_timesteps=ppo_cfg["total_timesteps"])

    return model


def validate_trained_policy(
    model,
    val_prices: pd.DataFrame,
    val_features: pd.DataFrame,
    env_cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    # run one deterministic validation episode and summarize it with our shared metric layer
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

    return backtest, summary


def save_training_summary(
    summary_path: str,
    cfg: dict,
    dataset_info: dict,
    validation_summary: dict,
) -> None:
    # store enough metadata that we can later explain exactly what was trained and how
    payload = {
        "timestamp": datetime.now().isoformat(),
        "seed": cfg["seed"],
        "env": cfg["env"],
        "ppo": cfg["ppo"],
        "policy_kwargs": cfg["policy_kwargs"],
        "dataset_info": dataset_info,
        "validation_summary": validation_summary,
    }

    ensure_dir(os.path.dirname(summary_path))

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def run_training_pipeline(
    data_config_path: str = DATA_CONFIG_PATH,
    train_config_path: str = TRAIN_CONFIG_PATH,
    cfg_override: dict | None = None,
) -> dict:
    # end-to-end training entry point:
    # prepare data, train one model, validate it, save everything
    cfg = cfg_override if cfg_override is not None else load_train_config(train_config_path)
    set_global_seed(cfg["seed"])

    datasets = prepare_datasets(
        data_config_path=data_config_path,
        train_config_path=train_config_path,
        train_cfg_override=cfg,
    )

    model = train_single_ppo(
        train_prices=datasets["train_prices"],
        train_features=datasets["train_features"],
        cfg=cfg,
    )

    validation_backtest, validation_summary = validate_trained_policy(
        model=model,
        val_prices=datasets["val_prices"],
        val_features=datasets["val_features"],
        env_cfg=cfg["env"],
    )

    paths = build_run_paths(cfg)

    model.save(paths["model_path"])
    validation_backtest.to_csv(paths["validation_backtest_path"])

    dataset_info = {
        "train_rows": int(len(datasets["train_prices"])),
        "val_rows": int(len(datasets["val_prices"])),
        "test_rows": int(len(datasets["test_prices"])),
        "feature_dim": int(datasets["train_features"].shape[1]),
        "asset_names": list(datasets["train_prices"].columns),
        "scaler_path": datasets["scaler_path"],
    }

    save_training_summary(
        summary_path=paths["summary_path"],
        cfg=cfg,
        dataset_info=dataset_info,
        validation_summary=validation_summary,
    )

    return {
        "model": model,
        "validation_backtest": validation_backtest,
        "validation_summary": validation_summary,
        "paths": paths,
        "dataset_info": dataset_info,
    }


if __name__ == "__main__":
    # direct script entry point
    result = run_training_pipeline()

    print("\n[INFO] validation summary:")
    for key, value in result["validation_summary"].items():
        print(f"{key}: {value}")

    print("\n[INFO] saved artifacts:")
    for key, value in result["paths"].items():
        print(f"{key}: {value}")