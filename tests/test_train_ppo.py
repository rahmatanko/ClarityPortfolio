import json
import os

import numpy as np
import pandas as pd
import yaml

from src.models.train_ppo import (
    build_policy_kwargs,
    load_train_config,
    rollout_policy,
    save_training_summary,
    set_global_seed,
)
from src.env.portfolio_env import PortfolioEnv
from src.data.features import build_feature_frame, fit_standard_scaler, transform_with_scaler


def make_sample_prices(n_rows: int = 80) -> pd.DataFrame:
    # deterministic fake price data so these tests stay fast and stable
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


class DummyModel:
    # lightweight stand-in for PPO so rollout tests don't require real training
    def predict(self, obs, deterministic=True):
        # always return a valid 6D raw action
        return np.zeros(6, dtype=np.float32), None


def test_load_train_config_valid(tmp_path):
    config_file = tmp_path / "train.yaml"
    config_file.write_text(
        """
seed: 42
env:
  transaction_cost: 0.001
  downside_penalty_lambda: 10.0
  cash_return: 0.0
ppo:
  policy: "MlpPolicy"
  learning_rate: 0.0003
  n_steps: 256
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  total_timesteps: 1000
  device: "cpu"
policy_kwargs:
  activation_fn: "ReLU"
  net_arch: [64, 64]
artifacts:
  model_dir: "data/artifacts/models"
  run_dir: "data/artifacts/runs"
  model_name: "ppo_seed42"
  scaler_name: "feature_scaler_seed42.json"
  summary_name: "training_summary_seed42.json"
  validation_backtest_name: "validation_backtest_seed42.csv"
""",
        encoding="utf-8",
    )

    cfg = load_train_config(str(config_file))

    assert cfg["seed"] == 42
    assert cfg["ppo"]["policy"] == "MlpPolicy"
    assert cfg["policy_kwargs"]["activation_fn"] == "ReLU"


def test_build_policy_kwargs_maps_activation_correctly():
    cfg = {
        "policy_kwargs": {
            "activation_fn": "ReLU",
            "net_arch": [64, 64],
        }
    }

    kwargs = build_policy_kwargs(cfg)

    assert kwargs["activation_fn"].__name__ == "ReLU"
    assert kwargs["net_arch"] == [64, 64]


def test_set_global_seed_makes_numpy_repeatable():
    set_global_seed(123)
    a = np.random.rand(5)

    set_global_seed(123)
    b = np.random.rand(5)

    assert np.allclose(a, b)


def test_rollout_policy_returns_expected_columns():
    prices = make_sample_prices()
    features = make_scaled_features(prices)
    prices = prices.loc[features.index]

    env = PortfolioEnv(prices=prices, features=features)
    model = DummyModel()

    backtest = rollout_policy(model, env)

    assert not backtest.empty
    assert list(backtest.columns) == [
        "gross_return",
        "transaction_cost",
        "net_return",
        "reward",
        "turnover",
    ]


def test_save_training_summary_writes_json(tmp_path):
    summary_path = tmp_path / "summary.json"

    cfg = {
        "seed": 42,
        "env": {"transaction_cost": 0.001},
        "ppo": {"policy": "MlpPolicy"},
        "policy_kwargs": {"activation_fn": "ReLU", "net_arch": [64, 64]},
    }

    dataset_info = {
        "train_rows": 100,
        "val_rows": 20,
        "test_rows": 20,
        "feature_dim": 25,
        "asset_names": ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"],
        "scaler_path": "data/artifacts/feature_scaler.json",
    }

    validation_summary = {
        "cumulative_return": 0.05,
        "sortino_ratio": 1.2,
    }

    save_training_summary(
        summary_path=str(summary_path),
        cfg=cfg,
        dataset_info=dataset_info,
        validation_summary=validation_summary,
    )

    assert os.path.exists(summary_path)

    with open(summary_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["seed"] == 42
    assert payload["dataset_info"]["feature_dim"] == 25
    assert payload["validation_summary"]["sortino_ratio"] == 1.2