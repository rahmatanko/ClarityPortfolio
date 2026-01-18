import os
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data.features import build_feature_frame, fit_standard_scaler, transform_with_scaler
from src.eval.compare import save_validation_comparison, validate_saved_ppo
from src.env.portfolio_env import PortfolioEnv


def make_sample_prices(n_rows: int = 80) -> pd.DataFrame:
    # deterministic fake prices keep tests stable and fast
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
    # tiny model stub so validation can be tested without real SB3 training
    def predict(self, obs, deterministic=True):
        return np.zeros(6, dtype=np.float32), None


def test_validate_saved_ppo_returns_summary_and_backtest(tmp_path):
    prices = make_sample_prices()
    features = make_scaled_features(prices)
    prices = prices.loc[features.index]

    fake_model_path = tmp_path / "fake_model.zip"
    fake_model_path.write_text("placeholder", encoding="utf-8")

    env_cfg = {
        "transaction_cost": 0.001,
        "downside_penalty_lambda": 10.0,
        "cash_return": 0.0,
    }

    with patch("src.eval.compare.PPO.load", return_value=DummyModel()):
        backtest, summary = validate_saved_ppo(
            model_path=str(fake_model_path),
            val_prices=prices,
            val_features=features,
            env_cfg=env_cfg,
        )

    assert not backtest.empty
    assert summary["strategy_name"] == "ppo"
    assert "cumulative_return" in summary
    assert "sortino_ratio" in summary


def test_save_validation_comparison_writes_files(tmp_path):
    comparison_df = pd.DataFrame(
        [
            {"strategy_name": "ppo", "cumulative_return": 0.1},
            {"strategy_name": "equal_weight", "cumulative_return": 0.05},
        ]
    ).set_index("strategy_name")

    ppo_backtest = pd.DataFrame({"net_return": [0.01, -0.02]})
    eq_backtest = pd.DataFrame({"net_return": [0.005, 0.003]})
    cash_backtest = pd.DataFrame({"net_return": [0.0, 0.0]})
    mom_backtest = pd.DataFrame({"net_return": [0.02, -0.01]})

    result = {
        "ppo_backtest": ppo_backtest,
        "equal_weight_backtest": eq_backtest,
        "cash_only_backtest": cash_backtest,
        "momentum_backtest": mom_backtest,
        "comparison_df": comparison_df,
        "summaries": [
            {"strategy_name": "ppo", "cumulative_return": 0.1},
            {"strategy_name": "equal_weight", "cumulative_return": 0.05},
        ],
        "paths": {
            "ppo_validation_backtest_path": str(tmp_path / "ppo.csv"),
            "equal_weight_backtest_path": str(tmp_path / "eq.csv"),
            "cash_only_backtest_path": str(tmp_path / "cash.csv"),
            "momentum_backtest_path": str(tmp_path / "mom.csv"),
            "comparison_table_path": str(tmp_path / "comparison.csv"),
            "comparison_summary_path": str(tmp_path / "comparison.json"),
        },
    }

    save_validation_comparison(result)

    assert os.path.exists(result["paths"]["ppo_validation_backtest_path"])
    assert os.path.exists(result["paths"]["equal_weight_backtest_path"])
    assert os.path.exists(result["paths"]["cash_only_backtest_path"])
    assert os.path.exists(result["paths"]["momentum_backtest_path"])
    assert os.path.exists(result["paths"]["comparison_table_path"])
    assert os.path.exists(result["paths"]["comparison_summary_path"])