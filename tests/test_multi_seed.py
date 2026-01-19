import json
import os

from src.models.multi_seed import load_multi_seed_config, save_multi_seed_summary
from src.models.train_ppo import apply_seed_to_config


def test_load_multi_seed_config_valid(tmp_path):
    config_file = tmp_path / "multi_seed.yaml"
    config_file.write_text(
        """
seeds:
  - 42
  - 43
naming:
  model_prefix: "ppo"
  scaler_prefix: "feature_scaler"
  summary_prefix: "training_summary"
  validation_backtest_prefix: "validation_backtest"
""",
        encoding="utf-8",
    )

    cfg = load_multi_seed_config(str(config_file))

    assert cfg["seeds"] == [42, 43]
    assert cfg["naming"]["model_prefix"] == "ppo"


def test_apply_seed_to_config_updates_artifact_names():
    base_cfg = {
        "seed": 42,
        "artifacts": {
            "model_name": "ppo_seed42",
            "scaler_name": "feature_scaler_seed42.json",
            "summary_name": "training_summary_seed42.json",
            "validation_backtest_name": "validation_backtest_seed42.csv",
        },
    }

    seeded = apply_seed_to_config(base_cfg, 99)

    assert seeded["seed"] == 99
    assert seeded["artifacts"]["model_name"] == "ppo_seed99"
    assert seeded["artifacts"]["scaler_name"] == "feature_scaler_seed99.json"
    assert seeded["artifacts"]["summary_name"] == "training_summary_seed99.json"
    assert seeded["artifacts"]["validation_backtest_name"] == "validation_backtest_seed99.csv"


def test_save_multi_seed_summary_writes_json(tmp_path):
    summaries = [
        {"seed": 42, "cumulative_return": -0.1},
        {"seed": 43, "cumulative_return": -0.2},
    ]

    output_path = tmp_path / "multi_seed_summary.json"
    save_multi_seed_summary(summaries, str(output_path))

    assert os.path.exists(output_path)

    with open(output_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    assert len(payload) == 2
    assert payload[0]["seed"] == 42
    assert payload[1]["cumulative_return"] == -0.2