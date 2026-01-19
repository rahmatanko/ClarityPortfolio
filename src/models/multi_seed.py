import json
import os

import yaml

from src.models.train_ppo import (
    DATA_CONFIG_PATH,
    TRAIN_CONFIG_PATH,
    apply_seed_to_config,
    load_train_config,
    run_training_pipeline,
)


MULTI_SEED_CONFIG_PATH = os.path.join("configs", "multi_seed.yaml")


def load_multi_seed_config(config_path: str = MULTI_SEED_CONFIG_PATH) -> dict:
    # load the list of seeds and naming preferences for ensemble training
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required_keys = {"seeds", "naming"}
    missing = required_keys - set(cfg.keys())
    if missing:
        raise KeyError(f"multi-seed config missing required keys: {sorted(missing)}")

    if not cfg["seeds"]:
        raise ValueError("multi-seed config contains no seeds")

    return cfg


def train_across_seeds(
    data_config_path: str = DATA_CONFIG_PATH,
    train_config_path: str = TRAIN_CONFIG_PATH,
    multi_seed_config_path: str = MULTI_SEED_CONFIG_PATH,
) -> dict:
    # train one PPO policy per seed and collect the results
    base_cfg = load_train_config(train_config_path)
    multi_cfg = load_multi_seed_config(multi_seed_config_path)

    results = {}
    summaries = []

    for seed in multi_cfg["seeds"]:
        print(f"\n[INFO] training seed {seed}...")

        seeded_cfg = apply_seed_to_config(base_cfg, seed)

        result = run_training_pipeline(
            data_config_path=data_config_path,
            train_config_path=train_config_path,
            cfg_override=seeded_cfg,
        )

        results[seed] = result

        summaries.append(
            {
                "seed": seed,
                "model_path": result["paths"]["model_path"],
                **result["validation_summary"],
            }
        )

    return {
        "results": results,
        "summaries": summaries,
    }


def save_multi_seed_summary(
    summaries: list[dict],
    output_path: str,
) -> None:
    # save an aggregated overview across all seed runs for later comparison
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, default=str)


def run_multi_seed_training(
    data_config_path: str = DATA_CONFIG_PATH,
    train_config_path: str = TRAIN_CONFIG_PATH,
    multi_seed_config_path: str = MULTI_SEED_CONFIG_PATH,
) -> dict:
    # full entry point for ensemble-member training
    result = train_across_seeds(
        data_config_path=data_config_path,
        train_config_path=train_config_path,
        multi_seed_config_path=multi_seed_config_path,
    )

    base_cfg = load_train_config(train_config_path)
    run_dir = base_cfg["artifacts"]["run_dir"]
    summary_path = os.path.join(run_dir, "multi_seed_summary.json")

    save_multi_seed_summary(
        summaries=result["summaries"],
        output_path=summary_path,
    )

    result["multi_seed_summary_path"] = summary_path
    return result


if __name__ == "__main__":
    result = run_multi_seed_training()

    print("\n[INFO] multi-seed validation summaries:")
    for row in result["summaries"]:
        print(row)

    print(f"\n[INFO] saved multi-seed summary: {result['multi_seed_summary_path']}")