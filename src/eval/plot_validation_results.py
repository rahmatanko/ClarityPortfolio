import os
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = os.path.join("data", "artifacts", "runs")
FIG_DIR = os.path.join("figures")

os.makedirs(FIG_DIR, exist_ok=True)


def load_backtest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df


def equity_curve(net_returns: pd.Series) -> pd.Series:
    return (1.0 + net_returns).cumprod()


def main():
    files = {
        "PPO": "ppo_seed42_ppo_validation.csv",
        "Equal-weight": "ppo_seed42_equal_weight_validation.csv",
        "Momentum": "ppo_seed42_momentum_validation.csv",
        "Cash-only": "ppo_seed42_cash_only_validation.csv",
    }

    plt.figure(figsize=(10, 6))

    for label, filename in files.items():
        path = os.path.join(RUNS_DIR, filename)
        df = load_backtest(path)
        curve = equity_curve(df["net_return"])
        plt.plot(curve.index, curve.values, label=label)

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("Validation Equity Curves")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, "validation_equity_curves.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] saved {out_path}")


if __name__ == "__main__":
    main()