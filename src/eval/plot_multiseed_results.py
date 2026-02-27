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
        "Seed 42": "validation_backtest_seed42.csv",
        "Seed 43": "validation_backtest_seed43.csv",
        "Seed 44": "validation_backtest_seed44.csv",
    }

    plt.figure(figsize=(10, 6))

    for label, filename in files.items():
        path = os.path.join(RUNS_DIR, filename)
        df = load_backtest(path)
        curve = equity_curve(df["net_return"])
        plt.plot(curve.index, curve.values, label=label)

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("Validation Equity Curves Across PPO Seeds")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, "multiseed_equity_curves.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] saved {out_path}")


if __name__ == "__main__":
    main()