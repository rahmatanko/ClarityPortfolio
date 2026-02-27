import os
import pandas as pd
import matplotlib.pyplot as plt

FIG_DIR = os.path.join("figures")
RUNS_DIR = os.path.join("data", "artifacts", "runs")
INPUT_CSV = os.path.join(RUNS_DIR, "explanation_regime_validation.csv")

os.makedirs(FIG_DIR, exist_ok=True)


def main():
    df = pd.read_csv(INPUT_CSV)

    plt.figure(figsize=(8, 5))
    plt.bar(df["regime"], df["confidence"])

    plt.xlabel("Market Regime")
    plt.ylabel("Confidence")
    plt.title("Confidence Across Market Regimes")
    plt.xticks(rotation=15)
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, "regime_confidence.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] saved {out_path}")


if __name__ == "__main__":
    main()