import os
import pandas as pd
import yfinance as yf
import yaml


CONFIG_PATH = os.path.join("configs", "data.yaml")


def load_config(config_path=CONFIG_PATH):
    # read experiment/data settings from yaml so the pipeline is reproducible
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # fail early if the config is missing something important
    required_keys = ["assets", "start_date", "end_date", "data_dir", "output_file"]
    for key in required_keys:
        if key not in cfg:
            raise KeyError(f"missing required config key: {key}")

    return cfg


def fetch_price_data(assets, start, end):
    # fetch adjusted close prices so splits/dividends don't mess up training data
    data = yf.download(
        assets,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    # yfinance sometimes gives multi-index columns; normalize that here
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError("downloaded data does not contain Close prices")
        prices = data["Close"]
    else:
        prices = data

    # keep only complete rows so downstream feature generation stays clean
    prices = prices.dropna()

    if prices.empty:
        raise ValueError("download failed: received empty dataframe after cleaning")

    # keep dates in chronological order just in case
    prices = prices.sort_index()

    # force column order to match the configured asset list
    # this avoids subtle bugs later when state and action vectors assume a fixed asset order
    prices = prices[assets]

    return prices


def save_price_data(prices, data_dir, output_file):
    # make sure the target directory exists before writing
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, output_file)
    prices.to_csv(output_path)
    return output_path


def load_price_data(data_dir, output_file):
    # load previously saved prices for reproducibility and speed
    output_path = os.path.join(data_dir, output_file)

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"no saved data found at {output_path}")

    prices = pd.read_csv(output_path, index_col=0, parse_dates=True)

    if prices.empty:
        raise ValueError("loaded price data is empty")

    if prices.isnull().values.any():
        raise ValueError("loaded price data contains NaNs")

    prices = prices.sort_index()

    return prices


def download_price_data(config_path=CONFIG_PATH):
    # top-level helper: load config, fetch data, validate it, and save it
    cfg = load_config(config_path)

    prices = fetch_price_data(
        assets=cfg["assets"],
        start=cfg["start_date"],
        end=cfg["end_date"]
    )

    output_path = save_price_data(
        prices=prices,
        data_dir=cfg["data_dir"],
        output_file=cfg["output_file"]
    )

    print(f"[INFO] downloaded shape: {prices.shape}")
    print(f"[INFO] saved to {output_path}")

    return prices


if __name__ == "__main__":
    prices = download_price_data()
    print(prices.head())