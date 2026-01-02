import os
import pandas as pd
import pytest
from unittest.mock import patch

from src.data.download import (
    load_config,
    fetch_price_data,
    save_price_data,
    load_price_data,
)


def make_sample_prices():
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "AAPL": [100, 101, 102, 103, 104],
            "MSFT": [200, 201, 202, 203, 204],
            "NVDA": [300, 301, 302, 303, 304],
            "GOOGL": [90, 91, 92, 93, 94],
            "TSLA": [150, 151, 152, 153, 154],
        },
        index=dates,
    )


def test_load_config_valid(tmp_path):
    config_file = tmp_path / "data.yaml"
    config_file.write_text(
        """
assets:
  - AAPL
  - MSFT
start_date: "2018-01-01"
end_date: "2024-01-01"
data_dir: "data/raw"
output_file: "prices.csv"
""",
        encoding="utf-8",
    )

    cfg = load_config(str(config_file))

    assert cfg["assets"] == ["AAPL", "MSFT"]
    assert cfg["start_date"] == "2018-01-01"
    assert cfg["output_file"] == "prices.csv"


def test_fetch_price_data_with_flat_columns():
    sample = make_sample_prices()

    with patch("src.data.download.yf.download", return_value=sample):
        prices = fetch_price_data(
            assets=["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"],
            start="2023-01-01",
            end="2023-02-01",
        )

    assert isinstance(prices, pd.DataFrame)
    assert not prices.empty
    assert list(prices.columns) == ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]
    assert prices.index.is_monotonic_increasing
    assert not prices.isnull().values.any()


def test_fetch_price_data_with_multiindex_columns():
    sample = make_sample_prices()
    multi = pd.concat({"Close": sample}, axis=1)

    with patch("src.data.download.yf.download", return_value=multi):
        prices = fetch_price_data(
            assets=["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"],
            start="2023-01-01",
            end="2023-02-01",
        )

    assert list(prices.columns) == ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]
    assert not prices.empty


def test_fetch_price_data_drops_nans():
    sample = make_sample_prices().copy()
    sample.iloc[2, 0] = None

    with patch("src.data.download.yf.download", return_value=sample):
        prices = fetch_price_data(
            assets=["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"],
            start="2023-01-01",
            end="2023-02-01",
        )

    assert not prices.isnull().values.any()
    assert len(prices) == 4


def test_fetch_price_data_empty_raises():
    empty = pd.DataFrame()

    with patch("src.data.download.yf.download", return_value=empty):
        with pytest.raises(ValueError, match="empty dataframe"):
            fetch_price_data(
                assets=["AAPL", "MSFT"],
                start="2023-01-01",
                end="2023-02-01",
            )


def test_save_and_load_roundtrip(tmp_path):
    sample = make_sample_prices()

    output_path = save_price_data(
        prices=sample,
        data_dir=str(tmp_path),
        output_file="prices.csv",
    )

    assert os.path.exists(output_path)

    loaded = load_price_data(
        data_dir=str(tmp_path),
        output_file="prices.csv",
    )

    assert list(loaded.columns) == list(sample.columns)
    assert loaded.index.is_monotonic_increasing
    assert not loaded.isnull().values.any()
    assert loaded.shape == sample.shape


def test_load_price_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_price_data(
            data_dir=str(tmp_path),
            output_file="does_not_exist.csv",
        )