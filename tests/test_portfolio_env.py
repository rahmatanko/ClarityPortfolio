import numpy as np
import pandas as pd

from src.data.features import build_feature_frame, fit_standard_scaler, transform_with_scaler
from src.env.portfolio_env import PortfolioEnv


def make_sample_prices(n_rows: int = 80) -> pd.DataFrame:
    # deterministic fake dataset so tests stay stable and easy to debug
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
    # build a realistic feature matrix exactly the way the pipeline would
    features = build_feature_frame(prices)
    scaler = fit_standard_scaler(features)
    scaled = transform_with_scaler(features, scaler)
    return scaled


def test_env_reset_returns_correct_observation_shape():
    prices = make_sample_prices()
    features = make_scaled_features(prices)

    env = PortfolioEnv(prices=prices.loc[features.index], features=features)

    obs, info = env.reset()

    assert obs.shape == (features.shape[1] + 5 + 1,)
    assert "date" in info
    assert "previous_weights" in info


def test_env_reset_starts_all_cash():
    prices = make_sample_prices()
    features = make_scaled_features(prices)

    env = PortfolioEnv(prices=prices.loc[features.index], features=features)

    _, info = env.reset()

    prev = info["previous_weights"]
    assert prev.shape == (6,)
    assert np.allclose(prev[:-1], 0.0)
    assert np.isclose(prev[-1], 1.0)


def test_softmax_action_produces_valid_weights():
    prices = make_sample_prices()
    features = make_scaled_features(prices)

    env = PortfolioEnv(prices=prices.loc[features.index], features=features)
    env.reset()

    raw_action = np.array([0.5, -0.2, 1.5, 0.0, 0.7, -1.0], dtype=np.float32)
    weights = env._softmax(raw_action)

    assert weights.shape == (6,)
    assert np.all(weights >= 0.0)
    assert np.isclose(weights.sum(), 1.0)


def test_step_returns_valid_gymnasium_tuple():
    prices = make_sample_prices()
    features = make_scaled_features(prices)

    env = PortfolioEnv(prices=prices.loc[features.index], features=features)
    env.reset()

    action = np.zeros(6, dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (features.shape[1] + 5 + 1,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "weights" in info
    assert "gross_return" in info
    assert "transaction_cost" in info
    assert "net_return" in info
    assert "turnover" in info


def test_step_updates_previous_weights():
    prices = make_sample_prices()
    features = make_scaled_features(prices)

    env = PortfolioEnv(prices=prices.loc[features.index], features=features)
    env.reset()

    action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _, _, _, _, info = env.step(action)

    assert np.allclose(env.previous_weights, info["weights"])


def test_transaction_cost_is_positive_when_allocations_change():
    prices = make_sample_prices()
    features = make_scaled_features(prices)

    env = PortfolioEnv(
        prices=prices.loc[features.index],
        features=features,
        transaction_cost=0.01,
    )
    env.reset()

    # starting from all-cash means any non-cash allocation should incur cost
    action = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -2.0], dtype=np.float32)
    _, _, _, _, info = env.step(action)

    assert info["turnover"] > 0.0
    assert info["transaction_cost"] > 0.0


def test_reward_penalizes_negative_returns_more_than_net_return():
    prices = make_sample_prices()
    features = make_scaled_features(prices)

    env = PortfolioEnv(
        prices=prices.loc[features.index],
        features=features,
        downside_penalty_lambda=10.0,
    )
    env.reset()

    reward = env._compute_reward(net_return=-0.02)
    assert reward < -0.02


def test_zero_action_gives_uniform_weights():
    prices = make_sample_prices()
    features = make_scaled_features(prices)

    env = PortfolioEnv(prices=prices.loc[features.index], features=features)
    env.reset()

    weights = env._softmax(np.zeros(6, dtype=np.float32))

    assert np.allclose(weights, np.ones(6) / 6.0)


def test_episode_eventually_terminates():
    prices = make_sample_prices()
    features = make_scaled_features(prices)

    env = PortfolioEnv(prices=prices.loc[features.index], features=features)
    env.reset()

    terminated = False
    steps = 0

    while not terminated and steps < 1000:
        _, _, terminated, _, _ = env.step(np.zeros(6, dtype=np.float32))
        steps += 1

    assert terminated is True
    assert steps == len(features) - 1


def test_observation_contains_previous_portfolio_context():
    prices = make_sample_prices()
    features = make_scaled_features(prices)

    env = PortfolioEnv(prices=prices.loc[features.index], features=features)
    obs, _ = env.reset()

    risky_prev = obs[-6:-1]
    cash_prev = obs[-1]

    assert np.allclose(risky_prev, 0.0)
    assert np.isclose(cash_prev, 1.0)