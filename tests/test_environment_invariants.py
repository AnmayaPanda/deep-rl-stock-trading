import numpy as np

from src.environment.trading_env_weighted import TradingEnvironment


env = TradingEnvironment(
    start_date="2009-01-01",
    end_date="2015-09-30",
    turbulence_threshold=71.1,
)

print("=== ENVIRONMENT INVARIANT TESTS ===")
print("Stocks:", env.n_stocks)

state = env.reset()

print("State dimension:", state.shape)
print()


tests = [
    (
        "ALL -1",
        np.full(
            env.n_stocks,
            -1.0,
            dtype=np.float32,
        ),
    ),
    (
        "ALL 0",
        np.zeros(
            env.n_stocks,
            dtype=np.float32,
        ),
    ),
    (
        "ALL +1",
        np.ones(
            env.n_stocks,
            dtype=np.float32,
        ),
    ),
]


for name, action in tests:

    env.reset()

    state, reward, done, info = env.step(action)

    weights = state[30:59]

    print(name)

    print(
        "  Cash weight:",
        state[0],
    )

    print(
        "  Equity weight:",
        weights.sum(),
    )

    print(
        "  Weight min:",
        weights.min(),
    )

    print(
        "  Weight max:",
        weights.max(),
    )

    print(
        "  Cash + equity:",
        state[0] + weights.sum(),
    )

    print(
        "  Portfolio:",
        info["portfolio_value"],
    )

    print(
        "  Transaction costs:",
        info["transaction_costs"],
    )

    print(
        "  Negative cash:",
        env.cash < 0,
    )

    print(
        "  Negative holdings:",
        (env.holdings < 0).any(),
    )

    print(
        "  Holdings:",
        env.holdings.astype(int),
    )

    print()