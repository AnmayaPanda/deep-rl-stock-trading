import numpy as np

from src.environment.trading_env_weighted import TradingEnvironment


env = TradingEnvironment(
    start_date="2015-10-01",
    end_date="2015-12-31",
    turbulence_threshold=None,
)

env.reset()

print("=== ACTION → ALLOCATION SENSITIVITY ===")
print()

base = np.zeros(env.n_stocks, dtype=np.float32)

test_values = [
    -1.00,
    -0.99,
    -0.95,
    -0.90,
    -0.50,
     0.00,
     0.50,
     0.90,
     0.95,
     0.99,
     1.00,
]

for value in test_values:

    action = base.copy()

    # Change only stock 0
    action[0] = value

    target_shares = env._action_to_shares(action)

    prices = env.prices[env.current_step]

    portfolio_value = env.initial_cash

    position_values = (
        target_shares * prices
    )

    weights = (
        position_values / portfolio_value
    )

    print(
        f"Action[0]={value:+.2f} | "
        f"Target shares={target_shares[0]:6d} | "
        f"Stock 0 weight={weights[0]:.6f}"
    )