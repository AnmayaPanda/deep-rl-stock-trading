import numpy as np

from src.environment.trading_env_weighted import TradingEnvironment


env = TradingEnvironment(
    start_date="2009-01-01",
    end_date="2015-09-30",
    turbulence_threshold=None,
)

env.reset()

# Equal-weight portfolio
action = np.zeros(
    env.n_stocks,
    dtype=np.float32,
)

old_value = env._calculate_portfolio_value()

state, reward, done, info = env.step(action)

new_value = info["portfolio_value"]

manual_reward = (
    100.0
    * (new_value - old_value)
    / old_value
)

print("=== REWARD TEST ===")
print()

print("Old portfolio value:", old_value)
print("New portfolio value:", new_value)

print()
print("Environment reward:", reward)
print("Manual reward:", manual_reward)

print()
print(
    "Difference:",
    abs(reward - manual_reward),
)

print()
print("Transaction costs:", info["transaction_costs"])
print("Done:", done)

print()

if np.isclose(
    reward,
    manual_reward,
    atol=1e-6,
):
    print("PASS: Reward calculation is correct.")
else:
    print("FAIL: Reward calculation mismatch.")