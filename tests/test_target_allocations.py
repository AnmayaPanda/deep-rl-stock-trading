import numpy as np

from src.environment.trading_env_weighted import TradingEnvironment


env = TradingEnvironment(
    start_date="2009-01-01",
    end_date="2015-09-30",
    turbulence_threshold=71.1,
)

env.reset()

action = np.zeros(
    env.n_stocks,
    dtype=np.float32,
)

# Strong preference for stock 0
action[0] = 1.0

# Neutral preference for stock 1
action[1] = 0.0

# Explicitly exclude stock 2
action[2] = -1.0


state, reward, done, info = env.step(action)

weights = state[30:59]

print("=== TARGET ALLOCATION TEST ===")
print()

print("Action[0]:", action[0])
print("Action[1]:", action[1])
print("Action[2]:", action[2])
print()

print("Weight[0]:", weights[0])
print("Weight[1]:", weights[1])
print("Weight[2]:", weights[2])
print()

print("Total equity weight:", weights.sum())
print("Cash weight:", state[0])
print("Cash + equity:", state[0] + weights.sum())
print()

print("Top allocations:")

for i in np.argsort(weights)[::-1][:10]:
    print(
        f"  Stock {i:2d}: "
        f"action={action[i]:+.1f} "
        f"weight={weights[i]:.6f}"
    )

print()

print("Excluded stock holdings:", env.holdings[2])
print("Excluded stock weight:", weights[2])

print()
print("Transaction costs:", info["transaction_costs"])
print("Portfolio value:", info["portfolio_value"])