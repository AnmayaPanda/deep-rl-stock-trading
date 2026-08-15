import numpy as np

from src.environment.trading_env_weighted import TradingEnvironment


env = TradingEnvironment(
    start_date="2009-01-01",
    end_date="2015-09-30",
    turbulence_threshold=None,
)

env.reset()

initial_cash = env.cash
price = env.prices[0, 0]

shares = 100

print("=== TRANSACTION COST TEST ===")
print()

print("Stock:", env.tickers[0])
print("Price:", price)
print("Shares:", shares)
print("Transaction cost rate:", env.transaction_cost_rate)
print()

# ------------------------------------------------------------
# BUY
# ------------------------------------------------------------

cash_before_buy = env.cash

gross_buy_value = shares * price
expected_buy_cost = (
    gross_buy_value
    * env.transaction_cost_rate
)

expected_buy_total = (
    gross_buy_value
    + expected_buy_cost
)

actual_buy_total, actual_buy_cost = (
    env._execute_buy(
        stock_idx=0,
        shares=shares,
        price=price,
    )
)

cash_after_buy = env.cash

print("=== BUY ===")

print("Gross value:", gross_buy_value)
print("Expected transaction cost:", expected_buy_cost)
print("Actual transaction cost:", actual_buy_cost)
print("Expected total cash reduction:", expected_buy_total)
print(
    "Actual cash reduction:",
    cash_before_buy - cash_after_buy,
)

print(
    "Cost difference:",
    actual_buy_cost - expected_buy_cost,
)

print()

# ------------------------------------------------------------
# SELL
# ------------------------------------------------------------

cash_before_sell = env.cash

holdings_before_sell = env.holdings[0]

gross_sell_value = shares * price
expected_sell_cost = (
    gross_sell_value
    * env.transaction_cost_rate
)

expected_sell_proceeds = (
    gross_sell_value
    - expected_sell_cost
)

actual_sell_proceeds, actual_sell_cost = (
    env._execute_sell(
        stock_idx=0,
        shares=shares,
        price=price,
    )
)

cash_after_sell = env.cash

print("=== SELL ===")

print("Gross value:", gross_sell_value)
print("Expected transaction cost:", expected_sell_cost)
print("Actual transaction cost:", actual_sell_cost)
print("Expected cash increase:", expected_sell_proceeds)
print(
    "Actual cash increase:",
    cash_after_sell - cash_before_sell,
)

print(
    "Cost difference:",
    actual_sell_cost - expected_sell_cost,
)

print()

# ------------------------------------------------------------
# FINAL CHECK
# ------------------------------------------------------------

print("=== FINAL CHECK ===")

print(
    "Holdings before sell:",
    holdings_before_sell,
)

print(
    "Holdings after sell:",
    env.holdings[0],
)

print(
    "Final cash:",
    env.cash,
)

print(
    "Initial cash:",
    initial_cash,
)

print(
    "Net cash difference:",
    env.cash - initial_cash,
)