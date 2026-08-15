from pathlib import Path

import numpy as np

from src.agents.ddpg_agent import DDPGAgent
from src.environment.trading_env_weighted import TradingEnvironment


MODEL_PATH = "ddpg_train_2009_2015_weighted.pt"

START_DATE = "2015-10-01"
END_DATE = "2015-12-31"

INITIAL_CASH = 1_000_000.0
TURBULENCE_THRESHOLD = 71.1


def main():

    print("=== DDPG POLICY DIAGNOSTICS ===")
    print(f"Model: {Path(MODEL_PATH).name}")

    # --------------------------------------------------------------
    # Environment
    # --------------------------------------------------------------

    env = TradingEnvironment(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_cash=INITIAL_CASH,
        turbulence_threshold=TURBULENCE_THRESHOLD,
    )

    state = env.reset()

    print(f"Stocks: {env.n_stocks}")
    print(f"Validation days: {env.n_steps}")

    # --------------------------------------------------------------
    # Agent
    # --------------------------------------------------------------

    agent = DDPGAgent(
        state_dim=state.shape[0],
        action_dim=env.n_stocks,
    )

    agent.load("ddpg_train_2009_2015_weighted.pt")

    # --------------------------------------------------------------
    # Collect diagnostics
    # --------------------------------------------------------------

    actions = []
    portfolio_values = []
    cash_values = []
    holdings_history = []
    turbulence_values = []
    risk_off_days = 0
    transaction_costs = []

    done = False

    while not done:

        action = agent.select_action(
            state,
            add_noise=False,
        )

        actions.append(action.copy())

        next_state, reward, done, info = env.step(action)

        portfolio_values.append(
            info["portfolio_value"]
        )

        cash_values.append(
            info["cash"]
        )

        holdings_history.append(
            info["holdings"].copy()
        )

        turbulence_values.append(
            info["turbulence"]
        )

        transaction_costs.append(
            info["transaction_costs"]
        )

        if info["risk_off"]:
            risk_off_days += 1

        state = next_state

    actions = np.asarray(actions)
    holdings_history = np.asarray(holdings_history)
    portfolio_values = np.asarray(portfolio_values)
    cash_values = np.asarray(cash_values)
    turbulence_values = np.asarray(turbulence_values)

    # --------------------------------------------------------------
    # ACTION STATISTICS
    # --------------------------------------------------------------

    print("\n=== ACTION STATISTICS ===")

    print(
        f"Mean: {actions.mean():.6f}"
    )

    print(
        f"Std: {actions.std():.6f}"
    )

    print(
        f"Min: {actions.min():.6f}"
    )

    print(
        f"Max: {actions.max():.6f}"
    )

    print(
        f"Mean absolute action: "
        f"{np.abs(actions).mean():.6f}"
    )

    print(
        f"Fraction positive: "
        f"{(actions > 0).mean() * 100:.2f}%"
    )

    print(
        f"Fraction negative: "
        f"{(actions < 0).mean() * 100:.2f}%"
    )

    print(
        f"Fraction zero: "
        f"{(actions == 0).mean() * 100:.2f}%"
    )

    print(
        f"Actions >= +0.95: "
        f"{(actions >= 0.95).mean() * 100:.2f}%"
    )

    print(
        f"Actions <= -0.95: "
        f"{(actions <= -0.95).mean() * 100:.2f}%"
    )

    # --------------------------------------------------------------
    # POSITION STATISTICS
    # --------------------------------------------------------------

    total_shares = holdings_history.sum(axis=1)

    print("\n=== POSITION STATISTICS ===")

    print(
        f"Initial shares: "
        f"{total_shares[0]:.0f}"
    )

    print(
        f"Final shares: "
        f"{total_shares[-1]:.0f}"
    )

    print(
        f"Maximum shares: "
        f"{total_shares.max():.0f}"
    )

    print(
        f"Minimum shares: "
        f"{total_shares.min():.0f}"
    )

    # --------------------------------------------------------------
    # CASH
    # --------------------------------------------------------------

    cash_percentage = (
        cash_values / portfolio_values
    )

    print("\n=== CASH UTILIZATION ===")

    print(
        f"Average cash: "
        f"${cash_values.mean():,.2f}"
    )

    print(
        f"Minimum cash: "
        f"${cash_values.min():,.2f}"
    )

    print(
        f"Maximum cash: "
        f"${cash_values.max():,.2f}"
    )

    print(
        f"Average cash percentage: "
        f"{cash_percentage.mean() * 100:.2f}%"
    )

    # --------------------------------------------------------------
    # FINAL PORTFOLIO ALLOCATION
    # --------------------------------------------------------------

    final_holdings = holdings_history[-1]

    final_prices = env.prices[env.current_step]

    final_values = (
        final_holdings * final_prices
    )

    final_portfolio = (
        env.cash + final_values.sum()
    )

    weights = (
        final_values / final_portfolio
    )

    print("\n=== FINAL PORTFOLIO ALLOCATION ===")

    ranked_indices = np.argsort(
        weights
    )[::-1]

    for idx in ranked_indices:

        if final_holdings[idx] <= 0:
            continue

        print(
            f"{env.tickers[idx]:<5} "
            f"weight={weights[idx] * 100:6.2f}% "
            f"shares={final_holdings[idx]:7.0f}"
        )

    # --------------------------------------------------------------
    # CONCENTRATION
    # --------------------------------------------------------------

    sorted_weights = np.sort(
        weights[weights > 0]
    )[::-1]

    print("\n=== CONCENTRATION ===")

    print(
        f"Top 1 weight: "
        f"{sorted_weights[:1].sum() * 100:.2f}%"
    )

    print(
        f"Top 5 weight: "
        f"{sorted_weights[:5].sum() * 100:.2f}%"
    )

    print(
        f"Top 10 weight: "
        f"{sorted_weights[:10].sum() * 100:.2f}%"
    )

    print(
        f"Number of stocks held: "
        f"{np.count_nonzero(final_holdings)}"
    )

    # --------------------------------------------------------------
    # TURBULENCE
    # --------------------------------------------------------------

    print("\n=== TURBULENCE ===")

    print(
        f"Maximum turbulence: "
        f"{turbulence_values.max():.4f}"
    )

    print(
        f"Threshold: "
        f"{TURBULENCE_THRESHOLD}"
    )

    print(
        f"Days above threshold: "
        f"{np.sum(turbulence_values > TURBULENCE_THRESHOLD)}"
    )

    print(
        f"Risk-off days encountered: "
        f"{risk_off_days}"
    )

    # --------------------------------------------------------------
    # TRANSACTIONS
    # --------------------------------------------------------------

    print("\n=== TRANSACTIONS ===")

    print(
        f"Total transaction costs: "
        f"${sum(transaction_costs):,.2f}"
    )

    # --------------------------------------------------------------
    # PORTFOLIO
    # --------------------------------------------------------------

    final_value = portfolio_values[-1]

    cumulative_return = (
        final_value / INITIAL_CASH - 1
    ) * 100

    print("\n=== PORTFOLIO ===")

    print(
        f"Initial: "
        f"${INITIAL_CASH:,.2f}"
    )

    print(
        f"Final: "
        f"${final_value:,.2f}"
    )

    print(
        f"Return: "
        f"{cumulative_return:.2f}%"
    )


if __name__ == "__main__":
    main()

