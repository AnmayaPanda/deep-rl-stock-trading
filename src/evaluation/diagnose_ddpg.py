import numpy as np
import torch

from src.agents.ddpg_agent import DDPGAgent
from src.environment.trading_env import TradingEnvironment


MODEL_PATH = "ddpg_train_2009_2015.pt"


def main():

    env = TradingEnvironment(
        start_date="2015-10-01",
        end_date="2015-12-31",
    )

    agent = DDPGAgent(
        state_dim=175,
        action_dim=29,
    )

    agent.load(MODEL_PATH)

    state = env.reset()

    action_history = []
    portfolio_history = []
    holdings_history = []
    cash_history = []

    print("=== DDPG POLICY DIAGNOSTICS ===")
    print("Model:", MODEL_PATH)
    print("Stocks:", env.n_stocks)
    print("Validation days:", env.n_steps)
    print()

    for step in range(env.n_steps - 1):

        action = agent.select_action(
            state,
            add_noise=False,
        )

        action_history.append(action.copy())

        next_state, reward, done, info = env.step(
            action
        )

        portfolio_history.append(
            info["portfolio_value"]
        )

        holdings_history.append(
            info["holdings"].copy()
        )

        cash_history.append(
            info["cash"]
        )

        state = next_state

        if done:
            break

    # --------------------------------------------------------------
    # Convert to arrays
    # --------------------------------------------------------------

    actions = np.asarray(action_history)
    holdings = np.asarray(holdings_history)
    portfolios = np.asarray(portfolio_history)
    cash = np.asarray(cash_history)

    # --------------------------------------------------------------
    # Action statistics
    # --------------------------------------------------------------

    print("=== ACTION STATISTICS ===")

    print(
        "Mean:",
        f"{actions.mean():.6f}",
    )

    print(
        "Std:",
        f"{actions.std():.6f}",
    )

    print(
        "Min:",
        f"{actions.min():.6f}",
    )

    print(
        "Max:",
        f"{actions.max():.6f}",
    )

    print(
        "Mean absolute action:",
        f"{np.abs(actions).mean():.6f}",
    )

    print(
        "Fraction positive:",
        f"{(actions > 0).mean() * 100:.2f}%",
    )

    print(
        "Fraction negative:",
        f"{(actions < 0).mean() * 100:.2f}%",
    )

    print(
        "Fraction zero:",
        f"{(actions == 0).mean() * 100:.2f}%",
    )

    # --------------------------------------------------------------
    # Action saturation
    # --------------------------------------------------------------

    near_positive_one = np.mean(
        actions >= 0.95
    )

    near_negative_one = np.mean(
        actions <= -0.95
    )

    print(
        "Actions >= +0.95:",
        f"{near_positive_one * 100:.2f}%",
    )

    print(
        "Actions <= -0.95:",
        f"{near_negative_one * 100:.2f}%",
    )

    # --------------------------------------------------------------
    # Position statistics
    # --------------------------------------------------------------

    print()
    print("=== POSITION STATISTICS ===")

    total_holdings = holdings.sum(axis=1)

    print(
        "Initial shares:",
        f"{total_holdings[0]:.0f}",
    )

    print(
        "Final shares:",
        f"{total_holdings[-1]:.0f}",
    )

    print(
        "Maximum shares:",
        f"{total_holdings.max():.0f}",
    )

    print(
        "Minimum shares:",
        f"{total_holdings.min():.0f}",
    )

    # --------------------------------------------------------------
    # Cash utilization
    # --------------------------------------------------------------

    initial_cash = env.initial_cash

    cash_ratio = cash / initial_cash

    print()
    print("=== CASH UTILIZATION ===")

    print(
        "Average cash:",
        f"${cash.mean():,.2f}",
    )

    print(
        "Minimum cash:",
        f"${cash.min():,.2f}",
    )

    print(
        "Maximum cash:",
        f"${cash.max():,.2f}",
    )

    print(
        "Average cash percentage:",
        f"{cash_ratio.mean() * 100:.2f}%",
    )

    # --------------------------------------------------------------
    # Final portfolio allocation
    # --------------------------------------------------------------

    final_prices = env.prices[env.current_step]

    final_holdings = env.holdings.copy()

    final_values = (
        final_holdings * final_prices
    )

    final_portfolio = (
        env.cash + final_values.sum()
    )

    weights = final_values / final_portfolio

    print()
    print("=== FINAL PORTFOLIO ALLOCATION ===")

    ranked = np.argsort(
        weights
    )[::-1]

    for idx in ranked:

        if weights[idx] <= 0:
            continue

        print(
            f"{env.tickers[idx]:5s} "
            f"weight={weights[idx] * 100:6.2f}% "
            f"shares={final_holdings[idx]:8.0f}"
        )

    # --------------------------------------------------------------
    # Concentration
    # --------------------------------------------------------------

    print()
    print("=== CONCENTRATION ===")

    sorted_weights = np.sort(
        weights
    )[::-1]

    print(
        "Top 1 weight:",
        f"{sorted_weights[0] * 100:.2f}%",
    )

    print(
        "Top 5 weight:",
        f"{sorted_weights[:5].sum() * 100:.2f}%",
    )

    print(
        "Top 10 weight:",
        f"{sorted_weights[:10].sum() * 100:.2f}%",
    )

    print(
        "Number of stocks held:",
        f"{np.sum(final_holdings > 0)}",
    )

    # --------------------------------------------------------------
    # Portfolio result
    # --------------------------------------------------------------

    print()
    print("=== PORTFOLIO ===")

    print(
        "Initial:",
        f"${env.initial_cash:,.2f}",
    )

    print(
        "Final:",
        f"${env.portfolio_value:,.2f}",
    )

    print(
        "Return:",
        f"{(env.portfolio_value / env.initial_cash - 1) * 100:.2f}%",
    )


if __name__ == "__main__":
    main()