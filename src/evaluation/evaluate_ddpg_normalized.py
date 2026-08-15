import numpy as np

from src.agents.ddpg_agent import DDPGAgent
from src.environment.trading_env import TradingEnvironment


def evaluate():
    # --------------------------------------------------------------
    # Validation period
    # --------------------------------------------------------------

    env = TradingEnvironment(
        start_date="2015-10-01",
        end_date="2015-12-31",
        turbulence_threshold=71.1,
    )

    # --------------------------------------------------------------
    # Load trained DDPG
    # --------------------------------------------------------------

    agent = DDPGAgent(
        state_dim=175,
        action_dim=29,
        batch_size=32,
    )

    agent.load("ddpg_train_2009_2015_normalized.pt")

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------

    state = env.reset()

    initial_portfolio = env.portfolio_value

    portfolio_values = [initial_portfolio]
    rewards = []
    transaction_costs = []

    print("=== DDPG VALIDATION ===")
    print("Device:", agent.device)
    print("Stocks:", env.n_stocks)
    print("Validation days:", env.n_steps)
    print("Initial portfolio:", initial_portfolio)
    print()

    for step in range(env.n_steps - 1):

        # No exploration noise during evaluation.
        action = agent.select_action(
            state,
            add_noise=False,
        )

        next_state, reward, done, info = env.step(action)

        state = next_state

        rewards.append(reward)
        portfolio_values.append(
            info["portfolio_value"]
        )
        transaction_costs.append(
            info["transaction_costs"]
        )

        if step < 5 or (step + 1) % 20 == 0:
            print(
                f"Step {step + 1}/{env.n_steps - 1} | "
                f"Date: {info['date']} | "
                f"Reward: {reward:.2f} | "
                f"Portfolio: "
                f"{info['portfolio_value']:.2f} | "
                f"Shares: "
                f"{info['holdings'].sum():.0f}"
            )

        if done:
            break

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------

    portfolio_values = np.asarray(
        portfolio_values,
        dtype=np.float64,
    )

    rewards = np.asarray(
        rewards,
        dtype=np.float64,
    )

    final_portfolio = portfolio_values[-1]

    cumulative_return = (
        final_portfolio / initial_portfolio
    ) - 1.0

    daily_returns = (
        portfolio_values[1:]
        / portfolio_values[:-1]
    ) - 1.0

    if len(daily_returns) > 1:
        volatility = (
            np.std(daily_returns, ddof=1)
            * np.sqrt(252)
        )

        if np.std(daily_returns, ddof=1) > 0:
            sharpe = (
                np.mean(daily_returns)
                / np.std(daily_returns, ddof=1)
            ) * np.sqrt(252)
        else:
            sharpe = 0.0
    else:
        volatility = 0.0
        sharpe = 0.0

    running_max = np.maximum.accumulate(
        portfolio_values
    )

    drawdowns = (
        portfolio_values / running_max
    ) - 1.0

    max_drawdown = np.min(drawdowns)

    total_transaction_costs = sum(
        transaction_costs
    )

    # --------------------------------------------------------------
    # Results
    # --------------------------------------------------------------

    print()
    print("=== VALIDATION RESULTS ===")
    print(
        f"Initial portfolio: "
        f"{initial_portfolio:.2f}"
    )
    print(
        f"Final portfolio: "
        f"{final_portfolio:.2f}"
    )
    print(
        f"Cumulative return: "
        f"{cumulative_return * 100:.2f}%"
    )
    print(
        f"Annualized volatility: "
        f"{volatility * 100:.2f}%"
    )
    print(
        f"Sharpe ratio: "
        f"{sharpe:.4f}"
    )
    print(
        f"Maximum drawdown: "
        f"{max_drawdown * 100:.2f}%"
    )
    print(
        f"Total transaction costs: "
        f"{total_transaction_costs:.2f}"
    )
    print(
        f"Final holdings: "
        f"{env.holdings.sum():.0f}"
    )


if __name__ == "__main__":
    evaluate()