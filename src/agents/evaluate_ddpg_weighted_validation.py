import numpy as np
import torch

from src.agents.ddpg_agent import DDPGAgent
from src.environment.trading_env_weighted import TradingEnvironment


MODEL_PATH = "ddpg_train_2009_2015_weighted_10ep_no_turbulence.pt"

VALIDATION_START = "2015-10-01"
VALIDATION_END = "2015-12-31"


def evaluate():

    print("=== DDPG VALIDATION ===")
    print()

    # ----------------------------------------------------------
    # Validation environment
    # ----------------------------------------------------------

    env = TradingEnvironment(
        start_date=VALIDATION_START,
        end_date=VALIDATION_END,
        turbulence_threshold=None,
    )

    # ----------------------------------------------------------
    # Agent
    # ----------------------------------------------------------

    agent = DDPGAgent(
        state_dim=175,
        action_dim=29,
        batch_size=32,
    )

    agent.load(MODEL_PATH)

    print("Model:", MODEL_PATH)
    print("Validation period:", VALIDATION_START, "to", VALIDATION_END)
    print("Turbulence threshold: None")
    print("Stocks:", env.n_stocks)
    print("Trading days:", env.n_steps)
    print()

    # ----------------------------------------------------------
    # Reset
    # ----------------------------------------------------------

    state = env.reset()

    initial_portfolio = env.portfolio_value

    total_reward = 0.0
    portfolio_values = [initial_portfolio]
    actions = []

    # ----------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------

    for step in range(env.n_steps - 1):

        # IMPORTANT:
        # No exploration noise during evaluation.
        action = agent.select_action(
            state,
            add_noise=False,
        )

        actions.append(action.copy())

        next_state, reward, done, info = env.step(action)

        total_reward += reward
        portfolio_values.append(info["portfolio_value"])

        state = next_state

        if step < 5 or (step + 1) % 20 == 0:
            print(
                f"Step {step + 1}/{env.n_steps - 1} | "
                f"Reward: {reward:.4f} | "
                f"Portfolio: {info['portfolio_value']:.2f}"
            )

        if done:
            break

    # ----------------------------------------------------------
    # Statistics
    # ----------------------------------------------------------

    portfolio_values = np.asarray(portfolio_values)

    final_portfolio = portfolio_values[-1]

    total_return = (
        final_portfolio / initial_portfolio - 1.0
    )

    # Daily portfolio returns
    daily_returns = (
        portfolio_values[1:] / portfolio_values[:-1] - 1.0
    )

    if len(daily_returns) > 1:
        volatility = np.std(daily_returns, ddof=1) * np.sqrt(252)

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

    # ----------------------------------------------------------
    # Maximum drawdown
    # ----------------------------------------------------------

    running_max = np.maximum.accumulate(portfolio_values)

    drawdowns = (
        portfolio_values / running_max - 1.0
    )

    max_drawdown = drawdowns.min()

    # ----------------------------------------------------------
    # Action statistics
    # ----------------------------------------------------------

    actions = np.asarray(actions)

    action_abs_mean = np.mean(np.abs(actions))

    action_saturation_95 = np.mean(
        np.abs(actions) >= 0.95
    )

    action_saturation_99 = np.mean(
        np.abs(actions) >= 0.99
    )

    # ----------------------------------------------------------
    # Final report
    # ----------------------------------------------------------

    print()
    print("=== VALIDATION RESULTS ===")
    print()

    print(
        f"Initial portfolio:       "
        f"{initial_portfolio:,.2f}"
    )

    print(
        f"Final portfolio:         "
        f"{final_portfolio:,.2f}"
    )

    print(
        f"Total return:            "
        f"{total_return * 100:.2f}%"
    )

    print(
        f"Total reward:            "
        f"{total_reward:.4f}"
    )

    print(
        f"Annualized volatility:   "
        f"{volatility * 100:.2f}%"
    )

    print(
        f"Sharpe ratio:            "
        f"{sharpe:.4f}"
    )

    print(
        f"Maximum drawdown:        "
        f"{max_drawdown * 100:.2f}%"
    )

    print(
        f"Transaction costs:       "
        f"{env.total_transaction_costs:,.2f}"
    )

    print()

    print("=== ACTION STATISTICS ===")
    print()

    print(
        f"Mean |action|:           "
        f"{action_abs_mean:.4f}"
    )

    print(
        f"|action| >= 0.95:        "
        f"{action_saturation_95 * 100:.2f}%"
    )

    print(
        f"|action| >= 0.99:        "
        f"{action_saturation_99 * 100:.2f}%"
    )

    print()

    print("=== FINAL PORTFOLIO ===")
    print()

    print(
        "Final holdings:",
        env.holdings.astype(int)
    )

    print(
        f"Final cash:              "
        f"{env.cash:,.2f}"
    )

    print(
        f"Final portfolio value:   "
        f"{env.portfolio_value:,.2f}"
    )

    return {
        "initial_portfolio": initial_portfolio,
        "final_portfolio": final_portfolio,
        "total_return": total_return,
        "total_reward": total_reward,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "transaction_costs": env.total_transaction_costs,
        "action_abs_mean": action_abs_mean,
        "action_saturation_95": action_saturation_95,
        "action_saturation_99": action_saturation_99,
    }


if __name__ == "__main__":
    evaluate()