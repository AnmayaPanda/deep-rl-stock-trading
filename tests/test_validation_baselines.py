import numpy as np

from src.environment.trading_env_weighted import TradingEnvironment
from src.agents.ddpg_agent import DDPGAgent


MODEL_PATH = "ddpg_train_2009_2015_weighted_10ep_no_turbulence.pt"

START_DATE = "2015-10-01"
END_DATE = "2015-12-31"

INITIAL_CAPITAL = 1_000_000.0


def calculate_metrics(portfolio_values):
    portfolio_values = np.asarray(portfolio_values, dtype=float)

    daily_returns = (
        portfolio_values[1:] / portfolio_values[:-1] - 1.0
    )

    total_return = (
        portfolio_values[-1] / portfolio_values[0] - 1.0
    )

    if len(daily_returns) > 1:
        daily_std = np.std(daily_returns, ddof=1)

        volatility = daily_std * np.sqrt(252)

        if daily_std > 0:
            sharpe = (
                np.mean(daily_returns)
                / daily_std
            ) * np.sqrt(252)
        else:
            sharpe = 0.0
    else:
        volatility = 0.0
        sharpe = 0.0

    running_max = np.maximum.accumulate(portfolio_values)

    drawdowns = (
        portfolio_values / running_max - 1.0
    )

    max_drawdown = drawdowns.min()

    return {
        "final_portfolio": portfolio_values[-1],
        "return": total_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def run_ddpg():
    env = TradingEnvironment(
        start_date=START_DATE,
        end_date=END_DATE,
        turbulence_threshold=None,
    )

    agent = DDPGAgent(
        state_dim=175,
        action_dim=29,
        batch_size=32,
    )

    agent.load(MODEL_PATH)

    state = env.reset()

    portfolio_values = [env.portfolio_value]

    for _ in range(env.n_steps - 1):

        action = agent.select_action(
            state,
            add_noise=False,
        )

        next_state, reward, done, info = env.step(action)

        portfolio_values.append(
            info["portfolio_value"]
        )

        state = next_state

        if done:
            break

    metrics = calculate_metrics(portfolio_values)

    metrics["transaction_costs"] = (
        env.total_transaction_costs
    )

    return metrics

def run_action_diagnostics():
    env = TradingEnvironment(
        start_date=START_DATE,
        end_date=END_DATE,
        turbulence_threshold=None,
    )

    agent = DDPGAgent(
        state_dim=175,
        action_dim=29,
        batch_size=32,
    )

    agent.load(MODEL_PATH)

    state = env.reset()

    all_actions = []

    for _ in range(env.n_steps - 1):

        action = agent.select_action(
            state,
            add_noise=False,
        )

        all_actions.append(action.copy())

        next_state, reward, done, info = env.step(action)

        state = next_state

        if done:
            break

    actions = np.asarray(all_actions)

    print()
    print("=== ACTION DIAGNOSTICS ===")
    print()

    print(f"Shape: {actions.shape}")

    print(f"Mean action:       {actions.mean():.6f}")
    print(f"Mean |action|:     {np.abs(actions).mean():.6f}")
    print(f"Std action:        {actions.std():.6f}")
    print(f"Min action:        {actions.min():.6f}")
    print(f"Max action:        {actions.max():.6f}")

    print()
    print(
        f"|action| >= 0.95: "
        f"{(np.abs(actions) >= 0.95).mean() * 100:.2f}%"
    )

    print(
        f"|action| >= 0.99: "
        f"{(np.abs(actions) >= 0.99).mean() * 100:.2f}%"
    )

    print(
        f"action >= 0.95:   "
        f"{(actions >= 0.95).mean() * 100:.2f}%"
    )

    print(
        f"action <= -0.95:  "
        f"{(actions <= -0.95).mean() * 100:.2f}%"
    )

    print()
    print("Average number of stocks:")
    print(
        f"  action >= +0.95: "
        f"{(actions >= 0.95).sum(axis=1).mean():.2f}"
    )
    print(
        f"  action <= -0.95: "
        f"{(actions <= -0.95).sum(axis=1).mean():.2f}"
    )

    print()
    print("=== FIRST VALIDATION ACTION ===")

    for ticker, value in zip(
        env.tickers,
        actions[0],
    ):
        print(
            f"{ticker:<10} {value:+.6f}"
        )

    print()
    print("=== LAST VALIDATION ACTION ===")

    for ticker, value in zip(
        env.tickers,
        actions[-1],
    ):
        print(
            f"{ticker:<10} {value:+.6f}"
        )

    return actions

def run_equal_weight_buy_and_hold():
    env = TradingEnvironment(
        start_date=START_DATE,
        end_date=END_DATE,
        turbulence_threshold=None,
    )

    state = env.reset()

    n_stocks = env.n_stocks

    # Equal allocation across all stocks.
    action = np.zeros(
        n_stocks,
        dtype=np.float32,
    )

    # Buy the equal-weight portfolio on the first day.
    next_state, reward, done, info = env.step(action)

    portfolio_values = [
        env.portfolio_value
    ]

    # Hold the resulting positions.
    while not done:

        current_step = env.current_step

        if current_step >= env.n_steps - 1:
            break

        # Move to the next trading day without
        # changing the holdings.
        env.current_step += 1

        env.portfolio_value = (
            env._calculate_portfolio_value()
        )

        portfolio_values.append(
            env.portfolio_value
        )

        if env.current_step >= env.n_steps - 1:
            break

    metrics = calculate_metrics(
        portfolio_values
    )

    metrics["transaction_costs"] = (
        env.total_transaction_costs
    )

    return metrics


def run_cash():
    env = TradingEnvironment(
        start_date=START_DATE,
        end_date=END_DATE,
        turbulence_threshold=None,
    )

    env.reset()

    portfolio_values = [
        env.portfolio_value
    ]

    # Cash remains unchanged.
    for _ in range(env.n_steps - 1):
        portfolio_values.append(
            INITIAL_CAPITAL
        )

    metrics = calculate_metrics(
        portfolio_values
    )

    metrics["transaction_costs"] = 0.0

    return metrics


def print_results(name, metrics):
    print()
    print(name)
    print("-" * len(name))

    print(
        f"Final portfolio:       "
        f"{metrics['final_portfolio']:,.2f}"
    )

    print(
        f"Return:                "
        f"{metrics['return'] * 100:.2f}%"
    )

    print(
        f"Volatility:            "
        f"{metrics['volatility'] * 100:.2f}%"
    )

    print(
        f"Sharpe:                "
        f"{metrics['sharpe']:.4f}"
    )

    print(
        f"Maximum drawdown:      "
        f"{metrics['max_drawdown'] * 100:.2f}%"
    )

    print(
        f"Transaction costs:     "
        f"{metrics['transaction_costs']:,.2f}"
    )


def main():

    print("=== VALIDATION BASELINE COMPARISON ===")
    print()
    print(
        f"Period: {START_DATE} → {END_DATE}"
    )
    print(
        f"Initial capital: {INITIAL_CAPITAL:,.2f}"
    )
    print()

    ddpg = run_ddpg()

    equal_weight = run_equal_weight_buy_and_hold()

    cash = run_cash()

    print_results(
        "DDPG",
        ddpg,
    )

    print_results(
        "Equal Weight Buy & Hold",
        equal_weight,
    )

    print_results(
        "Cash",
        cash,
    )

    print()
    print("=== COMPARISON ===")
    print()

    print(
        f"{'Strategy':<25}"
        f"{'Return':>12}"
        f"{'Sharpe':>12}"
        f"{'Max DD':>12}"
    )

    print("-" * 61)

    for name, metrics in [
        ("DDPG", ddpg),
        ("Equal Weight", equal_weight),
        ("Cash", cash),
    ]:

        print(
            f"{name:<25}"
            f"{metrics['return'] * 100:>11.2f}%"
            f"{metrics['sharpe']:>12.4f}"
            f"{metrics['max_drawdown'] * 100:>11.2f}%"
        )


if __name__ == "__main__":
    run_action_diagnostics()