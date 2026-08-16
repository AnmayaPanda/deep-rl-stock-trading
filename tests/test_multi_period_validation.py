import numpy as np

from src.environment.trading_env_weighted import TradingEnvironment
from src.agents.ddpg_agent import DDPGAgent


MODEL_PATH = "ddpg_train_2009_2015_weighted_10ep_no_turbulence.pt"

INITIAL_CAPITAL = 1_000_000.0

PERIODS = [
    ("2015-10-01", "2015-12-31"),
    ("2016-01-01", "2016-06-30"),
    ("2016-07-01", "2016-12-31"),
]


def calculate_metrics(portfolio_values):

    portfolio_values = np.asarray(
        portfolio_values,
        dtype=float,
    )

    daily_returns = (
        portfolio_values[1:]
        / portfolio_values[:-1]
        - 1.0
    )

    total_return = (
        portfolio_values[-1]
        / portfolio_values[0]
        - 1.0
    )

    if len(daily_returns) > 1:

        daily_std = np.std(
            daily_returns,
            ddof=1,
        )

        volatility = (
            daily_std
            * np.sqrt(252)
        )

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

    running_max = np.maximum.accumulate(
        portfolio_values
    )

    drawdowns = (
        portfolio_values
        / running_max
        - 1.0
    )

    max_drawdown = drawdowns.min()

    return {
        "final_portfolio": portfolio_values[-1],
        "return": total_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def create_environment(
    start_date,
    end_date,
):

    return TradingEnvironment(
        start_date=start_date,
        end_date=end_date,
        turbulence_threshold=None,
    )


def create_agent():

    agent = DDPGAgent(
        state_dim=175,
        action_dim=29,
        batch_size=32,
    )

    agent.load(MODEL_PATH)

    return agent


def run_ddpg(
    start_date,
    end_date,
):

    env = create_environment(
        start_date,
        end_date,
    )

    agent = create_agent()

    state = env.reset()

    portfolio_values = [
        env.portfolio_value
    ]

    for _ in range(env.n_steps - 1):

        action = agent.select_action(
            state,
            add_noise=False,
        )

        next_state, reward, done, info = (
            env.step(action)
        )

        portfolio_values.append(
            info["portfolio_value"]
        )

        state = next_state

        if done:
            break

    metrics = calculate_metrics(
        portfolio_values
    )

    metrics["transaction_costs"] = (
        env.total_transaction_costs
    )

    metrics["days"] = len(
        portfolio_values
    )

    return metrics


def run_equal_weight_buy_and_hold(
    start_date,
    end_date,
):

    env = create_environment(
        start_date,
        end_date,
    )

    env.reset()

    n_stocks = env.n_stocks

    # Equal allocation across all stocks.
    action = np.zeros(
        n_stocks,
        dtype=np.float32,
    )

    # Execute equal-weight allocation once.
    _, _, done, _ = env.step(action)

    portfolio_values = [
        env.portfolio_value
    ]

    # --------------------------------------------------------------
    # Hold the resulting positions.
    #
    # We deliberately do NOT call env.step() here because that would
    # rebalance the portfolio.
    # --------------------------------------------------------------

    while env.current_step < env.n_steps - 1:

        env.current_step += 1

        env.portfolio_value = (
            env._calculate_portfolio_value()
        )

        portfolio_values.append(
            env.portfolio_value
        )

    metrics = calculate_metrics(
        portfolio_values
    )

    metrics["transaction_costs"] = (
        env.total_transaction_costs
    )

    metrics["days"] = len(
        portfolio_values
    )

    return metrics


def run_cash(
    start_date,
    end_date,
):

    env = create_environment(
        start_date,
        end_date,
    )

    env.reset()

    portfolio_values = [
        INITIAL_CAPITAL
    ]

    for _ in range(
        env.n_steps - 1
    ):

        portfolio_values.append(
            INITIAL_CAPITAL
        )

    metrics = calculate_metrics(
        portfolio_values
    )

    metrics["transaction_costs"] = 0.0

    metrics["days"] = len(
        portfolio_values
    )

    return metrics


def print_strategy(
    name,
    metrics,
):

    print(
        f"{name:<22}"
        f"{metrics['return'] * 100:>10.2f}%"
        f"{metrics['volatility'] * 100:>11.2f}%"
        f"{metrics['sharpe']:>11.4f}"
        f"{metrics['max_drawdown'] * 100:>11.2f}%"
        f"{metrics['transaction_costs']:>15,.2f}"
    )


def run_period(
    start_date,
    end_date,
):

    print()
    print("=" * 82)
    print(
        f"PERIOD: {start_date} → {end_date}"
    )
    print("=" * 82)

    ddpg = run_ddpg(
        start_date,
        end_date,
    )

    equal_weight = (
        run_equal_weight_buy_and_hold(
            start_date,
            end_date,
        )
    )

    cash = run_cash(
        start_date,
        end_date,
    )

    print(
        f"Trading days: "
        f"{ddpg['days']}"
    )

    print()

    print(
        f"{'Strategy':<22}"
        f"{'Return':>10}"
        f"{'Volatility':>11}"
        f"{'Sharpe':>11}"
        f"{'Max DD':>11}"
        f"{'Costs':>15}"
    )

    print("-" * 82)

    print_strategy(
        "DDPG",
        ddpg,
    )

    print_strategy(
        "Equal Weight",
        equal_weight,
    )

    print_strategy(
        "Cash",
        cash,
    )

    print()

    excess_return = (
        ddpg["return"]
        - equal_weight["return"]
    )

    print(
        f"DDPG excess return vs "
        f"equal weight: "
        f"{excess_return * 100:+.2f}%"
    )

    return {
        "ddpg": ddpg,
        "equal_weight": equal_weight,
        "cash": cash,
    }


def main():

    print(
        "=== MULTI-PERIOD OUT-OF-SAMPLE "
        "VALIDATION ==="
    )

    print()
    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        "No retraining."
    )

    print(
        "Turbulence threshold: None"
    )

    results = []

    for start_date, end_date in PERIODS:

        result = run_period(
            start_date,
            end_date,
        )

        results.append(
            (
                start_date,
                end_date,
                result,
            )
        )

    print()
    print("=" * 82)
    print("SUMMARY")
    print("=" * 82)

    print()

    print(
        f"{'Period':<25}"
        f"{'DDPG':>10}"
        f"{'Equal Wt':>10}"
        f"{'Excess':>10}"
        f"{'DDPG Sharpe':>14}"
    )

    print("-" * 82)

    for (
        start_date,
        end_date,
        result,
    ) in results:

        ddpg = result["ddpg"]
        equal_weight = (
            result["equal_weight"]
        )

        excess = (
            ddpg["return"]
            - equal_weight["return"]
        )

        period = (
            f"{start_date[:7]}"
            f" → "
            f"{end_date[:7]}"
        )

        print(
            f"{period:<25}"
            f"{ddpg['return'] * 100:>9.2f}%"
            f"{equal_weight['return'] * 100:>9.2f}%"
            f"{excess * 100:>9.2f}%"
            f"{ddpg['sharpe']:>14.4f}"
        )

    print()
    print(
        "Interpretation:"
    )
    print(
        "Positive excess return means "
        "DDPG outperformed equal-weight "
        "buy & hold for that period."
    )


if __name__ == "__main__":
    main()