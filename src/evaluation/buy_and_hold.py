import numpy as np

from src.environment.trading_env import TradingEnvironment


def evaluate():
    # --------------------------------------------------------------
    # Same validation period as DDPG
    # --------------------------------------------------------------

    env = TradingEnvironment(
        start_date="2015-10-01",
        end_date="2015-12-31",
    )

    state = env.reset()

    initial_cash = env.initial_cash
    initial_prices = env.prices[0]

    n_stocks = env.n_stocks

    # --------------------------------------------------------------
    # Equal-weight allocation
    # --------------------------------------------------------------

    allocation_per_stock = initial_cash / n_stocks

    shares = np.floor(
        allocation_per_stock / initial_prices
    )

    invested = np.dot(
        initial_prices,
        shares,
    )

    cash = initial_cash - invested

    # --------------------------------------------------------------
    # Hold until the final validation date
    # --------------------------------------------------------------

    final_prices = env.prices[-1]

    final_portfolio = (
        cash + np.dot(final_prices, shares)
    )

    cumulative_return = (
        final_portfolio / initial_cash
    ) - 1.0

    # --------------------------------------------------------------
    # Daily portfolio values
    # --------------------------------------------------------------

    portfolio_values = []

    for prices in env.prices:
        value = (
            cash + np.dot(prices, shares)
        )
        portfolio_values.append(value)

    portfolio_values = np.asarray(
        portfolio_values,
        dtype=np.float64,
    )

    daily_returns = (
        portfolio_values[1:]
        / portfolio_values[:-1]
    ) - 1.0

    if len(daily_returns) > 1:
        daily_std = np.std(
            daily_returns,
            ddof=1,
        )

        volatility = (
            daily_std * np.sqrt(252)
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

    # --------------------------------------------------------------
    # Maximum drawdown
    # --------------------------------------------------------------

    running_max = np.maximum.accumulate(
        portfolio_values
    )

    drawdowns = (
        portfolio_values / running_max
    ) - 1.0

    max_drawdown = np.min(drawdowns)

    # --------------------------------------------------------------
    # Results
    # --------------------------------------------------------------

    print("=== BUY & HOLD VALIDATION ===")
    print("Stocks:", n_stocks)
    print("Validation days:", env.n_steps)
    print(
        "Initial portfolio:",
        f"{initial_cash:.2f}",
    )
    print(
        "Final portfolio:",
        f"{final_portfolio:.2f}",
    )
    print(
        "Cumulative return:",
        f"{cumulative_return * 100:.2f}%",
    )
    print(
        "Annualized volatility:",
        f"{volatility * 100:.2f}%",
    )
    print(
        "Sharpe ratio:",
        f"{sharpe:.4f}",
    )
    print(
        "Maximum drawdown:",
        f"{max_drawdown * 100:.2f}%",
    )
    print(
        "Initial cash remaining:",
        f"{cash:.2f}",
    )
    print(
        "Total shares:",
        f"{shares.sum():.0f}",
    )


if __name__ == "__main__":
    evaluate()