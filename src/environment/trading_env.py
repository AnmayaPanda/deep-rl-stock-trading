from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT_DIR / "data" / "processed" / "features.csv"


class TradingEnvironment:
    """
    Multi-stock trading environment.

    Action:
        -1 = SELL
         0 = HOLD
        +1 = BUY

    State:
        [cash,
         prices,
         holdings,
         MACD,
         RSI,
         CCI,
         ADX]

    For 29 stocks:
        state dimension = 1 + 29 * 6 = 175

    Timing convention:

        Observe state at t
            ↓
        Choose action
            ↓
        Execute trade using prices at t
            ↓
        Move to t+1
            ↓
        Value portfolio using prices at t+1
            ↓
        reward = V(t+1) - V(t)
    """

    def __init__(
        self,
        data_path: Path = FEATURES_PATH,
        initial_cash: float = 1_000_000.0,
        transaction_cost: float = 0.001,
    ):
        self.data_path = Path(data_path)
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost

        self.data = pd.read_csv(self.data_path)

        self.data["Date"] = pd.to_datetime(
            self.data["Date"]
        )

        self.tickers = sorted(
            self.data["Ticker"].unique()
        )

        self.n_stocks = len(self.tickers)

        self.dates = sorted(
            self.data["Date"].unique()
        )

        self.n_steps = len(self.dates)

        self.price_data = self._build_price_matrix()

        self.indicator_data = (
            self._build_indicator_matrices()
        )

        self.current_step = 0

        self.cash = self.initial_cash

        self.holdings = np.zeros(
            self.n_stocks,
            dtype=np.float64,
        )

        self.portfolio_value = (
            self.initial_cash
        )

    def _build_price_matrix(self):
        """Create Date × Stock close-price matrix."""

        pivot = self.data.pivot(
            index="Date",
            columns="Ticker",
            values="Close",
        )

        pivot = pivot.reindex(
            index=self.dates,
            columns=self.tickers,
        )

        return pivot.to_numpy(
            dtype=np.float64
        )

    def _build_indicator_matrices(self):
        """Create Date × Stock indicator matrices."""

        indicators = {}

        for indicator in [
            "MACD",
            "RSI",
            "CCI",
            "ADX",
        ]:
            pivot = self.data.pivot(
                index="Date",
                columns="Ticker",
                values=indicator,
            )

            pivot = pivot.reindex(
                index=self.dates,
                columns=self.tickers,
            )

            indicators[indicator] = pivot.to_numpy(
                dtype=np.float64
            )

        return indicators

    def _get_prices(self, step=None):
        """Get prices for a timestep."""

        if step is None:
            step = self.current_step

        return self.price_data[step]

    def _get_state(self):
        """Construct the current 175-dimensional state."""

        prices = self._get_prices()

        macd = self.indicator_data["MACD"][
            self.current_step
        ]

        rsi = self.indicator_data["RSI"][
            self.current_step
        ]

        cci = self.indicator_data["CCI"][
            self.current_step
        ]

        adx = self.indicator_data["ADX"][
            self.current_step
        ]

        state = np.concatenate(
            [
                np.array([self.cash]),
                prices,
                self.holdings,
                macd,
                rsi,
                cci,
                adx,
            ]
        )

        return state.astype(np.float32)

    def _calculate_portfolio_value(self, step=None):
        """Calculate portfolio value using prices at step."""

        prices = self._get_prices(step)

        stock_value = np.sum(
            self.holdings * prices
        )

        return self.cash + stock_value

    def reset(self):
        """Reset the environment."""

        self.current_step = 0

        self.cash = self.initial_cash

        self.holdings = np.zeros(
            self.n_stocks,
            dtype=np.float64,
        )

        self.portfolio_value = (
            self.initial_cash
        )

        return self._get_state()

    def _sell(self, indices, prices):
        """Sell holdings for selected stocks."""

        transaction_costs = 0.0

        for i in indices:

            if self.holdings[i] <= 0:
                continue

            shares = self.holdings[i]

            gross_value = (
                shares * prices[i]
            )

            cost = (
                gross_value
                * self.transaction_cost
            )

            self.cash += (
                gross_value - cost
            )

            self.holdings[i] = 0.0

            transaction_costs += cost

        return transaction_costs

    def _buy(self, indices, prices):
        """
        Buy selected stocks.

        Available cash is divided equally among
        all stocks receiving a BUY action.
        """

        if len(indices) == 0:
            return 0.0

        transaction_costs = 0.0

        cash_per_stock = (
            self.cash / len(indices)
        )

        for i in indices:

            if prices[i] <= 0:
                continue

            shares = (
                cash_per_stock
                / (
                    prices[i]
                    * (1 + self.transaction_cost)
                )
            )

            gross_value = (
                shares * prices[i]
            )

            cost = (
                gross_value
                * self.transaction_cost
            )

            total_cost = (
                gross_value + cost
            )

            if total_cost > self.cash:

                shares = (
                    self.cash
                    / (
                        prices[i]
                        * (1 + self.transaction_cost)
                    )
                )

                gross_value = (
                    shares * prices[i]
                )

                cost = (
                    gross_value
                    * self.transaction_cost
                )

                total_cost = (
                    gross_value + cost
                )

            self.holdings[i] += shares

            self.cash -= total_cost

            transaction_costs += cost

        return transaction_costs

    def step(self, actions):
        """
        Execute one trading step.

        The action is selected using state t.

        Trades are executed at prices from t.

        The environment then advances to t+1.

        Reward is based on the change in portfolio
        value from t to t+1.
        """

        actions = np.asarray(
            actions,
            dtype=np.int8,
        )

        if actions.shape != (
            self.n_stocks,
        ):
            raise ValueError(
                f"Expected {self.n_stocks} actions, "
                f"got shape {actions.shape}."
            )

        if not np.isin(
            actions,
            [-1, 0, 1],
        ).all():
            raise ValueError(
                "Actions must contain only "
                "-1, 0, or 1."
            )

        # ---------------------------------------------
        # Prices available when action is selected
        # ---------------------------------------------

        current_prices = self._get_prices()

        old_portfolio_value = (
            self._calculate_portfolio_value()
        )

        transaction_costs = 0.0

        # ---------------------------------------------
        # SELL
        # ---------------------------------------------

        sell_indices = np.where(
            actions == -1
        )[0]

        transaction_costs += self._sell(
            sell_indices,
            current_prices,
        )

        # ---------------------------------------------
        # BUY
        # ---------------------------------------------

        buy_indices = np.where(
            actions == 1
        )[0]

        transaction_costs += self._buy(
            buy_indices,
            current_prices,
        )

        # ---------------------------------------------
        # Move forward one trading day
        # ---------------------------------------------

        self.current_step += 1

        done = (
            self.current_step
            >= self.n_steps - 1
        )

        next_prices = self._get_prices()

        new_portfolio_value = (
            self._calculate_portfolio_value()
        )

        # ---------------------------------------------
        # Reward
        # ---------------------------------------------

        reward = (
            new_portfolio_value
            - old_portfolio_value
        )

        self.portfolio_value = (
            new_portfolio_value
        )

        next_state = self._get_state()

        info = {
            "date": self.dates[
                self.current_step
            ],
            "portfolio_value": (
                self.portfolio_value
            ),
            "cash": self.cash,
            "transaction_costs": (
                transaction_costs
            ),
            "previous_portfolio_value": (
                old_portfolio_value
            ),
            "current_prices": current_prices.copy(),
            "next_prices": next_prices.copy(),
        }

        return (
            next_state,
            float(reward),
            done,
            info,
        )