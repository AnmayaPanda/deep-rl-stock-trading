from pathlib import Path

import numpy as np
import pandas as pd


class TradingEnvironment:
    """
    Paper-faithful multi-stock trading environment.

    Adaptation:
        - Paper: 30 stocks
        - This implementation: 29 stocks
        - Initial capital: $1,000,000
        - Transaction cost: 0.1%

    State:
        [cash,
         adjusted_close_prices,
         holdings,
         MACD,
         RSI,
         CCI,
         ADX]

    For 29 stocks:
        1 + 29 * 6 = 175 dimensions

    Action:
        Continuous vector in [-1, 1].
        Each value is converted into a number of shares.
            positive -> buy
            negative -> sell
            zero     -> hold
    """
    def _calculate_turbulence(self):
        """
        Calculate the daily market turbulence index.

        Turbulence is based on the Mahalanobis distance between
        the current stock-return vector and historical average returns.
        """

        returns = self.prices[1:] / self.prices[:-1] - 1.0

        turbulence = np.zeros(self.n_steps, dtype=np.float64)

        for t in range(1, self.n_steps):
            current_returns = returns[t - 1]

            historical_returns = returns[:t]

            if len(historical_returns) < 2:
                turbulence[t] = 0.0
                continue

            mean_returns = np.mean(
                historical_returns,
                axis=0,
            )

            covariance = np.cov(
                historical_returns,
                rowvar=False,
            )

            # Numerical stability
            covariance += np.eye(self.n_stocks) * 1e-6

            diff = current_returns - mean_returns

            try:
                inv_covariance = np.linalg.pinv(covariance)

                turbulence[t] = float(
                    diff.T
                    @ inv_covariance
                    @ diff
                )

            except np.linalg.LinAlgError:
                turbulence[t] = 0.0

        return turbulence

    def __init__(
        self,
        data_path="data/processed/features.csv",
        initial_cash=1_000_000.0,
        transaction_cost=0.001,
        max_trade_shares=100,
        turbulence_threshold=None,
        start_date=None,
        end_date=None,
    ):
        self.data_path = Path(data_path)

        self.initial_cash = float(initial_cash)
        self.transaction_cost_rate = float(transaction_cost)
        self.max_trade_shares = int(max_trade_shares)

        # Optional turbulence control.
        # We will implement the paper's turbulence calculation
        # separately before enabling this by default.
        self.turbulence_threshold = turbulence_threshold

        self.data = pd.read_csv(self.data_path)
        self.data["Date"] = pd.to_datetime(self.data["Date"])
        if start_date is not None:
            start_date = pd.Timestamp(start_date)
            self.data = self.data[
                self.data["Date"] >= start_date
            ]

        if end_date is not None:
            end_date = pd.Timestamp(end_date)
            self.data = self.data[
                self.data["Date"] <= end_date
            ]

        if self.data.empty:
            raise ValueError(
                "No data remains after applying the date range."
            )
        self.data = self.data.sort_values(
            ["Date", "Ticker"]
        ).reset_index(drop=True)

        self.tickers = sorted(self.data["Ticker"].unique())
        self.n_stocks = len(self.tickers)

        self.dates = sorted(self.data["Date"].unique())
        self.n_steps = len(self.dates)

        self._prepare_arrays()
        self.turbulence = self._calculate_turbulence()
        self.reset()

    # ------------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------------

    def _prepare_arrays(self):
        """
        Convert the long-form feature dataset into aligned
        date x stock arrays.
        """

        required_columns = [
            "Date",
            "Ticker",
            "Close",
            "MACD",
            "RSI",
            "CCI",
            "ADX",
        ]

        missing = [
            col for col in required_columns
            if col not in self.data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        price_df = self.data.pivot(
            index="Date",
            columns="Ticker",
            values="Close",
        ).reindex(columns=self.tickers)

        macd_df = self.data.pivot(
            index="Date",
            columns="Ticker",
            values="MACD",
        ).reindex(columns=self.tickers)

        rsi_df = self.data.pivot(
            index="Date",
            columns="Ticker",
            values="RSI",
        ).reindex(columns=self.tickers)

        cci_df = self.data.pivot(
            index="Date",
            columns="Ticker",
            values="CCI",
        ).reindex(columns=self.tickers)

        adx_df = self.data.pivot(
            index="Date",
            columns="Ticker",
            values="ADX",
        ).reindex(columns=self.tickers)

        self.prices = price_df.to_numpy(dtype=np.float64)
        self.macd = macd_df.to_numpy(dtype=np.float64)
        self.rsi = rsi_df.to_numpy(dtype=np.float64)
        self.cci = cci_df.to_numpy(dtype=np.float64)
        self.adx = adx_df.to_numpy(dtype=np.float64)

        if np.isnan(self.prices).any():
            raise ValueError("Price data contains NaN values.")

        if (
            np.isnan(self.macd).any()
            or np.isnan(self.rsi).any()
            or np.isnan(self.cci).any()
            or np.isnan(self.adx).any()
        ):
            raise ValueError(
                "Technical indicators contain NaN values."
            )

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(self):
        self.current_step = 0

        self.cash = self.initial_cash

        self.holdings = np.zeros(
            self.n_stocks,
            dtype=np.float64,
        )

        self.portfolio_value = self.initial_cash

        self.previous_portfolio_value = self.initial_cash

        self.total_transaction_costs = 0.0

        self.done = False

        return self._get_state()

    # ------------------------------------------------------------------
    # STATE
    # ------------------------------------------------------------------

    def _get_state(self):
        """
        Paper state:

        [cash,
         prices,
         holdings,
         MACD,
         RSI,
         CCI,
         ADX]
        """

        state = np.concatenate(
            [
                np.array([self.cash], dtype=np.float64),

                self.prices[self.current_step],

                self.holdings,

                self.macd[self.current_step],

                self.rsi[self.current_step],

                self.cci[self.current_step],

                self.adx[self.current_step],
            ]
        )

        return state.astype(np.float32)

    # ------------------------------------------------------------------
    # PORTFOLIO
    # ------------------------------------------------------------------

    def _calculate_portfolio_value(self, prices=None):
        if prices is None:
            prices = self.prices[self.current_step]

        return float(
            self.cash + np.dot(prices, self.holdings)
        )

    # ------------------------------------------------------------------
    # ACTION CONVERSION
    # ------------------------------------------------------------------

    def _action_to_shares(self, action):
        """
        Convert continuous [-1, 1] actions into share quantities.

        Example:

            action =  1.0 -> +max_trade_shares
            action =  0.5 -> +50% of max_trade_shares
            action =  0.0 -> 0 shares
            action = -0.5 -> -50% of max_trade_shares
            action = -1.0 -> -max_trade_shares
        """

        action = np.asarray(action, dtype=np.float64)

        if action.shape != (self.n_stocks,):
            raise ValueError(
                f"Expected action shape "
                f"({self.n_stocks},), "
                f"got {action.shape}"
            )

        action = np.clip(action, -1.0, 1.0)

        shares = np.floor(
            action * self.max_trade_shares
        ).astype(np.int64)

        return shares

    # ------------------------------------------------------------------
    # EXECUTE SELL
    # ------------------------------------------------------------------

    def _execute_sell(self, stock_idx, shares, price):
        """
        Sell shares while respecting current holdings.
        """

        shares = min(
            int(abs(shares)),
            int(self.holdings[stock_idx]),
        )

        if shares <= 0:
            return 0.0, 0.0

        gross_value = shares * price

        transaction_cost = (
            gross_value * self.transaction_cost_rate
        )

        net_proceeds = gross_value - transaction_cost

        self.cash += net_proceeds

        self.holdings[stock_idx] -= shares

        return net_proceeds, transaction_cost

    # ------------------------------------------------------------------
    # EXECUTE BUY
    # ------------------------------------------------------------------

    def _execute_buy(self, stock_idx, shares, price):
        """
        Buy as many requested shares as possible while
        maintaining non-negative cash.
        """

        shares = int(max(shares, 0))

        if shares <= 0:
            return 0.0, 0.0

        total_cost_per_share = (
            price * (1.0 + self.transaction_cost_rate)
        )

        affordable_shares = int(
            self.cash // total_cost_per_share
        )

        shares = min(
            shares,
            affordable_shares,
        )

        if shares <= 0:
            return 0.0, 0.0

        gross_value = shares * price

        transaction_cost = (
            gross_value * self.transaction_cost_rate
        )

        total_cost = gross_value + transaction_cost

        self.cash -= total_cost

        self.holdings[stock_idx] += shares

        return total_cost, transaction_cost

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------

    def step(self, action):
        """
        Execute action at time t.

        Prices then move to t+1.

        Reward:

            r_t = V_{t+1} - V_t

        with transaction costs incorporated into the
        portfolio balance.

        Returns:
            next_state
            reward
            done
            info
        """

        if self.done:
            raise RuntimeError(
                "Environment is done. Call reset()."
            )
    
        # --------------------------------------------------------------
        # Current prices
        # --------------------------------------------------------------

        current_prices = self.prices[self.current_step]

        # Portfolio value before action.
        old_portfolio_value = (
            self.cash
            + np.dot(current_prices, self.holdings)
        )

        # --------------------------------------------------------------
        # Convert continuous actions to share quantities
        # --------------------------------------------------------------

        requested_shares = self._action_to_shares(action)

        current_turbulence = self.turbulence[self.current_step]
        risk_off = (
            self.turbulence_threshold is not None
            and current_turbulence > self.turbulence_threshold
        )

        if risk_off:
            # High turbulence: liquidate all existing positions
            requested_shares = -self.holdings.astype(np.int64)

        transaction_costs = 0.0

        # --------------------------------------------------------------
        # SELL FIRST
        # --------------------------------------------------------------

        for i in range(self.n_stocks):

            if requested_shares[i] < 0:

                _, cost = self._execute_sell(
                    stock_idx=i,
                    shares=requested_shares[i],
                    price=current_prices[i],
                )

                transaction_costs += cost

        # --------------------------------------------------------------
        # BUY
        # --------------------------------------------------------------

        for i in range(self.n_stocks):

            if requested_shares[i] > 0:

                _, cost = self._execute_buy(
                    stock_idx=i,
                    shares=requested_shares[i],
                    price=current_prices[i],
                )

                transaction_costs += cost

        # --------------------------------------------------------------
        # Advance time
        # --------------------------------------------------------------

        self.current_step += 1

        if self.current_step >= self.n_steps - 1:

            self.current_step = self.n_steps - 1

            self.done = True

        next_prices = self.prices[self.current_step]

        # --------------------------------------------------------------
        # New portfolio value
        # --------------------------------------------------------------

        new_portfolio_value = (
            self.cash
            + np.dot(next_prices, self.holdings)
        )

        # --------------------------------------------------------------
        # Reward
        # --------------------------------------------------------------

        reward = (
            new_portfolio_value - old_portfolio_value
        ) / max(old_portfolio_value, 1e-8)

        # --------------------------------------------------------------
        # Update bookkeeping
        # --------------------------------------------------------------

        self.previous_portfolio_value = (
            old_portfolio_value
        )

        self.portfolio_value = (
            new_portfolio_value
        )

        self.total_transaction_costs += (
            transaction_costs
        )

        info = {
            "decision_date": self.dates[self.current_step - 1],
            "date": self.dates[self.current_step],
            "turbulence": float(current_turbulence),
            "turbulence_threshold": (
                float(self.turbulence_threshold)
                if self.turbulence_threshold is not None
                else None
            ),
            "risk_off": bool(risk_off),
            "cash": float(self.cash),
            "portfolio_value": float(
                self.portfolio_value
            ),
            "reward": float(reward),
            "transaction_costs": float(
                transaction_costs
            ),
            "total_transaction_costs": float(
                self.total_transaction_costs
            ),
            "holdings": self.holdings.copy(),
            "requested_shares": requested_shares.copy(),
        }

        return (
            self._get_state(),
            float(reward),
            self.done,
            info,
        )
    