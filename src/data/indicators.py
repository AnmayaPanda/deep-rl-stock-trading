from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config.yaml"

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"


def load_config() -> dict:
    """Load indicator parameters from config.yaml."""

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def calculate_macd(
    close: pd.Series,
    fast_period: int,
    slow_period: int,
    signal_period: int,
) -> pd.DataFrame:
    """Calculate MACD and MACD signal line."""

    fast_ema = close.ewm(
        span=fast_period,
        adjust=False,
    ).mean()

    slow_ema = close.ewm(
        span=slow_period,
        adjust=False,
    ).mean()

    macd = fast_ema - slow_ema

    signal = macd.ewm(
        span=signal_period,
        adjust=False,
    ).mean()

    return pd.DataFrame(
        {
            "MACD": macd,
            "MACD_signal": signal,
        }
    )


def calculate_rsi(
    close: pd.Series,
    period: int,
) -> pd.Series:
    """Calculate Relative Strength Index."""

    delta = close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = average_gain / average_loss

    rsi = 100 - (
        100 / (1 + relative_strength)
    )

    return rsi.rename("RSI")


def calculate_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """Calculate Commodity Channel Index."""

    typical_price = (
        high + low + close
    ) / 3

    moving_average = typical_price.rolling(
        window=period
    ).mean()

    mean_deviation = typical_price.rolling(
        window=period
    ).apply(
        lambda values: np.mean(
            np.abs(values - np.mean(values))
        ),
        raw=True,
    )

    cci = (
        (typical_price - moving_average)
        / (0.015 * mean_deviation)
    )

    return cci.rename("CCI")


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """Calculate Average Directional Index."""

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0,
        ),
        index=high.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0,
        ),
        index=high.index,
    )

    smoothed_tr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    smoothed_plus_dm = plus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    smoothed_minus_dm = minus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_di = (
        100 * smoothed_plus_dm / smoothed_tr
    )

    minus_di = (
        100 * smoothed_minus_dm / smoothed_tr
    )

    denominator = plus_di + minus_di

    dx = (
        100
        * (plus_di - minus_di).abs()
        / denominator.replace(0, np.nan)
    )

    adx = dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return adx.rename("ADX")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to a stock dataframe."""

    config = load_config()
    indicator_config = config["indicators"]

    macd_config = indicator_config["macd"]
    rsi_config = indicator_config["rsi"]
    cci_config = indicator_config["cci"]
    adx_config = indicator_config["adx"]

    macd = calculate_macd(
        df["Close"],
        fast_period=macd_config["fast_period"],
        slow_period=macd_config["slow_period"],
        signal_period=macd_config["signal_period"],
    )

    df["MACD"] = macd["MACD"]
    df["MACD_signal"] = macd["MACD_signal"]

    df["RSI"] = calculate_rsi(
        df["Close"],
        period=rsi_config["period"],
    )

    df["CCI"] = calculate_cci(
        df["High"],
        df["Low"],
        df["Close"],
        period=cci_config["period"],
    )

    df["ADX"] = calculate_adx(
        df["High"],
        df["Low"],
        df["Close"],
        period=adx_config["period"],
    )

    return df


def process_stock(file_path: Path) -> None:
    """Process one stock and save its feature dataset."""

    ticker = file_path.stem

    print(f"Processing {ticker}...")

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    df = add_indicators(df)

    output_path = PROCESSED_DATA_DIR / f"{ticker}.csv"

    df.to_csv(output_path, index=False)

    print(
        f"  Saved {len(df):,} rows -> {output_path}"
    )


def main() -> None:
    """Process all downloaded stock datasets."""

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        RAW_DATA_DIR.glob("*.csv")
    )

    if not files:
        print("No raw CSV files found.")
        return

    print(
        f"Found {len(files)} stock datasets.\n"
    )

    for file_path in files:
        process_stock(file_path)

    print("\nFeature generation complete.")


if __name__ == "__main__":
    main()