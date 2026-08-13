from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DATA_DIR / "features.csv"

INDICATOR_COLUMNS = [
    "MACD",
    "RSI",
    "CCI",
    "ADX",
]

PRICE_COLUMNS = [
    "Close",
]


def load_stock_data(file_path: Path) -> pd.DataFrame:
    """Load one processed stock dataset."""

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])

    return df.sort_values("Date").reset_index(drop=True)


def find_common_valid_start(files: list[Path]) -> pd.Timestamp:
    """
    Find the earliest date on which every stock has
    valid values for all required features.
    """

    valid_start_dates = []

    for file_path in files:
        ticker = file_path.stem

        df = load_stock_data(file_path)

        required_columns = PRICE_COLUMNS + INDICATOR_COLUMNS

        valid_rows = df.dropna(
            subset=required_columns
        )

        if valid_rows.empty:
            raise ValueError(
                f"{ticker} has no rows with complete features."
            )

        first_valid_date = valid_rows["Date"].iloc[0]

        valid_start_dates.append(first_valid_date)

        print(
            f"{ticker}: first valid date = "
            f"{first_valid_date.date()}"
        )

    common_start = max(valid_start_dates)

    return common_start


def build_features(
    files: list[Path],
    start_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build a synchronized long-format feature dataset."""

    all_data = []

    for file_path in files:
        ticker = file_path.stem

        df = load_stock_data(file_path)

        df = df[
            df["Date"] >= start_date
        ].copy()

        required_columns = PRICE_COLUMNS + INDICATOR_COLUMNS

        df = df[
            ["Date"] + required_columns
        ]

        df["Ticker"] = ticker

        all_data.append(df)

    combined = pd.concat(
        all_data,
        ignore_index=True,
    )

    combined = combined.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    return combined


def validate_features(
    df: pd.DataFrame,
    expected_tickers: list[str],
) -> None:
    """Validate the synchronized feature dataset."""

    print("\n" + "=" * 80)
    print("FEATURE DATASET VALIDATION")
    print("=" * 80)

    required_columns = [
        "Date",
        "Ticker",
        "Close",
        "MACD",
        "RSI",
        "CCI",
        "ADX",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    actual_tickers = sorted(
        df["Ticker"].unique()
    )

    if actual_tickers != sorted(expected_tickers):
        raise ValueError(
            "Ticker mismatch.\n"
            f"Expected: {sorted(expected_tickers)}\n"
            f"Found: {actual_tickers}"
        )

    if df[INDICATOR_COLUMNS].isna().any().any():
        raise ValueError(
            "Feature dataset contains NaN values."
        )

    duplicate_rows = df.duplicated(
        subset=["Date", "Ticker"]
    ).sum()

    if duplicate_rows:
        raise ValueError(
            f"Found {duplicate_rows} duplicate "
            "(Date, Ticker) rows."
        )

    dates_per_ticker = (
        df.groupby("Ticker")["Date"]
        .nunique()
    )

    if dates_per_ticker.nunique() != 1:
        raise ValueError(
            "Stocks do not have the same number "
            "of trading dates."
        )

    unique_dates = df["Date"].nunique()

    expected_rows = (
        unique_dates * len(expected_tickers)
    )

    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} rows, "
            f"found {len(df):,}."
        )

    print(
        f"Stocks:       {len(expected_tickers)}"
    )

    print(
        f"Trading days: {unique_dates:,}"
    )

    print(
        f"Total rows:   {len(df):,}"
    )

    print(
        f"Start date:   {df['Date'].min().date()}"
    )

    print(
        f"End date:     {df['Date'].max().date()}"
    )

    print(
        f"NaN values:   "
        f"{df[INDICATOR_COLUMNS].isna().sum().sum()}"
    )

    print(
        f"Duplicates:   {duplicate_rows}"
    )

    print("\nValidation: PASS")


def main() -> None:
    """Build the synchronized feature dataset."""

    files = sorted(
        PROCESSED_DATA_DIR.glob("*.csv")
    )

    # Exclude the output file if the script is run again.
    files = [
        file_path
        for file_path in files
        if file_path.name != OUTPUT_PATH.name
    ]

    if not files:
        raise FileNotFoundError(
            "No processed stock datasets found."
        )

    print(
        f"Found {len(files)} stock datasets."
    )

    expected_tickers = [
        file_path.stem
        for file_path in files
    ]

    print("\nFinding common valid start date...\n")

    common_start = find_common_valid_start(files)

    print(
        "\nCommon valid start date:",
        common_start.date(),
    )

    features = build_features(
        files,
        common_start,
    )

    validate_features(
        features,
        expected_tickers,
    )

    features.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved feature dataset -> "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()