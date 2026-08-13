from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"

REQUIRED_COLUMNS = {
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
}


def validate_file(file_path: Path) -> dict:
    """Validate one stock CSV file."""

    ticker = file_path.stem
    df = pd.read_csv(file_path)

    result = {
        "ticker": ticker,
        "rows": len(df),
        "start_date": None,
        "end_date": None,
        "duplicate_dates": 0,
        "missing_values": 0,
        "invalid_prices": 0,
        "status": "PASS",
    }

    # Check columns
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        result["status"] = "FAIL"
        print(
            f"{ticker}: missing columns -> "
            f"{sorted(missing_columns)}"
        )
        return result

    # Convert dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if df["Date"].isna().any():
        result["status"] = "FAIL"
        print(f"{ticker}: invalid dates found")

    # Date range
    result["start_date"] = df["Date"].min()
    result["end_date"] = df["Date"].max()

    # Duplicate dates
    result["duplicate_dates"] = int(df["Date"].duplicated().sum())

    if result["duplicate_dates"] > 0:
        result["status"] = "FAIL"

    # Missing values
    result["missing_values"] = int(
        df[list(REQUIRED_COLUMNS)].isna().sum().sum()
    )

    if result["missing_values"] > 0:
        result["status"] = "FAIL"

    # Invalid prices
    price_columns = ["Open", "High", "Low", "Close", "Adj Close"]

    result["invalid_prices"] = int(
        (df[price_columns] <= 0).sum().sum()
    )

    if result["invalid_prices"] > 0:
        result["status"] = "FAIL"

    return result


def main() -> None:
    """Validate all downloaded stock datasets."""

    files = sorted(RAW_DATA_DIR.glob("*.csv"))

    if not files:
        print("No CSV files found.")
        return

    print(f"Found {len(files)} stock datasets.\n")

    results = []

    for file_path in files:
        result = validate_file(file_path)
        results.append(result)

    results_df = pd.DataFrame(results)
    # Check whether all stocks share the same trading dates.
    date_sets = {}

    for file_path in files:
        df = pd.read_csv(file_path)
        dates = pd.to_datetime(df["Date"])
        date_sets[file_path.stem] = set(dates)

    reference_ticker = files[0].stem
    reference_dates = date_sets[reference_ticker]

    inconsistent_dates = {}

    for ticker, dates in date_sets.items():
        missing_dates = reference_dates - dates
        extra_dates = dates - reference_dates

        if missing_dates or extra_dates:
            inconsistent_dates[ticker] = {
                "missing": len(missing_dates),
                "extra": len(extra_dates),
            }

    print("\n" + "=" * 80)
    print("TRADING DATE CONSISTENCY")
    print("=" * 80)

    if not inconsistent_dates:
        print("All stocks have identical trading dates.")
    else:
        print("Inconsistent trading dates found:")

        for ticker, differences in inconsistent_dates.items():
            print(
                f"{ticker}: "
                f"{differences['missing']} missing, "
                f"{differences['extra']} extra"
            )

    print("=" * 80)
    print("DATA VALIDATION SUMMARY")
    print("=" * 80)

    print(results_df.to_string(index=False))

    print("\n" + "=" * 80)

    passed = (results_df["status"] == "PASS").sum()
    failed = (results_df["status"] == "FAIL").sum()

    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {len(results_df)}")


if __name__ == "__main__":
    main()