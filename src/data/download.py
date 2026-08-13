from pathlib import Path

import yfinance as yf


# Project paths
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"

# Initial universe: Dow Jones 30
# This allows us to first reproduce the paper's methodology.
DOW_30_TICKERS = [
    "MMM",   # 3M
    "AXP",   # American Express
    "AAPL",  # Apple
    "BA",    # Boeing
    "CAT",   # Caterpillar
    "CVX",   # Chevron
    "CSCO",  # Cisco
    "KO",    # Coca-Cola
    "DIS",   # Walt Disney
    "DD",    # DuPont
    "XOM",   # Exxon Mobil
    "GE",    # General Electric
    "GS",    # Goldman Sachs
    "HD",    # Home Depot
    "IBM",   # IBM
    "INTC",  # Intel
    "JNJ",   # Johnson & Johnson
    "JPM",   # JPMorgan Chase
    "MCD",   # McDonald's
    "MRK",   # Merck
    "MSFT",  # Microsoft
    "NKE",   # Nike
    "PFE",   # Pfizer
    "PG",    # Procter & Gamble
    "TRV",   # Travelers
    "UTX",   # United Technologies
    "UNH",   # UnitedHealth
    "VZ",    # Verizon
    "V",     # Visa
    "WMT",   # Walmart
]

START_DATE = "2009-01-01"
END_DATE = "2020-05-09"


def download_stock_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> None:
    """Download historical OHLCV data for the supplied tickers."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    failed_tickers = []

    for ticker in tickers:
        print(f"Downloading {ticker}...")

        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
            )

            if data.empty:
                print(f"  No data returned for {ticker}")
                failed_tickers.append(ticker)
                continue

            # yfinance can return a MultiIndex even for one ticker.
            if hasattr(data.columns, "levels"):
                data.columns = data.columns.get_level_values(0)

            data.reset_index(inplace=True)

            output_path = RAW_DATA_DIR / f"{ticker}.csv"
            data.to_csv(output_path, index=False)

            print(f"  Saved {len(data):,} rows -> {output_path}")

        except Exception as exc:
            print(f"  Failed: {exc}")
            failed_tickers.append(ticker)

    print("\nDownload complete.")

    if failed_tickers:
        print("\nFailed tickers:")
        for ticker in failed_tickers:
            print(f"  - {ticker}")


if __name__ == "__main__":
    download_stock_data(
        tickers=DOW_30_TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
    )