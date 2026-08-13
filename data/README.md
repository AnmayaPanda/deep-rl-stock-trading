# Data

This directory contains market data used by the project.

## Raw Data

Raw historical market data is downloaded using `yfinance`.

Raw data is intentionally excluded from Git because it can be
re-generated using the data acquisition pipeline.

Run:

```bash
python src/data/download.py