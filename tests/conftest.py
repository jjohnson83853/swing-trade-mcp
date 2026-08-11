"""Shared fixtures and synthetic-data helpers for the swing-trade-mcp test suite.

No test in this package hits the network or the real repo cache.db: yfinance
calls are mocked per-test and the SQLite cache path is redirected to a
throwaway file under pytest's tmp_path for every test.
"""
import pandas as pd
import pytest

import server


@pytest.fixture(autouse=True)
def isolate_cache_db(tmp_path, monkeypatch):
    """Point the module-level DB_PATH at a throwaway file so no test ever
    touches the real repo cache.db."""
    monkeypatch.setattr(server, "DB_PATH", str(tmp_path / "test_cache.db"))
    yield


def make_ohlcv(prices, start="2023-01-02", high_offset=1.0, low_offset=1.0, volume=1_000_000):
    """Build a synthetic OHLCV DataFrame with a DatetimeIndex from a list of
    close prices. Volume may be a scalar (constant) or a list matching prices."""
    idx = pd.bdate_range(start=start, periods=len(prices))
    close = pd.Series([float(p) for p in prices], index=idx)
    vol = pd.Series(volume, index=idx)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + high_offset,
            "Low": close - low_offset,
            "Close": close,
            "Volume": vol,
        },
        index=idx,
    )


def make_multi_ticker_df(ticker_frames: dict) -> pd.DataFrame:
    """Combine {ticker: single-ticker OHLCV DataFrame} into the MultiIndex
    (field, ticker) column layout yf.download returns for multi-symbol requests."""
    pieces = {}
    for ticker, df in ticker_frames.items():
        for field in df.columns:
            pieces[(field, ticker)] = df[field]
    out = pd.concat(pieces, axis=1)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out
