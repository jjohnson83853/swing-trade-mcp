"""Unit tests for the private helper functions _extract_close_column and
_compute_atr (server.py:114-125).

Both helpers are exercised transitively by the tool-level tests, but none of
those tests verify the helpers' own logic in isolation.  These tests do that
directly so a regression in either helper shows up with a precise failure
message rather than a confusing tool-level error.
"""
import numpy as np
import pandas as pd
import pytest

import server
from conftest import make_ohlcv, make_multi_ticker_df


# ---------------------------------------------------------------------------
# _extract_close_column
# ---------------------------------------------------------------------------

def test_extract_close_column_level0_multiindex():
    """When 'Close' appears at level-0 (the (field, ticker) layout), the
    function should return data["Close"] — a DataFrame whose columns are the
    individual tickers."""
    frame = make_multi_ticker_df(
        {
            "AAPL": make_ohlcv([100.0] * 5),
            "MSFT": make_ohlcv([200.0] * 5),
        }
    )
    # make_multi_ticker_df builds (field, ticker) layout, so level-0 values
    # are field names like "Close", "Open", etc.
    assert "Close" in frame.columns.get_level_values(0)

    result = server._extract_close_column(frame)

    assert isinstance(result, pd.DataFrame)
    assert "AAPL" in result.columns
    assert "MSFT" in result.columns
    assert set(result.columns) == {"AAPL", "MSFT"}


def test_extract_close_column_level1_multiindex():
    """When 'Close' appears at level-1 (the (ticker, field) layout), the
    function should fall back to xs('Close', level=1)."""
    frame = make_multi_ticker_df(
        {
            "AAPL": make_ohlcv([100.0] * 5),
            "MSFT": make_ohlcv([200.0] * 5),
        }
    )
    # Swap the MultiIndex levels to simulate (ticker, field) layout.
    frame.columns = frame.columns.swaplevel(0, 1)

    assert "Close" not in frame.columns.get_level_values(0)
    assert "Close" in frame.columns.get_level_values(1)

    result = server._extract_close_column(frame)

    assert isinstance(result, pd.DataFrame)
    assert "AAPL" in result.columns
    assert "MSFT" in result.columns


# ---------------------------------------------------------------------------
# _compute_atr
# ---------------------------------------------------------------------------

def test_compute_atr_constant_series_equals_expected():
    """With a flat OHLC series (Close=100, High=101, Low=99) the True Range
    each day is max(High-Low, |High-prevClose|, |Low-prevClose|) = max(2,1,1)=2,
    so ATR14 converges to exactly 2.0 once the rolling window is full."""
    df = make_ohlcv([100.0] * 25, high_offset=1.0, low_offset=1.0)
    atr = server._compute_atr(df, period=14)

    # The rolling window needs 14 bars; from bar 14 onward every value == 2.0.
    assert float(atr.iloc[-1]) == pytest.approx(2.0)


def test_compute_atr_returns_nan_before_window_filled():
    """The first `period-1` values must be NaN because the rolling window is
    not yet full."""
    df = make_ohlcv([100.0] * 25, high_offset=1.0, low_offset=1.0)
    atr = server._compute_atr(df, period=14)

    # pandas rolling(14).mean() gives NaN for indices 0..12 (13 NaN values).
    assert atr.iloc[:13].isna().all()


def test_compute_atr_larger_high_low_dominates():
    """When High-Low is larger than the close-to-close moves, ATR should
    reflect the High-Low range."""
    # Close flat at 100; High = Close + 5; Low = Close - 5 → TR = 10 every bar.
    df = make_ohlcv([100.0] * 25, high_offset=5.0, low_offset=5.0)
    atr = server._compute_atr(df, period=14)

    assert float(atr.iloc[-1]) == pytest.approx(10.0)
