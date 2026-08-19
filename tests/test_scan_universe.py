"""Characterization tests for swing_scan_universe (server.py:288-338).

These call the tool's raw `__wrapped__` function (bypassing the
`cache_result` decorator) to isolate the pure scanning/indicator logic from
known bug #1 (list-of-BaseModel results crash the cache_result decorator with
TypeError on every call -- see tests/test_cache_result.py, which documents
that bug directly). Calling the live decorated `server.swing_scan_universe`
here would raise TypeError before any of these assertions ever ran.
"""
import pandas as pd

import server
from conftest import make_ohlcv, make_multi_ticker_df

swing_scan_universe = server.swing_scan_universe.__wrapped__


def test_scan_universe_computes_indicators_and_trend_alignment(mocker):
    # ALIGNED: strictly increasing close -> close > ema20 > sma50 > sma200 holds.
    aligned_closes = [100.0 + i for i in range(210)]
    # NOT_ALIGNED: strictly decreasing close -> close < ema20 < sma50 < sma200.
    not_aligned_closes = [300.0 - i for i in range(210)]
    # Volume spike on the final day for ALIGNED so volume_ratio is checkable.
    aligned_volume = [1_000_000] * 209 + [3_000_000]

    aligned_df = make_ohlcv(aligned_closes, high_offset=1.5, low_offset=1.5, volume=aligned_volume)
    not_aligned_df = make_ohlcv(not_aligned_closes, high_offset=1.5, low_offset=1.5, volume=1_000_000)

    frame = make_multi_ticker_df({"ALIGNED": aligned_df, "NOTALIGNED": not_aligned_df})
    mocker.patch("server.yf.download", return_value=frame)

    results = swing_scan_universe(symbols=["ALIGNED", "NOTALIGNED"])
    by_symbol = {r.symbol: r for r in results}

    assert set(by_symbol) == {"ALIGNED", "NOTALIGNED"}

    # Independently recompute the same standard indicators against the same
    # synthetic series to cross-check the tool's arithmetic.
    close = aligned_df["Close"]
    expected_close = float(close.iloc[-1])
    expected_ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    expected_sma50 = float(close.rolling(50).mean().iloc[-1])
    expected_sma200 = float(close.rolling(200).mean().iloc[-1])
    high_low = aligned_df["High"] - aligned_df["Low"]
    high_close = (aligned_df["High"] - close.shift()).abs()
    low_close = (aligned_df["Low"] - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    expected_atr14 = float(tr.rolling(14).mean().iloc[-1])
    expected_vol_ratio = float(aligned_df["Volume"].iloc[-1] / aligned_df["Volume"].rolling(20).mean().iloc[-1])

    aligned_result = by_symbol["ALIGNED"]
    assert aligned_result.close == round(expected_close, 2)
    assert aligned_result.ema20 == round(expected_ema20, 2)
    assert aligned_result.sma50 == round(expected_sma50, 2)
    assert aligned_result.sma200 == round(expected_sma200, 2)
    assert aligned_result.atr14 == round(expected_atr14, 2)
    assert aligned_result.volume_ratio == round(expected_vol_ratio, 2)
    assert aligned_result.trend_aligned is True

    assert by_symbol["NOTALIGNED"].trend_aligned is False


def test_scan_universe_skips_symbols_with_insufficient_history(mocker):
    # Fewer than 50 rows -> should be silently dropped (len(df) < 50 guard),
    # not raise.
    short_df = make_ohlcv([100.0 + i for i in range(10)])
    long_df = make_ohlcv([100.0 + i for i in range(60)])
    frame = make_multi_ticker_df({"SHORT": short_df, "LONG": long_df})
    mocker.patch("server.yf.download", return_value=frame)

    results = swing_scan_universe(symbols=["SHORT", "LONG"])

    symbols = {r.symbol for r in results}
    assert symbols == {"LONG"}


def test_scan_universe_clips_symbols_list_to_100(mocker):
    """A list longer than 100 symbols must be silently truncated to the first
    100 before download, per the len > 100 guard at server.py:325-326."""
    import pandas as pd

    mocker.patch("server.yf.download", return_value=pd.DataFrame())

    oversized = [f"T{i:03d}" for i in range(102)]
    captured = {}

    original_download = __import__("yfinance").download

    def capturing_download(symbols, **kwargs):
        captured["symbols"] = symbols
        return pd.DataFrame()

    import server as srv
    mocker.patch("server.yf.download", side_effect=capturing_download)

    swing_scan_universe(symbols=oversized)

    assert len(captured["symbols"]) == 100
    assert captured["symbols"] == oversized[:100]


def test_scan_universe_returns_empty_list_when_download_returns_no_data(mocker):
    """An empty DataFrame from yf.download should produce [] without raising."""
    import pandas as pd

    mocker.patch("server.yf.download", return_value=pd.DataFrame())

    results = swing_scan_universe(symbols=["AAPL", "MSFT"])

    assert results == []
