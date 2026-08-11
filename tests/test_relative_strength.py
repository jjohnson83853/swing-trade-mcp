"""Characterization tests for swing_rank_relative_strength (server.py:366-425).

The ranking test below calls the tool's raw `__wrapped__` function (bypassing
the `cache_result` decorator) to isolate the ranking logic from known bug #1
(list-of-BaseModel results crash the cache_result decorator with TypeError on
every call -- see tests/test_cache_result.py, which documents that bug
directly). Calling the live decorated `server.swing_rank_relative_strength`
with a non-empty result would raise TypeError before any assertion ran.
"""
import server
from conftest import make_ohlcv, make_multi_ticker_df

swing_rank_relative_strength = server.swing_rank_relative_strength.__wrapped__


def test_rank_relative_strength_orders_leaders_above_laggards(mocker):
    n = 70  # > 63 so both lookback windows are populated at the last row

    def linear_closes(slope):
        return [100.0 + slope * t for t in range(n)]

    frame = make_multi_ticker_df(
        {
            "BM": make_ohlcv(linear_closes(0.5)),
            "HIGH": make_ohlcv(linear_closes(2.0)),   # clear outperformer
            "MED": make_ohlcv(linear_closes(0.6)),    # mild outperformer
            "LOW": make_ohlcv(linear_closes(0.1)),    # underperformer
        }
    )
    mocker.patch("server.yf.download", return_value=frame)

    results = swing_rank_relative_strength(symbols=["HIGH", "MED", "LOW"], benchmark="BM")

    assert [r.symbol for r in results] == ["HIGH", "MED", "LOW"]
    # results are sorted descending by percentile_rank
    ranks = [r.percentile_rank for r in results]
    assert ranks == sorted(ranks, reverse=True)
    assert results[0].rs_vs_bm_21d > 0
    assert results[-1].rs_vs_bm_21d < 0


def test_rank_relative_strength_returns_empty_when_no_data(mocker):
    import pandas as pd

    mocker.patch("server.yf.download", return_value=pd.DataFrame())

    results = server.swing_rank_relative_strength(symbols=["AAPL"], benchmark="SPY")

    assert results == []
