"""Characterization tests for swing_get_market_regime (server.py:219-257)."""
import pytest

import server
from conftest import make_ohlcv, make_multi_ticker_df


def _regime_frame(vix, vix3m, spy_closes, rsp_closes=None):
    n = len(spy_closes)
    rsp_closes = rsp_closes or spy_closes
    return make_multi_ticker_df(
        {
            "^VIX": make_ohlcv([vix] * n),
            "^VIX3M": make_ohlcv([vix3m] * n),
            "SPY": make_ohlcv(spy_closes),
            "RSP": make_ohlcv(rsp_closes),
        }
    )


def test_bull_confirmed_when_contango_and_spy_uptrend(mocker):
    spy = [400 + i for i in range(25)]  # steady uptrend, ret_21 > 0
    frame = _regime_frame(vix=15.0, vix3m=20.0, spy_closes=spy)
    mocker.patch("server.yf.download", return_value=frame)

    result = server.swing_get_market_regime()

    assert result.vix_term_ratio < 0.95
    assert result.spy_21d_ret_pct > 0
    assert result.regime == "BULL_CONFIRMED"
    assert result.warning is None


def test_bear_defensive_on_vix_backwardation(mocker):
    spy = [400] * 25  # flat, doesn't matter: ratio > 1.0 forces BEAR
    frame = _regime_frame(vix=30.0, vix3m=20.0, spy_closes=spy)
    mocker.patch("server.yf.download", return_value=frame)

    result = server.swing_get_market_regime()

    assert result.vix_term_ratio > 1.0
    assert result.regime == "BEAR_DEFENSIVE"
    assert result.warning is not None


def test_bear_defensive_on_spy_selloff(mocker):
    # contango (ratio < 0.95) but SPY down more than 3% over 21 days
    spy = [400 - i for i in range(25)]  # steady downtrend
    frame = _regime_frame(vix=15.0, vix3m=20.0, spy_closes=spy)
    mocker.patch("server.yf.download", return_value=frame)

    result = server.swing_get_market_regime()

    assert result.vix_term_ratio < 0.95
    assert result.spy_21d_ret_pct < -3.0
    assert result.regime == "BEAR_DEFENSIVE"
    assert result.warning is not None


def test_neutral_caution_when_no_bull_or_bear_trigger(mocker):
    # ratio between 0.95 and 1.0, SPY mildly positive (not > threshold for bull's
    # "ratio < 0.95" branch, not down enough for bear)
    spy = [400.0] * 21 + [404.0, 404.0, 404.0, 404.0]  # small positive ret_21, well under 3% down
    frame = _regime_frame(vix=19.4, vix3m=20.0, spy_closes=spy)
    mocker.patch("server.yf.download", return_value=frame)

    result = server.swing_get_market_regime()

    assert 0.95 <= result.vix_term_ratio <= 1.0
    assert result.spy_21d_ret_pct > -3.0
    assert result.regime == "NEUTRAL_CAUTION"
    assert result.warning is not None


def test_missing_vix3m_column_raises_value_error(mocker):
    spy = [400 + i for i in range(25)]
    # Frame deliberately omits ^VIX3M entirely.
    frame = make_multi_ticker_df(
        {
            "^VIX": make_ohlcv([15.0] * 25),
            "SPY": make_ohlcv(spy),
            "RSP": make_ohlcv(spy),
        }
    )
    mocker.patch("server.yf.download", return_value=frame)

    with pytest.raises(ValueError, match="VIX3M"):
        server.swing_get_market_regime()


def test_zero_vix3m_raises_value_error(mocker):
    spy = [400 + i for i in range(25)]
    frame = _regime_frame(vix=15.0, vix3m=0.0, spy_closes=spy)
    mocker.patch("server.yf.download", return_value=frame)

    with pytest.raises(ValueError, match="VIX3M"):
        server.swing_get_market_regime()


def test_empty_vix_series_raises_rather_than_index_error(mocker):
    """^VIX column present but all-NaN (e.g. network hiccup for that ticker)
    must surface as a descriptive ValueError, not a bare IndexError from
    iloc[-1].  The guard mirrors the existing ^VIX3M empty-series check."""
    n = 25
    spy = [400 + i for i in range(n)]
    # Build a frame where ^VIX rows are all NaN so dropna() empties it.
    frame = make_multi_ticker_df(
        {
            "^VIX": make_ohlcv([float("nan")] * n),
            "^VIX3M": make_ohlcv([20.0] * n),
            "SPY": make_ohlcv(spy),
            "RSP": make_ohlcv(spy),
        }
    )
    mocker.patch("server.yf.download", return_value=frame)

    with pytest.raises(ValueError):
        server.swing_get_market_regime()
