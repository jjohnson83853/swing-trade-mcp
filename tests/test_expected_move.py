"""Characterization tests for swing_get_expected_move (server.py edge-case guards)."""
from unittest.mock import MagicMock

import pandas as pd
import pytest

import server
from conftest import make_ohlcv


def _mock_ticker(mocker, spot, expirations, calls_by_exp):
    """expirations: list[str] date strings. calls_by_exp: dict[exp_str, calls_df]."""
    hist = make_ohlcv([spot])
    fake_ticker = MagicMock()
    fake_ticker.history.return_value = hist
    fake_ticker.expirations = expirations

    def option_chain(exp):
        chain = MagicMock()
        chain.calls = calls_by_exp[exp]
        return chain

    fake_ticker.option_chain.side_effect = option_chain
    mocker.patch("server.yf.Ticker", return_value=fake_ticker)
    return fake_ticker


def _calls_df(strikes, iv=0.30):
    return pd.DataFrame({"strike": strikes, "impliedVolatility": [iv] * len(strikes)})


def test_empty_calls_chain_raises_value_error(mocker):
    future_exp = (pd.Timestamp.now() + pd.Timedelta(days=21)).strftime("%Y-%m-%d")
    _mock_ticker(
        mocker,
        spot=100.0,
        expirations=[future_exp],
        calls_by_exp={future_exp: _calls_df([])},
    )

    with pytest.raises(ValueError, match="No call options available"):
        server.swing_get_expected_move(symbol="AAPL", target_days=21)


def test_past_expirations_filtered_out_picks_future_date(mocker):
    past_exp = (pd.Timestamp.now() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    future_exp = (pd.Timestamp.now() + pd.Timedelta(days=25)).strftime("%Y-%m-%d")

    _mock_ticker(
        mocker,
        spot=100.0,
        # past_exp is numerically closer to target_days=21 in abs(days_diff) terms
        # if negative diffs weren't filtered out (|-5 - 21| = 26 vs |25 - 21| = 4),
        # so this alone doesn't prove filtering; the key assertion is that the
        # past expiration is never selected.
        expirations=[past_exp, future_exp],
        calls_by_exp={
            past_exp: _calls_df([100.0]),
            future_exp: _calls_df([100.0]),
        },
    )

    result = server.swing_get_expected_move(symbol="AAPL", target_days=21)

    assert result.expiration_used == future_exp


def test_only_past_expirations_raises_value_error(mocker):
    past_exp = (pd.Timestamp.now() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    _mock_ticker(
        mocker,
        spot=100.0,
        expirations=[past_exp],
        calls_by_exp={past_exp: _calls_df([100.0])},
    )

    with pytest.raises(ValueError, match="No future options expirations"):
        server.swing_get_expected_move(symbol="AAPL", target_days=21)


def test_empty_expirations_list_raises_value_error(mocker):
    """t.expirations returning an empty tuple/list must raise ValueError with
    an appropriate message rather than propagating an index or type error."""
    hist = make_ohlcv([100.0])
    from unittest.mock import MagicMock
    fake_ticker = MagicMock()
    fake_ticker.history.return_value = hist
    fake_ticker.expirations = []
    mocker.patch("server.yf.Ticker", return_value=fake_ticker)

    with pytest.raises(ValueError, match="No options chains"):
        server.swing_get_expected_move(symbol="AAPL", target_days=21)


def test_expected_move_arithmetic_is_correct(mocker):
    """The expected-move formula is spot * iv * sqrt(actual_days / 365).
    With known inputs we can cross-check the output values exactly."""
    import numpy as np

    future_exp = (pd.Timestamp.now() + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    spot = 200.0
    iv = 0.40  # 40% IV

    _mock_ticker(
        mocker,
        spot=spot,
        expirations=[future_exp],
        calls_by_exp={future_exp: _calls_df([200.0], iv=iv)},
    )

    result = server.swing_get_expected_move(symbol="AAPL", target_days=30)

    actual_days = result.target_days  # may differ slightly from 30 due to calendar
    expected_em = spot * iv * np.sqrt(actual_days / 365.0)

    assert result.atm_implied_volatility == pytest.approx(iv * 100, abs=0.01)
    assert result.expected_move_dollars == pytest.approx(round(expected_em, 2), abs=0.01)
    # Production uses round(spot ± em, 2) not spot ± round(em, 2) — match that
    # to avoid rounding-boundary disagreements (e.g. round(208.765,2) may differ
    # from 200.0 + round(8.765,2) due to Python's banker's rounding).
    assert result.expected_upper_bound == pytest.approx(round(spot + expected_em, 2), abs=0.01)
    assert result.expected_lower_bound == pytest.approx(round(spot - expected_em, 2), abs=0.01)
