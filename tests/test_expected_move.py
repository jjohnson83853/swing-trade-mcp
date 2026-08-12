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
