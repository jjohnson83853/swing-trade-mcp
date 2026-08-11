"""Characterization tests for swing_calculate_position_size (server.py:678-727).

Note: these tests call the tool function directly rather than through FastMCP,
so pydantic Field(...) defaults are never resolved (the raw FieldInfo object
would be bound as the default instead of e.g. []). Every test below passes
`current_holdings` explicitly to sidestep that harness quirk.
"""
from unittest.mock import MagicMock

import pytest

import server
from conftest import make_ohlcv, make_multi_ticker_df


def _mock_ticker(mocker, last_close):
    hist = make_ohlcv([last_close])
    fake_ticker = MagicMock()
    fake_ticker.history.return_value = hist
    mocker.patch("server.yf.Ticker", return_value=fake_ticker)


def test_position_size_share_math_normal_case(mocker):
    _mock_ticker(mocker, last_close=100.0)

    result = server.swing_calculate_position_size(
        symbol="NVDA", account_value=100_000.0, stop_price=95.0, risk_pct=1.0, current_holdings=[]
    )

    # dollar_risk_target = 100000 * 0.01 = 1000; risk_per_share = |100-95| = 5
    # shares = int(1000 / 5) = 200
    assert result.risk_per_share == 5.0
    assert result.allocated_shares == 200
    assert result.position_value == 20_000.0
    assert result.portfolio_allocation_pct == 20.0
    assert result.dollar_risk == 1000.0
    assert result.correlation_warnings == []


def test_position_size_zero_risk_per_share_raises(mocker):
    _mock_ticker(mocker, last_close=100.0)

    with pytest.raises(ValueError):
        server.swing_calculate_position_size(
            symbol="NVDA", account_value=100_000.0, stop_price=100.0, risk_pct=1.0, current_holdings=[]
        )


def test_position_size_flags_high_correlation_holding(mocker):
    _mock_ticker(mocker, last_close=100.0)

    n = 30
    # NVDA and AAPL move in perfect lockstep -> correlation should be ~1.0
    corr_frame = make_multi_ticker_df(
        {
            "NVDA": make_ohlcv([100.0 + i for i in range(n)]),
            "AAPL": make_ohlcv([50.0 + i for i in range(n)]),
        }
    )
    mocker.patch("server.yf.download", return_value=corr_frame)

    result = server.swing_calculate_position_size(
        symbol="NVDA",
        account_value=100_000.0,
        stop_price=95.0,
        risk_pct=1.0,
        current_holdings=["AAPL"],
    )

    assert len(result.correlation_warnings) == 1
    assert "AAPL" in result.correlation_warnings[0]


# --- Known bug #4: no input validation on account_value / risk_pct -------
# These document the gap called out in the architecture review. They assert
# the *intended* behavior (reject nonsensical negative inputs) and are
# expected to FAIL against the current implementation, which silently
# produces a nonsensical negative share count instead of raising.

def test_position_size_rejects_negative_account_value(mocker):
    _mock_ticker(mocker, last_close=100.0)

    with pytest.raises(ValueError):
        server.swing_calculate_position_size(
            symbol="NVDA", account_value=-100_000.0, stop_price=95.0, risk_pct=1.0, current_holdings=[]
        )


def test_position_size_rejects_negative_risk_pct(mocker):
    _mock_ticker(mocker, last_close=100.0)

    with pytest.raises(ValueError):
        server.swing_calculate_position_size(
            symbol="NVDA", account_value=100_000.0, stop_price=95.0, risk_pct=-1.0, current_holdings=[]
        )


# --- Known bug #5: KeyError if `symbol` itself is missing from the bulk ---
# correlation download (e.g. its own download failed/was delisted while a
# held ticker succeeded). Intended behavior: don't crash, just skip/warn.
# Expected to FAIL currently with KeyError from corr_matrix.loc[symbol, h].

def test_position_size_correlation_handles_missing_symbol_in_corr_matrix(mocker):
    _mock_ticker(mocker, last_close=100.0)

    n = 30
    # Correlation download only contains the held ticker, not `symbol` (NVDA) itself.
    corr_frame = make_multi_ticker_df({"AAPL": make_ohlcv([50.0 + i for i in range(n)])})
    mocker.patch("server.yf.download", return_value=corr_frame)

    # Should not raise even though NVDA never made it into the correlation matrix.
    result = server.swing_calculate_position_size(
        symbol="NVDA",
        account_value=100_000.0,
        stop_price=95.0,
        risk_pct=1.0,
        current_holdings=["AAPL"],
    )
    assert result.allocated_shares > 0
