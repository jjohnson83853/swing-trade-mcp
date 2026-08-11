"""Characterization tests for swing_backtest_signal (server.py:547-634).

Note (bug #3, not asserted here): the docstring/Field description call
hold_days "calendar days," but the implementation slices `hold_days` rows of
the trading-day DataFrame (df.iloc[loc+1 : loc+1+hold_days]), i.e. trading
days. swing_get_expected_move's target_days is genuinely calendar days
(computed off real calendar date deltas), so the two tools are inconsistent.
This is a docs/semantics bug, not something with a clean pass/fail assertion.
"""
import pytest

import server
from conftest import make_ohlcv


def test_sma_crossover_backtest_triggers_and_expectancy_is_internally_consistent(mocker):
    # 60 flat days (ema20 ~= sma50) followed by a sustained uptrend, which
    # forces at least one ema20/sma50 golden-cross signal with plenty of
    # trading days afterward to fill a hold_days=21 exit window.
    flat = [100.0] * 60
    uptrend = [100.0 + 1.5 * i for i in range(1, 41)]
    df = make_ohlcv(flat + uptrend)
    mocker.patch("server.yf.download", return_value=df)

    result = server.swing_backtest_signal(symbol="TEST", signal_type="sma_crossover", hold_days=21, lookback_years=2)

    assert result.total_trades >= 1
    assert 0.0 <= result.win_rate <= 1.0

    # expectancy_pct must equal win_rate * avg_win_pct + (1 - win_rate) * avg_loss_pct
    recomputed = result.win_rate * result.avg_win_pct + (1 - result.win_rate) * result.avg_loss_pct
    assert recomputed == pytest.approx(result.expectancy_pct, abs=0.05)


def test_backtest_returns_zeroed_result_when_no_signal_triggers(mocker):
    # Flat price series never crosses (ema20 == sma50 throughout, no golden cross).
    df = make_ohlcv([100.0] * 80)
    mocker.patch("server.yf.download", return_value=df)

    result = server.swing_backtest_signal(symbol="FLAT", signal_type="sma_crossover", hold_days=21, lookback_years=2)

    assert result.total_trades == 0
    assert result.win_rate == 0.0
    assert result.expectancy_pct == 0.0


def test_invalid_signal_type_raises(mocker):
    df = make_ohlcv([100.0 + i for i in range(80)])
    mocker.patch("server.yf.download", return_value=df)

    with pytest.raises(ValueError):
        server.swing_backtest_signal(symbol="TEST", signal_type="not_a_real_signal", hold_days=21, lookback_years=2)
