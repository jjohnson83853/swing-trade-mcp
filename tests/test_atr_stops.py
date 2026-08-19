"""Characterization test for swing_calculate_atr_stops (server.py:763-800),
with expected values computed by hand.

Synthetic series: Close is flat at 100 every day, High = Close + 1,
Low = Close - 1. True range each day is therefore exactly:
  max(High-Low, |High-prevClose|, |Low-prevClose|) = max(2, 1, 1) = 2
so ATR14 (rolling mean of a constant 2) is exactly 2.0 for any window,
and highest_high_22 = max(High) over the trailing 22 rows = 101.
"""
import pandas as pd
import pytest

import server
from conftest import make_ohlcv


def test_atr_stops_arithmetic_matches_hand_computed_values(mocker):
    df = make_ohlcv([100.0] * 25, high_offset=1.0, low_offset=1.0)
    mocker.patch("server.yf.download", return_value=df)

    entry_price = 150.0
    atr_multiplier = 2.0
    result = server.swing_calculate_atr_stops(symbol="TEST", entry_price=entry_price, atr_multiplier=atr_multiplier)

    assert result.current_atr_14 == 2.0
    # stop_loss = entry - multiplier*atr = 150 - 2*2 = 146
    assert result.suggested_stop_loss == 146.0
    # r_unit = entry - stop = 4
    assert result.target_1r == 154.0
    assert result.target_2r == 158.0
    assert result.target_3r == 162.0
    # chandelier = highest_high_22 (101) - 3*atr (6) = 95
    assert result.trailing_stop_chandelier == 95.0


def test_atr_stops_raises_value_error_when_no_price_data(mocker):
    """An empty DataFrame from yf.download must raise ValueError rather than
    cascade a pandas IndexError through _compute_atr."""
    mocker.patch("server.yf.download", return_value=pd.DataFrame())

    with pytest.raises(ValueError, match="No price data found"):
        server.swing_calculate_atr_stops(symbol="GONE", entry_price=150.0, atr_multiplier=2.0)


def test_atr_stops_non_default_multiplier_scales_stop_and_targets(mocker):
    """With atr_multiplier=3.0, stop_loss = entry - 3*ATR and targets scale
    accordingly.  ATR is 2.0 (same synthetic series), entry 100.0:
      stop = 100 - 6 = 94, r = 6, 1r=106, 2r=112, 3r=118."""
    df = make_ohlcv([100.0] * 25, high_offset=1.0, low_offset=1.0)
    mocker.patch("server.yf.download", return_value=df)

    result = server.swing_calculate_atr_stops(symbol="TEST", entry_price=100.0, atr_multiplier=3.0)

    assert result.current_atr_14 == pytest.approx(2.0)
    assert result.suggested_stop_loss == pytest.approx(94.0)
    assert result.target_1r == pytest.approx(106.0)
    assert result.target_2r == pytest.approx(112.0)
    assert result.target_3r == pytest.approx(118.0)
