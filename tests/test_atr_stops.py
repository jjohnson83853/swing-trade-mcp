"""Characterization test for swing_calculate_atr_stops (server.py:763-800),
with expected values computed by hand.

Synthetic series: Close is flat at 100 every day, High = Close + 1,
Low = Close - 1. True range each day is therefore exactly:
  max(High-Low, |High-prevClose|, |Low-prevClose|) = max(2, 1, 1) = 2
so ATR14 (rolling mean of a constant 2) is exactly 2.0 for any window,
and highest_high_22 = max(High) over the trailing 22 rows = 101.
"""
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
