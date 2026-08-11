import functools
import hashlib
import json
import logging
import os
import sqlite3
import time
import numpy as np
import pandas as pd
import yfinance as yf
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Setup startup and execution logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("swing-trade-mcp")

DB_PATH = os.getenv("CACHE_DB_PATH", "cache.db")
CACHE_TTL = 12 * 3600  # 12 hours in seconds

# Fixed: FastMCP initialized without removed keyword arguments 'description' or 'version'
mcp = FastMCP(os.getenv("MCP_SERVER_NAME", "swing-trade-mcp"))

# ------------------------------------------------------------------------------
# DATABASE & CACHING LAYER
# ------------------------------------------------------------------------------

def init_db():
    """Initialize SQLite cache table for persistence across sessions."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                timestamp REAL
            )
            """
        )
        conn.commit()


def get_cached_val(key: str):
    init_db()
    now = time.time()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value, timestamp FROM cache WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            val, ts = row
            if now - ts < CACHE_TTL:
                return json.loads(val)
            else:
                cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
    return None


def set_cached_val(key: str, value: dict | list):
    init_db()
    now = time.time()
    val_str = json.dumps(value)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, timestamp) VALUES (?, ?, ?)",
            (key, val_str, now),
        )
        conn.commit()


def cache_result(ttl_seconds: int = CACHE_TTL):
    """Decorator to cache tool execution outputs via SQLite."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            normalized_kwargs = {
                k: sorted(v) if isinstance(v, list) else v for k, v in kwargs.items()
            }
            raw_key = f"{func.__name__}:{args}:{sorted(normalized_kwargs.items())}"
            cache_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

            cached_data = get_cached_val(cache_key)
            if cached_data is not None:
                return cached_data

            result = func(*args, **kwargs)

            if isinstance(result, BaseModel):
                set_cached_val(cache_key, result.model_dump())
            elif isinstance(result, list) and all(isinstance(item, BaseModel) for item in result):
                set_cached_val(cache_key, [item.model_dump() for item in result])
            elif isinstance(result, (dict, list)) and not (
                isinstance(result, dict) and "error" in result
            ):
                set_cached_val(cache_key, result)

            return result
        return wrapper
    return decorator


# ------------------------------------------------------------------------------
# PYDANTIC RETURN SCHEMAS
# ------------------------------------------------------------------------------

class MarketRegimeResult(BaseModel):
    regime: str = Field(description="One of: BULL_CONFIRMED | NEUTRAL_CAUTION | BEAR_DEFENSIVE")
    vix: float = Field(description="Current VIX spot index reading")
    vix_3m: float = Field(description="3-month VIX3M index reading")
    vix_term_ratio: float = Field(description="Ratio of VIX to VIX3M (<1.0 = Contango/Safe, >1.0 = Backwardation/Stress)")
    term_structure: str = Field(description="Contango (Healthy) or Backwardation (Stress)")
    spy_21d_ret_pct: float = Field(description="SPY trailing 21-day return percentage")
    rsp_21d_ret_pct: float = Field(description="RSP (Equal Weight S&P 500) trailing 21-day return percentage")
    equal_weight_spread_pct: float = Field(description="Breadth spread (RSP return minus SPY return)")
    warning: str | None = Field(default=None, description="Human-readable warning if environment is defensive")


class ScanItem(BaseModel):
    symbol: str = Field(description="Stock ticker symbol")
    close: float = Field(description="Latest closing price in USD")
    ema20: float = Field(description="20-period Exponential Moving Average")
    sma50: float = Field(description="50-period Simple Moving Average")
    sma200: float = Field(description="200-period Simple Moving Average")
    atr14: float = Field(description="14-period Average True Range")
    volume_ratio: float = Field(description="Relative volume vs 20-day average volume")
    trend_aligned: bool = Field(description="True if Close > 20EMA > 50SMA > 200SMA")


class RelativeStrengthItem(BaseModel):
    symbol: str = Field(description="Stock ticker symbol")
    ret_21d_pct: float = Field(description="21-day percentage price change")
    ret_63d_pct: float = Field(description="63-day percentage price change")
    rs_vs_bm_21d: float = Field(description="Outperformance vs benchmark over 21 days (%)")
    rs_vs_bm_63d: float = Field(description="Outperformance vs benchmark over 63 days (%)")
    percentile_rank: float = Field(description="Relative strength rank from 0.0 to 99.0")


class ExpectedMoveResult(BaseModel):
    symbol: str = Field(description="Stock ticker symbol")
    spot_price: float = Field(description="Current underlying spot price in USD")
    target_days: int = Field(description="Holding window duration in calendar days")
    expiration_used: str = Field(description="Options chain expiration date used for IV extraction")
    atm_implied_volatility: float = Field(description="ATM call implied volatility percentage")
    expected_move_dollars: float = Field(description="Expected price move magnitude in USD")
    expected_move_pct: float = Field(description="Expected price move magnitude as percentage")
    expected_upper_bound: float = Field(description="Upper target price boundary")
    expected_lower_bound: float = Field(description="Lower stop/support boundary")


class BacktestSignalResult(BaseModel):
    symbol: str = Field(description="Stock ticker symbol")
    signal_type: str = Field(description="Signal tested: sma_crossover | rsi_oversold | breakout_20d")
    hold_days: int = Field(description="Fixed holding period in calendar days")
    total_trades: int = Field(description="Total trade setups triggered in sample window")
    win_rate: float = Field(description="Win rate percentage as decimal (0.0 to 1.0)")
    expectancy_pct: float = Field(description="Expected return per trade in %")
    avg_win_pct: float = Field(description="Average winning trade gain (%)")
    avg_loss_pct: float = Field(description="Average losing trade loss (%)")
    avg_mae_pct: float = Field(description="Average Maximum Adverse Excursion (%)")
    avg_mfe_pct: float = Field(description="Average Maximum Favorable Excursion (%)")


class PositionSizeResult(BaseModel):
    symbol: str = Field(description="Stock ticker symbol")
    entry_price: float = Field(description="Execution entry price in USD")
    stop_price: float = Field(description="Hard stop-loss exit price in USD")
    risk_per_share: float = Field(description="Dollar risk per share (Entry - Stop)")
    account_value: float = Field(description="Total portfolio account balance in USD")
    allocated_shares: int = Field(description="Recommended integer share count to buy")
    position_value: float = Field(description="Total position dollar commitment")
    portfolio_allocation_pct: float = Field(description="Percentage of portfolio committed")
    dollar_risk: float = Field(description="Total dollar amount risked on the position")
    correlation_warnings: list[str] = Field(description="Warnings for tickers correlated >0.70 with active holdings")


class ATRStopsResult(BaseModel):
    symbol: str = Field(description="Stock ticker symbol")
    entry_price: float = Field(description="Execution entry price in USD")
    current_atr_14: float = Field(description="Current 14-period Average True Range value")
    suggested_stop_loss: float = Field(description="Initial volatility stop loss price")
    target_1r: float = Field(description="1:1 Risk-to-Reward profit target")
    target_2r: float = Field(description="2:1 Risk-to-Reward profit target")
    target_3r: float = Field(description="3:1 Risk-to-Reward profit target")
    trailing_stop_chandelier: float = Field(description="3.0x ATR Chandelier trailing stop level")


# ------------------------------------------------------------------------------
# STAGE 1: REGIME DETECTION
# ------------------------------------------------------------------------------

@mcp.tool(
    name="swing_get_market_regime",
    description="""
Determine the current market regime using VIX term structure and SPY vs RSP breadth analysis to filter long swing entries.

USE WHEN: User asks about market conditions, broad market trend, bull market, bear market, risk-on, risk-off, SPY direction, VIX volatility level, market environment state, macro trend, or whether it is safe to enter long stock trades. Always call this FIRST before executing swing trade entry analysis.

DOMAIN KEYWORDS: market regime, bull market, bear market, defensive, BULL_CONFIRMED, NEUTRAL_CAUTION, BEAR_DEFENSIVE, VIX, VIX3M, term structure, contango, backwardation, SPY, RSP, market breadth, volatility, macro, swing trading, risk management, market health

INPUTS: None required.

OUTPUTS:
  - regime (str): One of BULL_CONFIRMED | NEUTRAL_CAUTION | BEAR_DEFENSIVE
  - vix (float): Current VIX spot reading
  - vix_3m (float): 3-month VIX3M reading
  - vix_term_ratio (float): VIX / VIX3M ratio
  - term_structure (str): Contango (Healthy) or Backwardation (Stress)
  - spy_21d_ret_pct (float): Trailing 21-day SPY percentage return
  - rsp_21d_ret_pct (float): Trailing 21-day RSP percentage return
  - equal_weight_spread_pct (float): Breadth spread (RSP - SPY)
  - warning (str | null): Human-readable warning if environment is defensive

EXAMPLE:
  swing_get_market_regime()

RELATED TOOLS:
  Call BEFORE: None - this is the entry tool.
  Call AFTER: swing_scan_universe, swing_rank_relative_strength
"""
)
@cache_result()
def swing_get_market_regime() -> MarketRegimeResult:
    tickers = ["^VIX", "^VIX3M", "SPY", "RSP"]
    data = yf.download(tickers, period="3mo", interval="1d", progress=False)

    if data.empty:
        raise ValueError("Failed to fetch market regime data.")

    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"] if "Close" in data.columns.get_level_values(0) else data.xs("Close", level=1, axis=1)

    vix = float(data["^VIX"].dropna().iloc[-1])
    vix3m = float(data["^VIX3M"].dropna().iloc[-1])
    vix_ratio = vix / vix3m

    spy_ret_21 = float(data["SPY"].dropna().pct_change(21).iloc[-1])
    rsp_ret_21 = float(data["RSP"].dropna().pct_change(21).iloc[-1])
    breadth_spread = rsp_ret_21 - spy_ret_21

    warning = None
    if vix_ratio < 0.95 and spy_ret_21 > 0:
        regime = "BULL_CONFIRMED"
    elif vix_ratio > 1.0 or spy_ret_21 < -0.03:
        regime = "BEAR_DEFENSIVE"
        warning = "BEAR_DEFENSIVE mode active. Volatility spikes or downtrend detected. Avoid aggressive long trades."
    else:
        regime = "NEUTRAL_CAUTION"
        warning = "NEUTRAL_CAUTION mode active. Tighten position sizing and select high-conviction setups only."

    return MarketRegimeResult(
        regime=regime,
        vix=round(vix, 2),
        vix_3m=round(vix3m, 2),
        vix_term_ratio=round(vix_ratio, 3),
        term_structure="Contango (Healthy)" if vix_ratio < 1.0 else "Backwardation (Stress)",
        spy_21d_ret_pct=round(spy_ret_21 * 100, 2),
        rsp_21d_ret_pct=round(rsp_ret_21 * 100, 2),
        equal_weight_spread_pct=round(breadth_spread * 100, 2),
        warning=warning,
    )


# ------------------------------------------------------------------------------
# STAGE 2: UNIVERSE SCANNING & RELATIVE STRENGTH
# ------------------------------------------------------------------------------

@mcp.tool(
    name="swing_scan_universe",
    description="""
Scans a batch of up to 100 ticker symbols for moving average alignment, volume expansion, and Average True Range metrics.

USE WHEN: User asks to scan stocks, find swing trading setups, filter candidate tickers, check 20 EMA, 50 SMA, 200 SMA trend alignment, locate volume surges, or screen equities for momentum setups.

DOMAIN KEYWORDS: scan universe, technical screening, moving averages, 20EMA, 50SMA, 200SMA, trend alignment, ATR14, relative volume, volume surge, stock watchlist, momentum setups, technical alignment, swing candidate

INPUTS:
  - symbols (list[str]): List of stock ticker strings (e.g. ['AAPL', 'NVDA', 'MSFT']). Max 100 tickers per request.

OUTPUTS:
  - List of objects containing symbol, close, ema20, sma50, sma200, atr14, volume_ratio, and trend_aligned boolean.

EXAMPLE:
  swing_scan_universe(symbols=['AAPL', 'NVDA', 'AMD'])

RELATED TOOLS:
  Call BEFORE: swing_get_market_regime
  Call AFTER: swing_rank_relative_strength, swing_get_expected_move
"""
)
@cache_result()
def swing_scan_universe(
    symbols: list[str] = Field(description="List of stock tickers to scan. Max 100 items. Example: ['AAPL', 'MSFT']")
) -> list[ScanItem]:
    if len(symbols) > 100:
        symbols = symbols[:100]

    data = yf.download(symbols, period="6mo", interval="1d", progress=False)
    if data.empty:
        return []

    out = []
    for s in symbols:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df = data.xs(s, level=1, axis=1) if s in data.columns.get_level_values(1) else data.xs(s, level=0, axis=1)
            else:
                df = data

            df = df.dropna()
            if df.empty or len(df) < 50:
                continue

            close = float(df["Close"].iloc[-1])
            ema20 = float(df["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
            sma50 = float(df["Close"].rolling(50).mean().iloc[-1])
            sma200 = float(df["Close"].rolling(200).mean().iloc[-1])

            high_low = df["High"] - df["Low"]
            high_close = np.abs(df["High"] - df["Close"].shift())
            low_close = np.abs(df["Low"] - df["Close"].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr14 = float(tr.rolling(14).mean().iloc[-1])

            vol_ratio = float(df["Volume"].iloc[-1] / df["Volume"].rolling(20).mean().iloc[-1])

            out.append(
                ScanItem(
                    symbol=s,
                    close=round(close, 2),
                    ema20=round(ema20, 2),
                    sma50=round(sma50, 2),
                    sma200=round(sma200, 2),
                    atr14=round(atr14, 2),
                    volume_ratio=round(vol_ratio, 2),
                    trend_aligned=bool(close > ema20 > sma50 > sma200),
                )
            )
        except Exception:
            continue

    return out


@mcp.tool(
    name="swing_rank_relative_strength",
    description="""
Ranks candidate stock tickers by relative strength performance against a benchmark index (e.g., SPY) over 21-day and 63-day lookback windows.

USE WHEN: User wants to rank stocks by relative strength, identify market outperforming tickers, compare momentum leaders against SPY/QQQ, evaluate Mansfield relative strength, or pick top leaders in a sector list.

DOMAIN KEYWORDS: relative strength, rank momentum, outperform benchmark, SPY, QQQ, market leaders, alpha generation, composite score, percentile rank, 21d momentum, 63d momentum, relative performance, stock selection

INPUTS:
  - symbols (list[str]): List of candidate ticker symbols to rank.
  - benchmark (str, default 'SPY'): Target benchmark ticker symbol.

OUTPUTS:
  - List of objects containing symbol, ret_21d_pct, ret_63d_pct, rs_vs_bm_21d, rs_vs_bm_63d, and percentile_rank.

EXAMPLE:
  swing_rank_relative_strength(symbols=['NVDA', 'AMD', 'INTC'], benchmark='SPY')

RELATED TOOLS:
  Call BEFORE: swing_scan_universe
  Call AFTER: swing_get_expected_move, swing_backtest_signal
"""
)
@cache_result()
def swing_rank_relative_strength(
    symbols: list[str] = Field(description="Candidate ticker list. Example: ['NVDA', 'AAPL', 'MSFT']"),
    benchmark: str = Field(default="SPY", description="Benchmark symbol to compare against. Default 'SPY'")
) -> list[RelativeStrengthItem]:
    all_tickers = list(set(symbols + [benchmark]))
    data = yf.download(all_tickers, period="6mo", interval="1d", progress=False)

    if data.empty:
        return []

    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"] if "Close" in data.columns.get_level_values(0) else data.xs("Close", level=1, axis=1)

    returns_21 = data.pct_change(21).iloc[-1]
    returns_63 = data.pct_change(63).iloc[-1]

    if benchmark not in returns_21:
        return []

    bm_21 = returns_21[benchmark]
    bm_63 = returns_63[benchmark]

    results = []
    for s in symbols:
        if s not in data.columns or s == benchmark:
            continue

        rs_21 = float(returns_21[s] - bm_21)
        rs_63 = float(returns_63[s] - bm_63)
        composite_score = (rs_21 * 0.4) + (rs_63 * 0.6)

        results.append(
            {
                "symbol": s,
                "ret_21d_pct": round(float(returns_21[s]) * 100, 2),
                "ret_63d_pct": round(float(returns_63[s]) * 100, 2),
                "rs_vs_bm_21d": round(rs_21 * 100, 2),
                "rs_vs_bm_63d": round(rs_63 * 100, 2),
                "composite_score": composite_score,
            }
        )

    res_df = pd.DataFrame(results)
    if res_df.empty:
        return []

    res_df["percentile_rank"] = (res_df["composite_score"].rank(pct=True) * 99).round(0)
    res_df = res_df.sort_values(by="percentile_rank", ascending=False)

    return [
        RelativeStrengthItem(
            symbol=row["symbol"],
            ret_21d_pct=row["ret_21d_pct"],
            ret_63d_pct=row["ret_63d_pct"],
            rs_vs_bm_21d=row["rs_vs_bm_21d"],
            rs_vs_bm_63d=row["rs_vs_bm_63d"],
            percentile_rank=row["percentile_rank"],
        )
        for _, row in res_df.iterrows()
    ]


# ------------------------------------------------------------------------------
# STAGE 3: VOLATILITY & EXPECTED MOVE
# ------------------------------------------------------------------------------

@mcp.tool(
    name="swing_get_expected_move",
    description="""
Calculates options implied volatility and projected dollar/percentage move range for a stock over a specified holding window.

USE WHEN: User asks for expected price targets, implied volatility range, option market price bounds, 21-day price range expectation, straddle pricing bounds, or target price limits for profit taking.

DOMAIN KEYWORDS: expected move, implied volatility, IV, options chain, ATM calls, volatility pricing, target range, upper bound, lower bound, target days, expected deviation, trade targets, price channel

INPUTS:
  - symbol (str): Target stock ticker symbol.
  - target_days (int, default 21): Trade holding window in calendar days (1 to 90).

OUTPUTS:
  - symbol (str): Ticker symbol evaluated
  - spot_price (float): Current underlying price in USD
  - target_days (int): Holding period in days
  - expiration_used (str): Options expiration date used
  - atm_implied_volatility (float): ATM IV percentage
  - expected_move_dollars (float): Expected move in USD
  - expected_move_pct (float): Expected move percentage
  - expected_upper_bound (float): Projected resistance/target price
  - expected_lower_bound (float): Projected support/stop price

EXAMPLE:
  swing_get_expected_move(symbol='NVDA', target_days=21)

RELATED TOOLS:
  Call BEFORE: swing_scan_universe, swing_rank_relative_strength
  Call AFTER: swing_calculate_position_size, swing_calculate_atr_stops
"""
)
@cache_result()
def swing_get_expected_move(
    symbol: str = Field(description="Equity ticker symbol. Example: 'AAPL'"),
    target_days: int = Field(default=21, description="Holding duration in calendar days. Default 21.")
) -> ExpectedMoveResult:
    t = yf.Ticker(symbol)
    hist = t.history(period="1d")
    if hist.empty:
        raise ValueError(f"No market data for ticker {symbol}")

    spot = float(hist["Close"].iloc[-1])
    expirations = t.expirations
    if not expirations:
        raise ValueError(f"No options chains available for {symbol}")

    now = pd.Timestamp.now()
    exp_dates = pd.to_datetime(expirations)
    days_diff = (exp_dates - now).days
    closest_idx = int(np.abs(days_diff - target_days).argmin())
    target_exp = expirations[closest_idx]
    actual_days = max(1, int(days_diff[closest_idx]))

    opt = t.option_chain(target_exp)
    calls = opt.calls

    atm_call = calls.iloc[(calls["strike"] - spot).abs().argmin()]
    iv = float(atm_call["impliedVolatility"])

    expected_move = spot * iv * np.sqrt(actual_days / 365.0)

    return ExpectedMoveResult(
        symbol=symbol,
        spot_price=round(spot, 2),
        target_days=actual_days,
        expiration_used=str(target_exp),
        atm_implied_volatility=round(iv * 100, 2),
        expected_move_dollars=round(expected_move, 2),
        expected_move_pct=round((expected_move / spot) * 100, 2),
        expected_upper_bound=round(spot + expected_move, 2),
        expected_lower_bound=round(spot - expected_move, 2),
    )


# ------------------------------------------------------------------------------
# STAGE 4: BACKTESTING
# ------------------------------------------------------------------------------

@mcp.tool(
    name="swing_backtest_signal",
    description="""
Backtests a technical entry condition (moving average crossover, RSI oversold rebound, or 20-day breakout) on historical daily data.

USE WHEN: User asks to backtest an entry rule, evaluate win rate, test historical expectancy, check MAE/MFE metrics, validate 20 EMA > 50 SMA crossover performance, test RSI < 30 bounce, or quantify historical setup validity.

DOMAIN KEYWORDS: backtest, signal test, win rate, trade expectancy, MAE, MFE, sma_crossover, rsi_oversold, breakout_20d, hit rate, historical performance, backtesting engine, quantitative validation, setup edge

INPUTS:
  - symbol (str): Target ticker symbol.
  - signal_type (str, default 'sma_crossover'): One of 'sma_crossover', 'rsi_oversold', 'breakout_20d'.
  - hold_days (int, default 21): Fixed holding window in calendar days.
  - lookback_years (int, default 2): Historical window evaluation in years (1 to 5).

OUTPUTS:
  - symbol (str): Ticker symbol evaluated
  - signal_type (str): Tested strategy type
  - hold_days (int): Holding period evaluated
  - total_trades (int): Total triggered setups
  - win_rate (float): Winning percentage decimal (0.0 - 1.0)
  - expectancy_pct (float): Expected return % per trade
  - avg_win_pct (float): Average win gain %
  - avg_loss_pct (float): Average loss %
  - avg_mae_pct (float): Maximum Adverse Excursion %
  - avg_mfe_pct (float): Maximum Favorable Excursion %

EXAMPLE:
  swing_backtest_signal(symbol='AAPL', signal_type='sma_crossover', hold_days=21, lookback_years=2)

RELATED TOOLS:
  Call BEFORE: swing_scan_universe, swing_rank_relative_strength
  Call AFTER: swing_calculate_position_size
"""
)
@cache_result()
def swing_backtest_signal(
    symbol: str = Field(description="Equity ticker. Example: 'MSFT'"),
    signal_type: str = Field(default="sma_crossover", description="Signal strategy: 'sma_crossover' | 'rsi_oversold' | 'breakout_20d'"),
    hold_days: int = Field(default=21, description="Holding duration in calendar days. Default 21"),
    lookback_years: int = Field(default=2, description="Years of historical daily bars. Default 2")
) -> BacktestSignalResult:
    df = yf.download(symbol, period=f"{lookback_years}y", interval="1d", progress=False)
    if df.empty:
        raise ValueError(f"No historical price data found for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    if signal_type == "sma_crossover":
        ema20 = close.ewm(span=20, adjust=False).mean()
        sma50 = close.rolling(50).mean()
        signal = (ema20 > sma50) & (ema20.shift(1) <= sma50.shift(1))
    elif signal_type == "rsi_oversold":
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        signal = (rsi < 30) & (rsi.shift(1) >= 30)
    elif signal_type == "breakout_20d":
        high_20 = high.rolling(20).max().shift(1)
        signal = close > high_20
    else:
        raise ValueError("Invalid signal_type. Choose: 'sma_crossover', 'rsi_oversold', 'breakout_20d'")

    entry_dates = df.index[signal].tolist()
    results = []

    for date in entry_dates:
        loc = df.index.get_loc(date)
        if loc + hold_days >= len(df):
            continue

        entry_price = close.iloc[loc]
        future_window = df.iloc[loc + 1 : loc + 1 + hold_days]

        exit_price = future_window["Close"].iloc[-1]
        max_high = future_window["High"].max()
        min_low = future_window["Low"].min()

        pnl_pct = (exit_price - entry_price) / entry_price
        mae_pct = (min_low - entry_price) / entry_price
        mfe_pct = (max_high - entry_price) / entry_price

        results.append({"pnl": pnl_pct, "mae": mae_pct, "mfe": mfe_pct, "win": pnl_pct > 0})

    if not results:
        return BacktestSignalResult(
            symbol=symbol,
            signal_type=signal_type,
            hold_days=hold_days,
            total_trades=0,
            win_rate=0.0,
            expectancy_pct=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            avg_mae_pct=0.0,
            avg_mfe_pct=0.0,
        )

    res_df = pd.DataFrame(results)
    win_rate = float(res_df["win"].mean())
    avg_win = float(res_df[res_df["win"]]["pnl"].mean()) if win_rate > 0 else 0.0
    avg_loss = float(res_df[~res_df["win"]]["pnl"].mean()) if win_rate < 1 else 0.0

    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    return BacktestSignalResult(
        symbol=symbol,
        signal_type=signal_type,
        hold_days=hold_days,
        total_trades=int(len(res_df)),
        win_rate=round(win_rate, 4),
        expectancy_pct=round(expectancy * 100, 2),
        avg_win_pct=round(avg_win * 100, 2),
        avg_loss_pct=round(avg_loss * 100, 2),
        avg_mae_pct=round(float(res_df["mae"].mean()) * 100, 2),
        avg_mfe_pct=round(float(res_df["mfe"].mean()) * 100, 2),
    )


# ------------------------------------------------------------------------------
# STAGE 5: RISK MANAGEMENT & POSITION SIZING
# ------------------------------------------------------------------------------

@mcp.tool(
    name="swing_calculate_position_size",
    description="""
Calculates exact share position sizing based on risk percentage, stop-loss distance, and account value while auditing cross-portfolio correlation.

USE WHEN: User asks how many shares to buy, position sizing calculation, dollar risk management, capital allocation check, stop-loss distance sizing, correlation risk audit, or sizing trades for a retirement/swing trading account.

DOMAIN KEYWORDS: position size, share count, account risk, risk percentage, stop loss, dollar risk, capital allocation, portfolio risk, correlation check, high correlation warning, risk management, trade sizing

INPUTS:
  - symbol (str): Ticker symbol to buy.
  - account_value (float): Total portfolio value in USD (e.g. 100000.0).
  - stop_price (float): Target hard stop loss price in USD.
  - risk_pct (float, default 1.0): Max portfolio risk percentage (e.g. 1.0 = 1%).
  - current_holdings (list[str], optional): Active portfolio stock tickers to evaluate correlation against.

OUTPUTS:
  - symbol (str): Ticker symbol
  - entry_price (float): Current price / entry target in USD
  - stop_price (float): Exit stop price in USD
  - risk_per_share (float): Risk per share in USD
  - account_value (float): Total portfolio account balance
  - allocated_shares (int): Integer count of shares to buy
  - position_value (float): Dollar value of total allocation
  - portfolio_allocation_pct (float): % of account consumed by trade
  - dollar_risk (float): Total dollar risk exposed
  - correlation_warnings (list[str]): High correlation (>0.70) alerts

EXAMPLE:
  swing_calculate_position_size(symbol='NVDA', account_value=100000.0, risk_pct=1.0, stop_price=118.50, current_holdings=['AAPL'])

RELATED TOOLS:
  Call BEFORE: swing_get_expected_move, swing_calculate_atr_stops
  Call AFTER: None - final trade parameter output
"""
)
@cache_result()
def swing_calculate_position_size(
    symbol: str = Field(description="Equity ticker symbol. Example: 'NVDA'"),
    account_value: float = Field(description="Total portfolio equity account value in USD. Example: 100000.0"),
    stop_price: float = Field(description="Hard stop-loss exit price in USD. Example: 115.00"),
    risk_pct: float = Field(default=1.0, description="Max portfolio account risk percentage. Default 1.0 (1%)"),
    current_holdings: list[str] = Field(default=[], description="List of active stock tickers held in portfolio. Example: ['AAPL']")
) -> PositionSizeResult:
    if account_value <= 0:
        raise ValueError("account_value must be positive.")
    if not (0 < risk_pct <= 100):
        raise ValueError("risk_pct must be between 0 and 100.")

    t = yf.Ticker(symbol)
    hist = t.history(period="3mo")
    if hist.empty:
        raise ValueError(f"No data for {symbol}")

    entry_price = float(hist["Close"].iloc[-1])
    risk_per_share = abs(entry_price - stop_price)

    if risk_per_share == 0:
        raise ValueError("Stop price cannot equal current entry price.")

    dollar_risk_target = account_value * (risk_pct / 100.0)
    shares = int(dollar_risk_target / risk_per_share)
    total_position_value = shares * entry_price

    correlation_warnings = []
    if current_holdings:
        all_tickers = list(set([symbol] + current_holdings))
        data = yf.download(all_tickers, period="3mo", interval="1d", progress=False)

        if isinstance(data.columns, pd.MultiIndex):
            data = data["Close"] if "Close" in data.columns.get_level_values(0) else data.xs("Close", level=1, axis=1)

        corr_matrix = data.pct_change().corr()

        for h in current_holdings:
            if h in corr_matrix.columns and h != symbol and symbol in corr_matrix.index:
                rho = corr_matrix.loc[symbol, h]
                if rho > 0.70:
                    correlation_warnings.append(f"HIGH CORRELATION: {symbol} vs {h} (r = {round(float(rho), 2)})")

    return PositionSizeResult(
        symbol=symbol,
        entry_price=round(entry_price, 2),
        stop_price=round(stop_price, 2),
        risk_per_share=round(risk_per_share, 2),
        account_value=round(account_value, 2),
        allocated_shares=shares,
        position_value=round(total_position_value, 2),
        portfolio_allocation_pct=round((total_position_value / account_value) * 100, 2),
        dollar_risk=round(shares * risk_per_share, 2),
        correlation_warnings=correlation_warnings,
    )


@mcp.tool(
    name="swing_calculate_atr_stops",
    description="""
Calculates 14-period Average True Range (ATR) volatility stop-loss levels, multi-tier R-multiple profit targets, and Chandelier exit levels.

USE WHEN: User asks for ATR stop loss, target prices, risk reward targets, 1R 2R 3R profit target prices, Chandelier trailing stop level, or setting stop loss based on volatility.

DOMAIN KEYWORDS: ATR stop loss, Average True Range, profit targets, R-multiple, 1R, 2R, 3R target, Chandelier exit, trailing stop, volatility stop, entry price, profit taking, trade management

INPUTS:
  - symbol (str): Target stock ticker.
  - entry_price (float): Planned execution price in USD.
  - atr_multiplier (float, default 2.0): Volatility stop distance multiplier.

OUTPUTS:
  - symbol (str): Stock ticker
  - entry_price (float): Execution entry price in USD
  - current_atr_14 (float): 14-period Average True Range value
  - suggested_stop_loss (float): ATR-based initial stop price
  - target_1r (float): 1:1 R/R profit target
  - target_2r (float): 2:1 R/R profit target
  - target_3r (float): 3:1 R/R profit target
  - trailing_stop_chandelier (float): Chandelier Exit trailing stop price

EXAMPLE:
  swing_calculate_atr_stops(symbol='AMD', entry_price=150.00, atr_multiplier=2.0)

RELATED TOOLS:
  Call BEFORE: swing_scan_universe
  Call AFTER: swing_calculate_position_size
"""
)
@cache_result()
def swing_calculate_atr_stops(
    symbol: str = Field(description="Equity ticker symbol. Example: 'AMD'"),
    entry_price: float = Field(description="Planned entry price in USD. Example: 150.00"),
    atr_multiplier: float = Field(default=2.0, description="ATR multiplier for initial stop loss. Default 2.0")
) -> ATRStopsResult:
    df = yf.download(symbol, period="3mo", interval="1d", progress=False)
    if df.empty:
        raise ValueError(f"No price data found for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr14 = float(tr.rolling(14).mean().iloc[-1])

    stop_loss = entry_price - (atr_multiplier * atr14)
    r_unit = entry_price - stop_loss

    target_1r = entry_price + r_unit
    target_2r = entry_price + (2.0 * r_unit)
    target_3r = entry_price + (3.0 * r_unit)

    highest_high_22 = float(df["High"].iloc[-22:].max())
    chandelier_stop = highest_high_22 - (3.0 * atr14)

    return ATRStopsResult(
        symbol=symbol,
        entry_price=round(entry_price, 2),
        current_atr_14=round(atr14, 2),
        suggested_stop_loss=round(stop_loss, 2),
        target_1r=round(target_1r, 2),
        target_2r=round(target_2r, 2),
        target_3r=round(target_3r, 2),
        trailing_stop_chandelier=round(chandelier_stop, 2),
    )


# ------------------------------------------------------------------------------
# APPLICATION STARTUP LOGGING & EXECUTION
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()

    tools_list = [
        "swing_get_market_regime",
        "swing_scan_universe",
        "swing_rank_relative_strength",
        "swing_get_expected_move",
        "swing_backtest_signal",
        "swing_calculate_position_size",
        "swing_calculate_atr_stops",
    ]

    logger.info("==================================================")
    logger.info("REGISTERING MCP TOOLS FOR SWING TRADING PIPELINE")
    logger.info("==================================================")
    for tool_name in tools_list:
        logger.info(f"INFO: Registered tool: {tool_name}")
    logger.info(f"INFO: Total tools registered: {len(tools_list)}")
    logger.info("==================================================")

    transport = os.getenv("FASTMCP_TRANSPORT", "streamable-http")
    host = os.getenv("FASTMCP_HOST", "0.0.0.0")
    port = int(os.getenv("FASTMCP_PORT", "8000"))

    mcp.run(transport=transport, host=host, port=port)
