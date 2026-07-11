"""Shared, causal indicator helpers for the public QuantJourney strategies."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from backtester.portfolio.rebalance import RebalancePolicy


def backtester_config(
    *,
    strategy_name: str,
    instruments: list[str],
    start: str,
    max_position_size: float = 0.25,
) -> dict:
    sample_mode = os.environ.get("QJ_SAMPLE_DATA", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    api_key = os.environ.get("QJ_API_KEY")
    return {
        "api_key": None if sample_mode else api_key,
        "email": None if sample_mode or api_key else os.environ.get("QJ_EMAIL"),
        "password": None if sample_mode or api_key else os.environ.get("QJ_PASSWORD"),
        "strategy_name": strategy_name,
        "strategy_type": "Long / Cash",
        "initial_capital": 100_000,
        "instruments": instruments,
        "backtest_period": {"start": start, "end": "2025-01-01"},
        "source": "sample" if sample_mode else "yfinance",
        "execution_mode": "weights",
        "max_position_size": max_position_size,
        "rebalance_policy": RebalancePolicy(frequency="D"),
        "indicators_config": [],
        "benchmark_symbol": "^GSPC",
        "benchmark_name": "S&P 500 Index",
        "show_text_reports": True,
        "save_text_reports": True,
        "save_portfolio_plots": True,
        "show_portfolio_plots": False,
    }


def equal_weights(signals: pd.DataFrame, cap: float = 0.25) -> pd.DataFrame:
    active = signals.gt(0.0)
    counts = active.sum(axis=1).replace(0, np.nan)
    return active.div(counts, axis=0).fillna(0.0).clip(upper=cap)


def stateful_signals(entries: pd.DataFrame, exits: pd.DataFrame) -> pd.DataFrame:
    signals = pd.DataFrame(0.0, index=entries.index, columns=entries.columns)
    holding = pd.Series(False, index=entries.columns)
    for date in entries.index:
        holding = holding.mask(exits.loc[date].fillna(False), False)
        holding = holding.mask(entries.loc[date].fillna(False), True)
        signals.loc[date] = holding.astype(float)
    return signals


def rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    relative_strength = gain.div(loss.replace(0.0, np.nan))
    return (100.0 - 100.0 / (1.0 + relative_strength)).fillna(50.0)


def true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    previous_close = close.shift(1)
    return pd.DataFrame(
        np.maximum.reduce(
            [
                (high - low).to_numpy(),
                (high - previous_close).abs().to_numpy(),
                (low - previous_close).abs().to_numpy(),
            ]
        ),
        index=close.index,
        columns=close.columns,
    )


def adx(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)
    average_range = (
        true_range(high, low, close).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    )
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / period, adjust=False).mean().div(average_range)
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / period, adjust=False).mean().div(average_range)
    denominator = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs().div(denominator)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def cci(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    period: int = 20,
) -> pd.DataFrame:
    typical = (high + low + close) / 3.0
    mean = typical.rolling(period).mean()
    deviation = typical.rolling(period).apply(
        lambda values: np.mean(np.abs(values - values.mean())), raw=True
    )
    return (typical - mean).div(0.015 * deviation.replace(0.0, np.nan))


def supertrend_direction(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    average_range = (
        true_range(high, low, close).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    )
    midpoint = (high + low) / 2.0
    basic_upper = midpoint + multiplier * average_range
    basic_lower = midpoint - multiplier * average_range
    directions = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    for instrument in close.columns:
        upper = basic_upper[instrument].copy()
        lower = basic_lower[instrument].copy()
        trend = pd.Series(np.nan, index=close.index)
        for position in range(1, len(close)):
            if pd.isna(upper.iloc[position]) or pd.isna(lower.iloc[position]):
                continue
            previous = position - 1
            if not pd.isna(upper.iloc[previous]):
                if (
                    upper.iloc[position] >= upper.iloc[previous]
                    and close[instrument].iloc[previous] <= upper.iloc[previous]
                ):
                    upper.iloc[position] = upper.iloc[previous]
                if (
                    lower.iloc[position] <= lower.iloc[previous]
                    and close[instrument].iloc[previous] >= lower.iloc[previous]
                ):
                    lower.iloc[position] = lower.iloc[previous]
            previous_trend = trend.iloc[previous]
            if pd.isna(previous_trend):
                trend.iloc[position] = lower.iloc[position]
            elif previous_trend == upper.iloc[previous]:
                trend.iloc[position] = (
                    upper.iloc[position]
                    if close[instrument].iloc[position] <= upper.iloc[position]
                    else lower.iloc[position]
                )
            else:
                trend.iloc[position] = (
                    lower.iloc[position]
                    if close[instrument].iloc[position] >= lower.iloc[position]
                    else upper.iloc[position]
                )
            directions.loc[close.index[position], instrument] = (
                1.0 if close[instrument].iloc[position] >= trend.iloc[position] else -1.0
            )
    return directions


def instruments(*symbols: str) -> list[str]:
    return list(symbols)
