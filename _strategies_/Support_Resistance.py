"""Causal support and resistance breakout strategy."""

import asyncio

import numpy as np
import pandas as pd

from backtester import Backtester

from .qj_strategy_common import backtester_config


class SupportResistanceBreakout(Backtester):
    LOOKBACK = 20

    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("close")
        high = self.instruments_data.get_feature("high")
        low = self.instruments_data.get_feature("low")
        volume = self.instruments_data.get_feature("volume")
        resistance = high.rolling(self.LOOKBACK).max().shift(1)
        support = low.rolling(self.LOOKBACK).min().shift(1)
        volume_threshold = volume.rolling(self.LOOKBACK).median().shift(1) * 1.5
        entries = (
            (close.shift(1) <= resistance.shift(1))
            & (close > resistance)
            & (volume > volume_threshold)
        )
        exits = close < support

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        for instrument in close.columns:
            holding = False
            age = 0
            for date in close.index:
                if holding and (bool(exits.loc[date, instrument]) or age >= 20):
                    holding = False
                    age = 0
                if not holding and bool(entries.loc[date, instrument]):
                    holding = True
                    age = 0
                signals.loc[date, instrument] = float(holding)
                age = age + 1 if holding else 0
        return signals

    def _compute_weights(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("close")
        volatility = close.pct_change().rolling(20).std()
        inverse_volatility = 1.0 / volatility.replace(0.0, np.nan)
        raw = self.signals.gt(0.0) * inverse_volatility
        normalized = raw.div(raw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        return normalized.clip(upper=0.20)


async def run_backtest() -> None:
    strategy = SupportResistanceBreakout(
        **backtester_config(
            strategy_name="Support_Resistance_Breakout",
            instruments=["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN"],
            start="2020-01-01",
            max_position_size=0.20,
        )
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(run_backtest())
