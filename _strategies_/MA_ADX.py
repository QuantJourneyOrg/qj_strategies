"""Causal EMA trend strategy filtered by ADX."""

import asyncio

import pandas as pd

from backtester import Backtester

from .qj_strategy_common import adx, backtester_config, equal_weights, stateful_signals


class MACrossADXStrategy(Backtester):
    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("close")
        high = self.instruments_data.get_feature("high")
        low = self.instruments_data.get_feature("low")
        fast = close.ewm(span=20, adjust=False).mean()
        slow = close.ewm(span=50, adjust=False).mean()
        trend_strength = adx(high, low, close)
        bullish = fast > slow
        entries = bullish & ~bullish.shift(1).fillna(False) & (trend_strength >= 25.0)
        exits = ~bullish
        return stateful_signals(entries, exits)

    def _compute_weights(self) -> pd.DataFrame:
        return equal_weights(self.signals)


async def run_backtest() -> None:
    strategy = MACrossADXStrategy(
        **backtester_config(
            strategy_name="MA_Cross_ADX_Strategy",
            instruments=["V", "MA", "PYPL", "WMT", "COST", "TGT", "HD", "LOW"],
            start="2018-01-01",
        )
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(run_backtest())
