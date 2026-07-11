"""Causal Ichimoku cloud and RSI trend strategy."""

import asyncio

import pandas as pd

from backtester import Backtester

from .qj_strategy_common import backtester_config, equal_weights, rsi, stateful_signals


class IchimokuRSIStrategy(Backtester):
    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("close")
        high = self.instruments_data.get_feature("high")
        low = self.instruments_data.get_feature("low")
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2.0
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2.0
        span_a = ((tenkan + kijun) / 2.0).shift(26)
        span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2.0).shift(26)
        momentum = rsi(close)
        above_cloud = (close > span_a) & (close > span_b)
        entries = above_cloud & ~above_cloud.shift(1).fillna(False) & (momentum >= 40.0)
        exits = (close < kijun) | (momentum >= 70.0)
        return stateful_signals(entries, exits)

    def _compute_weights(self) -> pd.DataFrame:
        return equal_weights(self.signals)


async def run_backtest() -> None:
    strategy = IchimokuRSIStrategy(
        **backtester_config(
            strategy_name="Ichimoku_RSI_Strategy",
            instruments=["V", "MA", "PYPL", "WMT", "COST", "TGT", "HD", "LOW"],
            start="2018-01-01",
        )
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(run_backtest())
