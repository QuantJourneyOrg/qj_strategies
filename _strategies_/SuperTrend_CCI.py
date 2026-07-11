"""Causal Supertrend and CCI strategy."""

import asyncio

import pandas as pd

from backtester import Backtester

from .qj_strategy_common import (
    backtester_config,
    cci,
    equal_weights,
    stateful_signals,
    supertrend_direction,
)


class SupertrendCCIStrategy(Backtester):
    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("close")
        high = self.instruments_data.get_feature("high")
        low = self.instruments_data.get_feature("low")
        direction = supertrend_direction(high, low, close)
        oscillator = cci(high, low, close)
        entries = direction.gt(0.0) & direction.shift(1).le(0.0) & oscillator.ge(-100.0)
        exits = direction.lt(0.0) | oscillator.ge(200.0)
        return stateful_signals(entries, exits)

    def _compute_weights(self) -> pd.DataFrame:
        return equal_weights(self.signals)


async def run_backtest() -> None:
    strategy = SupertrendCCIStrategy(
        **backtester_config(
            strategy_name="Supertrend_CCI_Strategy",
            instruments=["V", "MA", "PYPL", "WMT", "COST", "TGT", "HD", "LOW"],
            start="2018-01-01",
        )
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(run_backtest())
