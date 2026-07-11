"""Causal MACD and RSI confirmation strategy."""

import asyncio

import pandas as pd

from backtester import Backtester

from .qj_strategy_common import backtester_config, equal_weights, rsi, stateful_signals


class MACDRSIConfirmation(Backtester):
    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("close")
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        signal_line = macd.ewm(span=9, adjust=False).mean()
        momentum = rsi(close)
        bullish = macd > signal_line
        entries = bullish & ~bullish.shift(1).fillna(False) & (momentum >= 40.0)
        exits = (~bullish) | (momentum >= 70.0)
        return stateful_signals(entries, exits)

    def _compute_weights(self) -> pd.DataFrame:
        return equal_weights(self.signals)


async def run_backtest() -> None:
    strategy = MACDRSIConfirmation(
        **backtester_config(
            strategy_name="MACD_RSI_Confirmation",
            instruments=["V", "MA", "PYPL", "WMT", "COST", "TGT", "HD", "LOW"],
            start="2018-01-01",
        )
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(run_backtest())
