"""Causal Bollinger Band and stochastic mean-reversion strategy."""

import asyncio

import pandas as pd

from backtester import Backtester

from .qj_strategy_common import backtester_config, equal_weights, stateful_signals


class BollingerStochasticStrategy(Backtester):
    def _compute_signals(self) -> pd.DataFrame:
        close = self.instruments_data.get_feature("close")
        high = self.instruments_data.get_feature("high")
        low = self.instruments_data.get_feature("low")
        middle = close.rolling(20).mean()
        lower_band = middle - 2.0 * close.rolling(20).std()
        lowest = low.rolling(14).min()
        highest = high.rolling(14).max()
        stochastic = 100.0 * (close - lowest).div((highest - lowest).replace(0.0, float("nan")))
        stochastic = stochastic.rolling(3).mean()

        entries = (
            (close.shift(1) >= lower_band.shift(1))
            & (close < lower_band)
            & (stochastic.shift(1) <= 20.0)
            & (stochastic > 20.0)
        )
        exits = (close >= middle) | (stochastic >= 80.0)
        return stateful_signals(entries, exits)

    def _compute_weights(self) -> pd.DataFrame:
        return equal_weights(self.signals)


async def run_backtest() -> None:
    strategy = BollingerStochasticStrategy(
        **backtester_config(
            strategy_name="Bollinger_Stochastic_Strategy",
            instruments=["AAPL", "NVDA", "MSFT", "GOOGL", "META", "AMZN"],
            start="2015-01-01",
        )
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(run_backtest())
