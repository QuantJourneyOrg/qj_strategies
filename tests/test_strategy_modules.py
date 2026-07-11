import importlib

import pandas as pd

from _strategies_.qj_strategy_common import stateful_signals


MODULES = (
    "BBands_stoch",
    "Ichimoku_rsi",
    "MACD_RSI_confirmation",
    "MA_ADX",
    "SuperTrend_CCI",
    "Support_Resistance",
)


def test_all_strategy_modules_import():
    for module in MODULES:
        importlib.import_module(f"_strategies_.{module}")


def test_stateful_signals_enters_and_exits_on_current_observation():
    index = pd.RangeIndex(5)
    entries = pd.DataFrame({"A": [False, True, False, False, False]}, index=index)
    exits = pd.DataFrame({"A": [False, False, False, True, False]}, index=index)

    result = stateful_signals(entries, exits)

    assert result["A"].tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]
