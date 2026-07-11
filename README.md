# QuantJourney Strategies

Six causal, long/cash strategy examples built on the public
[`quantjourney-bt`](https://github.com/QuantJourneyOrg/quantjourney-bt) engine.
They use the engine's weights pipeline, cash-aware accounting, rebalance policy,
cost model, and delayed execution semantics. No private QuantJourney package is
required.

## Included strategies

- Bollinger Bands with stochastic confirmation
- Ichimoku cloud with RSI confirmation
- MACD with RSI confirmation
- EMA trend with ADX filter
- Supertrend with CCI confirmation
- Support/resistance breakout with inverse-volatility weights

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Set `QJ_API_KEY` for QuantJourney data, or use the deterministic data bundled
with the engine:

```bash
export QJ_SAMPLE_DATA=1
python -m _strategies_.MA_ADX
```

Each strategy computes indicators using current and historical observations
only. The backtester applies target weights on its configured execution lag, so
the examples do not trade on the bar that generated the signal.

## License

Licensed under Apache License 2.0.
