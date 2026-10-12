# Operational cost and capacity assumptions

The revised paper adds an operational sensitivity analysis. The Fraud Oracle
benchmark does not contain paid loss, recovered fraud, investigator time, or
claim-delay costs. Therefore, no monetary value in the paper is represented as
an observed insurer saving.

## Fixed-capacity evaluation

Claims are ranked by calibrated fraud probability. The highest-risk 5%, 10%,
20%, and 30% are selected for investigation. For every capacity level the
analysis reports:

- claims investigated;
- confirmed fraud cases captured;
- share of all fraud captured;
- investigation precision; and
- legitimate claims investigated.

## Illustrative cost scenarios

| Scenario | Investigation | Additional legitimate-claim burden | Missed fraud | Legitimate defer delay |
|---|---:|---:|---:|---:|
| Low-cost | £100 | £50 | £5,000 | £40 |
| Base | £250 | £150 | £10,000 | £100 |
| High-loss | £500 | £300 | £25,000 | £250 |

The calculation is:

```text
expected cost =
    investigations × investigation cost
  + legitimate investigations × legitimate-claim burden
  + (fraud approved + 0.25 × fraud deferred) × missed-fraud cost
  + legitimate deferrals × delay cost
```

The `0.25` multiplier assumes that manual review of deferred claims recovers
75% of the loss that would otherwise arise from missed fraud. This is a
scenario parameter, not an empirical estimate.

## Deployment requirement

An insurer must replace every assumption with governed internal estimates,
including investigator salary and overhead, average prevented loss, recovery
probability, payment-delay compensation, complaint and appeal costs, customer
attrition, and the cost of incorrectly delaying or scrutinising a legitimate
claim. Thresholds must then be selected under actual investigator capacity.
