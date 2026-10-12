# Exact prediction and decision calculation

## 1. Structured-data probability

The original fraud model receives 28 structured claim fields. `PolicyNumber`,
`Sex`, the exact numeric `Age` field, and `MaritalStatus` do not enter the
score. `AgeOfPolicyHolder` is a separate categorical age-band field and remains
in the original specification. It must not be described as removed. A separate
sensitivity specification can also exclude `AgeOfPolicyHolder`.

Numeric inputs are median-imputed and standardised. Categorical inputs are mode-imputed and one-hot encoded. A histogram gradient-boosting classifier produces a raw decision score for each of five training-fold models.

Each fold score, `z_j`, is calibrated using Platt's sigmoid:

```text
p_j = 1 / (1 + exp(a_j * z_j + b_j))
```

The saved sklearn implementation evaluates the equivalent `expit(-(a_j*z_j+b_j))`; its learned `a_j` values are negative. The final calibrated fraud probability is:

```text
p = (p_1 + p_2 + p_3 + p_4 + p_5) / 5
```

## 2. Validation-selected boundaries

The investigation threshold maximises validation F2:

```text
F2 = 5 * precision * recall / (4 * precision + recall)
```

F2 weights recall more heavily than precision. The approval threshold is the largest lower threshold where approved validation claims retain at least 99% negative predictive value:

```text
NPV = true negatives / (true negatives + false negatives)
```

For model `fairclaim-20260720T214443Z`:

```text
if p < 0.0469609584:
    tabular_decision = "approve"
elif p >= 0.0929560868:
    tabular_decision = "investigate"
else:
    tabular_decision = "defer"
```

The boundary is selected on validation data and evaluated once on the untouched test data.

## 3. Evidence-fusion workflow

The damage-image classifier separately predicts one of six classes: front/rear × normal/breakage/crushed. Notes are scanned for explicit front/rear and damage terms.

- A damage label by itself does not imply fraud.
- A high-confidence image/note contradiction moves `approve` to `defer`.
- A high-confidence contradiction moves `defer` or `investigate` to `investigate`.
- Consistent or insufficient evidence never reduces the tabular referral route.
- Audio files are stored locally; only a supplied/verified transcript participates in the current consistency check.

This rule changes the workflow route, not the calibrated fraud probability. It is deliberately conservative because the available image, narrative, and tabular datasets do not share verified row-level fraud labels.
