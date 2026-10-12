# FairClaim-MM model card

## Intended use

Research and controlled decision-support experiments for motor-insurance fraud referral. The model may prioritise claims for review. It must not autonomously deny a claim, accuse a claimant of fraud, or replace qualified investigators.

## Training data

Fraud Oracle: 15,420 historical motor-insurance records, 923 labelled fraud cases (5.99%). The data is an old benchmark and is not representative of a current insurer's population without external validation.

## Model selection

Six model families are compared using five-fold training cross-validation and a separate validation set. The primary selection criterion is mean five-fold training CV PR-AUC; validation PR-AUC is a secondary check. The selected classifier is calibrated using training folds. Thresholds are selected using validation predictions, and final metrics are generated once on the untouched test set.

## Protected attributes

`Sex`, `Age`, and `MaritalStatus` do not enter the production score. They are retained for post-prediction fairness measurement. Exclusion does not guarantee fairness because non-protected features can act as proxies and subgroup fraud prevalence can differ.

## Class imbalance

The training partition contains 554 fraud and 8,698 non-fraud records. Five strategies were compared inside training folds: no balancing, balanced class weights, random undersampling, random oversampling, and SMOTENC. No balancing achieved the strongest validation PR-AUC in the recorded imbalance comparison. The full comparison is stored in `reports/imbalance_strategy_comparison.csv`.

## Decision policy

- Below the low-risk threshold: `approve`
- Between low- and high-risk thresholds: `abstain`
- At or above the high-risk threshold: `investigate`

Thresholds are model-versioned and stored in `reports/decision_thresholds.json`. An investigation referral is not a fraud determination.

## Explanations

Global explanation uses held-out permutation importance. Local explanation replaces each input, one at a time, with a typical training value and reports the change in score. These are model-sensitivity explanations, not causal statements.

## Known limitations

- The primary data is small, dated, imbalanced, and not externally validated.
- The external `insurance_claims.csv` uses a different schema and cannot be merged as if it were the same cohort.
- Damage images do not carry fraud labels.
- Ecommerce evidence is synthetic.
- Fairness estimates for small and intersectional groups are unstable.
- Uploaded evidence is stored for review but is not scored by the current tabular model.
- The separate six-class image-damage baseline has only 55.4% held-out accuracy. It is suitable for prototype evidence display and high-confidence contradiction checks, not autonomous claim adjudication.
