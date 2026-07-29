"""Build and execute the supplementary notebook for the AOA revision."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "publication" / "aoa" / "notebooks"
OUT = OUT_DIR / "01_operational_selective_triage_analysis.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source)


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.cells = [
        markdown(
            """# Calibrated selective prediction for motor-insurance fraud triage

This executed notebook is the supplementary analytical record for the revised
Applied Operations and Analytics submission. The computationally expensive
training and bootstrap workflow is implemented in
`scripts/run_aoa_revision_analyses.py`; this notebook reads its machine-readable
outputs, reproduces the principal tables and figures, and records the equations
and assumptions used in the manuscript.

The image experiment is deliberately outside this paper because its observations
are not linked to verified fraud outcomes."""
        ),
        code(
            """from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

ROOT = Path.cwd().parents[2]
TABLES = ROOT / "publication" / "aoa" / "tables"
FIGURES = ROOT / "publication" / "aoa" / "figures"
REPORTS = ROOT / "reports" / "aoa_revision"
summary = json.loads((REPORTS / "analysis_summary.json").read_text())
summary"""
        ),
        markdown(
            r"""## 1. Methodologically consistent model comparison

The first experiment compares model families with **no class weighting or
resampling for any family**. A second experiment holds the model family fixed
and varies the imbalance strategy. Resampling, when used, occurs only within
training folds.

The primary selection statistic is validation PR-AUC because fraud prevalence
is approximately 6%."""
        ),
        code(
            """model_family = pd.read_csv(TABLES / "table_01_model_family_no_balancing.csv")
imbalance = pd.read_csv(TABLES / "table_02_hgb_imbalance_strategies.csv")
display(model_family)
display(imbalance)
display(Image(filename=str(FIGURES / "figure_01_consistent_model_comparison.png")))"""
        ),
        markdown(
            r"""## 2. Calibration and three-way decision

For fold \(k\), raw score \(z_k(x)\) is calibrated as

\[
p_k(x)=\frac{1}{1+\exp(a_k z_k(x)+b_k)},\qquad
\hat p(x)=\frac{1}{5}\sum_{k=1}^{5}p_k(x).
\]

The investigation threshold maximises validation \(F_2\). The approval
threshold is the largest lower threshold with validation NPV of at least 99%.
The thresholds are frozen before test evaluation."""
        ),
        code(
            """calibration = pd.read_csv(TABLES / "table_04_calibration_comparison.csv")
regions = pd.read_csv(TABLES / "table_05_decision_regions_with_ci.csv")
ablations = pd.read_csv(TABLES / "table_11_framework_ablations.csv")
display(calibration)
display(regions)
display(ablations)
display(Image(filename=str(FIGURES / "figure_04_coverage_risk.png")))"""
        ),
        markdown(
            """## 3. Age-proxy sensitivity and three-action fairness

The original model excludes the exact `Age` field but retains the categorical
`AgeOfPolicyHolder` feature. It is therefore inaccurate to claim that all
age-related information was removed. The sensitivity experiment removes that
field and compares discrimination, calibration, approval NPV and the
between-age-band investigation-rate gap.

Fairness is evaluated across approval, abstention and investigation rather than
only at the binary investigation boundary."""
        ),
        code(
            """age_proxy = pd.read_csv(TABLES / "table_03_age_proxy_sensitivity.csv")
action_fairness = pd.read_csv(TABLES / "table_08_three_action_fairness.csv")
display(age_proxy)
display(action_fairness)
display(Image(filename=str(FIGURES / "figure_02_age_proxy_sensitivity.png")))
display(Image(filename=str(FIGURES / "figure_05_three_action_fairness.png")))"""
        ),
        markdown(
            r"""## 3.1 Explainability audit

Global permutation importance is the held-out PR-AUC decrease after shuffling
feature \(j\):

\[
I_j=\operatorname{PR\!-\!AUC}(f,X,y)-
\operatorname{PR\!-\!AUC}(f,X_{\pi(j)},y).
\]

This is predictive sensitivity, not a causal explanation. The dashboard also
provides reference-replacement probability deltas for individual claims."""
        ),
        code(
            """importance = pd.read_csv(TABLES / "table_15_permutation_importance.csv")
display(importance.head(15))
display(Image(filename=str(FIGURES / "figure_07_permutation_importance.png")))"""
        ),
        markdown(
            r"""## 4. Operational value and investigation capacity

At capacity \(q\), the \(q n\) highest-scored claims are investigated. Fraud
capture and investigation precision are reported at 5%, 10%, 20% and 30%.

Illustrative expected cost is

\[
C = n_Ic_I + FP_Ic_L +
\left(FN_A + 0.25FN_U\right)c_F + TN_Uc_D,
\]

where \(I,A,U\) denote investigate, approve and abstain; \(c_I\) is investigation
cost, \(c_L\) is legitimate-claim investigation burden, \(c_F\) is missed-fraud
loss and \(c_D\) is legitimate-payment delay. These are sensitivity assumptions,
not observed claim amounts."""
        ),
        code(
            """capacity = pd.read_csv(TABLES / "table_09_fixed_capacity.csv")
costs = pd.read_csv(TABLES / "table_10_operational_cost_sensitivity.csv")
display(capacity)
display(costs)
display(Image(filename=str(FIGURES / "figure_03_operational_value.png")))"""
        ),
        markdown(
            """## 5. Robustness and uncertainty

The analysis reports bootstrap confidence intervals, threshold stability,
five repeated random seeds, sigmoid-versus-isotonic calibration, and a temporal
experiment trained on 1994, tuned on 1995 and tested on 1996."""
        ),
        code(
            """metric_ci = pd.read_csv(TABLES / "table_06_metric_bootstrap_ci.csv")
thresholds = pd.read_csv(TABLES / "table_07_threshold_bootstrap_samples.csv")
seeds = pd.read_csv(TABLES / "table_13_repeated_seeds.csv")
temporal = pd.read_csv(TABLES / "table_14_temporal_validation.csv")
display(metric_ci)
display(thresholds.describe(percentiles=[.025, .5, .975]))
display(seeds)
display(temporal)
display(Image(filename=str(FIGURES / "figure_06_uncertainty_robustness.png")))"""
        ),
        markdown(
            """## Reproduction

From the repository root:

```powershell
python -m pip install -r requirements.txt
python scripts\\run_aoa_revision_analyses.py
python scripts\\build_aoa_revision_notebook.py
```

The public benchmark is referenced by DOI in the manuscript. No test-set result
is used to select a model, calibration method or threshold."""
        ),
    ]
    nbf.write(nb, OUT)
    executed = NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(OUT_DIR)}},
    ).execute()
    nbf.write(executed, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
