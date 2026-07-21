"""Build and execute the publication notebooks and their figures/tables."""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "publication"
NOTEBOOKS = PUB / "notebooks"
FIGURES = PUB / "figures"
TABLES = PUB / "tables"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


SETUP = r"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, roc_curve, auc,
                             confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path.cwd()
PUB = ROOT / "publication"
FIGURES = PUB / "figures"
TABLES = PUB / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, GOLD, RED, GREEN, GREY = "#1f4e79", "#d89b2b", "#b03a2e", "#2e7d32", "#6b7280"
"""


def build_notebook(title: str, purpose: str, cells: list):
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    nb["cells"] = [
        md(f"# {title}\n\n{purpose}\n\n**Reproducibility rule:** outputs below are generated from the untouched held-out test set or from validation/CV reports; no test result is used to fit the model or choose thresholds."),
        code(SETUP),
        *cells,
    ]
    return nb


def save_execute(name: str, notebook):
    path = NOTEBOOKS / name
    nbf.write(notebook, path)
    client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbf.write(notebook, path)
    return path


def main():
    for directory in (NOTEBOOKS, FIGURES, TABLES):
        directory.mkdir(parents=True, exist_ok=True)

    nb1 = build_notebook(
        "01 — Data audit and exploratory analysis",
        "Audits the primary Fraud Oracle benchmark, quantifies imbalance, checks missingness and identifier leakage, and describes the auxiliary evidence datasets without joining unrelated records.",
        [
            code(r"""
df = pd.read_csv(ROOT / "fraud_oracle.csv")
target = "FraudFound_P"
summary = pd.DataFrame({
    "measure": ["rows", "columns", "fraud cases", "non-fraud cases", "fraud prevalence", "duplicate rows", "missing cells", "unique policy numbers"],
    "value": [len(df), df.shape[1], int(df[target].sum()), int((1-df[target]).sum()), float(df[target].mean()), int(df.duplicated().sum()), int(df.isna().sum().sum()), int(df["PolicyNumber"].nunique())],
})
summary
"""),
            code(r"""
summary.to_csv(TABLES / "table_01_dataset_audit.csv", index=False)
(TABLES / "table_01_dataset_audit.tex").write_text(summary.to_latex(index=False, escape=True, float_format="%.4f"), encoding="utf-8")

counts = df[target].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6.5, 4.1))
bars = ax.bar(["Non-fraud", "Fraud"], counts.values, color=[BLUE, RED])
ax.set_ylabel("Claims")
ax.set_title("Primary benchmark class distribution")
for b, v in zip(bars, counts.values): ax.text(b.get_x()+b.get_width()/2, v, f"{v:,}\n({v/len(df):.1%})", ha="center", va="bottom")
fig.tight_layout(); fig.savefig(FIGURES / "figure_01_class_distribution.png", bbox_inches="tight"); plt.show()
"""),
            code(r"""
aux = pd.DataFrame([
    ["Fraud Oracle", 15420, "structured claims", "FraudFound_P", "Primary fraud benchmark"],
    ["Comprehensive Car Damage", 2300, "vehicle images", "six damage/view classes", "Damage recognition only"],
    ["AI-generated E-commerce Images", 6031, "synthetic images + annotations", "12 issue classes", "Evidence prototype only"],
    ["INSURBIAS", 1388, "images + synthetic narratives + metadata", "condition/severity; no verified fraud target", "Counterfactual fairness/evidence audit only"],
])
aux.columns = ["dataset", "records", "modalities", "available labels", "permitted role"]
aux.to_csv(TABLES / "table_02_dataset_roles.csv", index=False)
(TABLES / "table_02_dataset_roles.tex").write_text(aux.to_latex(index=False, escape=True), encoding="utf-8")
aux
"""),
        ],
    )

    nb2 = build_notebook(
        "02 — Model and imbalance-strategy comparison",
        "Answers RQ1 using five-fold training-set cross-validation and a separate validation split. It compares six classifiers and five imbalance strategies; resampling was confined to training folds.",
        [
            code(r"""
models = pd.read_csv(ROOT / "reports" / "model_comparison.csv").sort_values("validation_pr_auc", ascending=False)
imb = pd.read_csv(ROOT / "reports" / "imbalance_strategy_comparison.csv").sort_values("validation_pr_auc", ascending=False)
models
"""),
            code(r"""
models.to_csv(TABLES / "table_03_model_comparison.csv", index=False)
imb.to_csv(TABLES / "table_04_imbalance_comparison.csv", index=False)
(TABLES / "table_03_model_comparison.tex").write_text(models.to_latex(index=False, escape=True, float_format="%.4f"), encoding="utf-8")
(TABLES / "table_04_imbalance_comparison.tex").write_text(imb.to_latex(index=False, escape=True, float_format="%.4f"), encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
axes[0].barh(models["model"], models["validation_pr_auc"], color=BLUE)
axes[0].invert_yaxis(); axes[0].set_xlabel("Validation PR-AUC"); axes[0].set_title("Model-family comparison")
axes[1].barh(imb["strategy"], imb["validation_pr_auc"], color=GOLD)
axes[1].invert_yaxis(); axes[1].set_xlabel("Validation PR-AUC"); axes[1].set_title("Imbalance-strategy comparison")
for ax in axes: ax.grid(axis="x", alpha=.2)
fig.tight_layout(); fig.savefig(FIGURES / "figure_02_model_imbalance_comparison.png", bbox_inches="tight"); plt.show()
"""),
            md("The selected HistGradientBoosting model without resampling achieved the highest cross-validated and validation PR-AUC. This directly answers why the production model was not artificially balanced: balancing was tested fairly, inside training folds, and did not improve the primary validation criterion."),
        ],
    )

    nb3 = build_notebook(
        "03 — Calibration and three-way decision policy",
        "Answers RQ2 by reconstructing held-out discrimination, calibration and the approve–abstain–investigate policy from saved predictions and validation-derived thresholds.",
        [
            code(r"""
pred = pd.read_csv(ROOT / "reports" / "held_out_test_predictions.csv")
cal = pd.read_csv(ROOT / "reports" / "calibration_curve.csv")
thresholds = json.loads((ROOT / "reports" / "decision_thresholds.json").read_text())
metrics = json.loads((ROOT / "reports" / "test_metrics.json").read_text())
thresholds, metrics
"""),
            code(r"""
y, p = pred["actual_fraud"].to_numpy(), pred["fraud_probability"].to_numpy()
precision, recall, _ = precision_recall_curve(y, p)
fpr, tpr, _ = roc_curve(y, p)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
axes[0].plot(recall, precision, color=BLUE, lw=2); axes[0].axhline(y.mean(), ls="--", color=GREY); axes[0].set(xlabel="Recall", ylabel="Precision", title=f"Precision–recall (AUC={metrics['pr_auc']:.3f})")
axes[1].plot(fpr, tpr, color=BLUE, lw=2); axes[1].plot([0,1],[0,1], ls="--", color=GREY); axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"ROC (AUC={metrics['roc_auc']:.3f})")
axes[2].plot(cal["mean_predicted_probability"], cal["observed_fraud_rate"], marker="o", color=BLUE); axes[2].plot([0,.25],[0,.25], ls="--", color=GREY); axes[2].set(xlabel="Mean predicted probability", ylabel="Observed fraud rate", title=f"Reliability (Brier={metrics['brier_score']:.3f})")
fig.tight_layout(); fig.savefig(FIGURES / "figure_03_discrimination_calibration.png", bbox_inches="tight"); plt.show()
"""),
            code(r"""
decision = pd.read_csv(ROOT / "reports" / "decision_policy_test_summary.csv")
decision.to_csv(TABLES / "table_05_decision_policy.csv", index=False)
(TABLES / "table_05_decision_policy.tex").write_text(decision.to_latex(index=False, escape=True, float_format="%.4f"), encoding="utf-8")

order = ["approve", "abstain", "investigate"]
d = decision.set_index("decision").loc[order]
fig, ax1 = plt.subplots(figsize=(7.2, 4.3))
x = np.arange(len(order)); bars = ax1.bar(x, d["claims"], color=[GREEN, GOLD, RED], alpha=.86)
ax1.set_xticks(x, [s.title() for s in order]); ax1.set_ylabel("Claims"); ax1.set_title("Held-out three-way triage outcomes")
ax2 = ax1.twinx(); ax2.plot(x, d["observed_fraud_rate"], color="#111827", marker="o", lw=2); ax2.set_ylabel("Observed fraud rate")
for b, n in zip(bars, d["claims"]): ax1.text(b.get_x()+b.get_width()/2, n, str(n), ha="center", va="bottom")
fig.tight_layout(); fig.savefig(FIGURES / "figure_04_decision_policy.png", bbox_inches="tight"); plt.show()
"""),
            md(r"""
### Decision function used by the prototype

For fold (k), a raw model score (z_k(x)) is sigmoid-calibrated as

\[
p_k(x)=\frac{1}{1+\exp(a_k z_k(x)+b_k)},\qquad
\hat p(x)=\frac{1}{K}\sum_{k=1}^{K}p_k(x),\quad K=5.
\]

The operational action is

\[
d(x)=\begin{cases}
\mathrm{approve},&\hat p(x)<\tau_A,\\
\mathrm{abstain},&\tau_A\leq\hat p(x)<\tau_I,\\
\mathrm{investigate},&\hat p(x)\geq\tau_I,
\end{cases}
\]

where \(\tau_A=0.0469609584\) is the largest validation threshold satisfying \(\mathrm{NPV}\geq0.99\), and \(\tau_I=0.0929560868\) maximizes validation \(F_2=5PR/(4P+R)\). No threshold was selected on the test set.
"""),
        ],
    )

    nb4 = build_notebook(
        "04 — Fairness and explainable-AI audit",
        "Answers RQ3 with subgroup outcome/performance metrics, bootstrap disparity intervals, global permutation importance and a clear statement that association is not causation.",
        [
            code(r"""
groups = pd.read_csv(ROOT / "reports" / "fairness_by_group.csv")
gaps = pd.read_csv(ROOT / "reports" / "fairness_gaps.csv")
ci = pd.read_csv(ROOT / "reports" / "fairness_gap_confidence_intervals.csv")
importance = pd.read_csv(ROOT / "reports" / "permutation_importance.csv")
gaps
"""),
            code(r"""
gaps.to_csv(TABLES / "table_06_fairness_gaps.csv", index=False)
ci.to_csv(TABLES / "table_07_fairness_bootstrap_ci.csv", index=False)
importance.head(15).to_csv(TABLES / "table_08_feature_importance.csv", index=False)
(TABLES / "table_06_fairness_gaps.tex").write_text(gaps.to_latex(index=False, escape=True, float_format="%.4f"), encoding="utf-8")

plot_gap = gaps.set_index("attribute")[["demographic_parity_difference", "equal_opportunity_difference", "false_positive_rate_difference"]]
ax = plot_gap.plot(kind="bar", figsize=(8.4, 4.6), color=[BLUE, GOLD, RED])
ax.set_ylabel("Max–min subgroup gap"); ax.set_title("Held-out fairness gaps"); ax.tick_params(axis="x", rotation=20); ax.legend(["DP difference", "EO difference", "FPR difference"], frameon=False)
ax.figure.tight_layout(); ax.figure.savefig(FIGURES / "figure_05_fairness_gaps.png", bbox_inches="tight"); plt.show()
"""),
            code(r"""
top = importance.head(15).sort_values("importance_mean")
fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color=BLUE, alpha=.88, capsize=2)
ax.set_xlabel("Decrease in PR-AUC after permutation"); ax.set_title("Global permutation importance on held-out data")
fig.tight_layout(); fig.savefig(FIGURES / "figure_06_permutation_importance.png", bbox_inches="tight"); plt.show()
"""),
            md("Sex, Age and MaritalStatus were excluded from model inputs but retained for post-hoc audit. The remaining disparities therefore show that removing protected fields alone does not guarantee parity: non-protected variables can encode population structure or proxy effects. Intersectional intervals are wide because several fraud-positive subgroups are small."),
        ],
    )

    nb5 = build_notebook(
        "05 — Multimodal evidence feasibility and architecture",
        "Documents the image branch, evaluates its held-out damage-class performance, and defines a conservative fusion rule. The branch does not estimate fraud because no downloaded image dataset has verified claim-level fraud labels linked to the tabular benchmark.",
        [
            code(r"""
im = json.loads((ROOT / "reports" / "image_damage_metrics.json").read_text())
cm = pd.read_csv(ROOT / "reports" / "image_damage_confusion_matrix.csv", index_col=0)
class_rows = [{"class": k, **{m: v for m, v in val.items() if m in {"precision", "recall", "f1-score", "support"}}}
              for k, val in im["classification_report"].items() if isinstance(val, dict) and k not in {"macro avg", "weighted avg"}]
image_table = pd.DataFrame(class_rows)
image_table.to_csv(TABLES / "table_09_image_model.csv", index=False)
image_table
"""),
            code(r"""
fig, ax = plt.subplots(figsize=(6.3, 5.2))
mat = cm.to_numpy()
img = ax.imshow(mat, cmap="Blues")
ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=35, ha="right")
ax.set_yticks(range(len(cm.index)), cm.index)
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_title(f"Damage classifier (macro-F1={im['macro_f1']:.3f})")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]): ax.text(j, i, int(mat[i,j]), ha="center", va="center", color="white" if mat[i,j] > mat.max()/2 else "black", fontsize=8)
fig.colorbar(img, ax=ax, fraction=.046); fig.tight_layout(); fig.savefig(FIGURES / "figure_07_image_confusion_matrix.png", bbox_inches="tight"); plt.show()
"""),
            code(r"""
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(12, 6.6)); ax.axis("off")
boxes = {
 "claim": (0.03,.70,.18,.16,"Structured claim\nfraud model",BLUE),
 "text": (0.03,.42,.18,.16,"Claim/witness notes\nconsistency checks",GOLD),
 "image": (0.03,.14,.18,.16,"Damage images\nclass + confidence",RED),
 "cal": (.31,.70,.20,.16,"5-fold sigmoid\ncalibration",BLUE),
 "cons": (.31,.28,.20,.22,"Evidence consistency\n(no fraud inference)",GOLD),
 "policy": (.61,.53,.18,.20,"Three-way policy\napprove / abstain /\ninvestigate",GREEN),
 "audit": (.84,.53,.14,.20,"Fairness + XAI\naudit trail",GREY),
}
for key,(x,y,w,h,label,color) in boxes.items():
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.012",facecolor="white",edgecolor=color,linewidth=2))
    ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=10)
def arrow(a,b):
    xa,ya,wa,ha,*_=boxes[a]; xb,yb,wb,hb,*_=boxes[b]
    ax.annotate("",xy=(xb,yb+hb/2),xytext=(xa+wa,ya+ha/2),arrowprops=dict(arrowstyle="->",lw=1.8,color="#374151"))
for a,b in [("claim","cal"),("text","cons"),("image","cons"),("cal","policy"),("cons","policy"),("policy","audit")]: arrow(a,b)
ax.text(.5,.96,"FairClaim-MM prototype architecture",ha="center",va="center",fontsize=16,fontweight="bold",color=BLUE)
ax.text(.5,.03,"Evidence can only escalate review; it never lowers a referral or changes the calibrated fraud probability.",ha="center",fontsize=10,color=GREY)
fig.tight_layout(); fig.savefig(FIGURES / "figure_08_architecture.png", bbox_inches="tight"); plt.show()
"""),
            md("The image result (accuracy 0.554; macro-F1 0.524) is a feasibility baseline for six damage/view classes. At confidence ≥0.65, an image–note contradiction may escalate approve→abstain or abstain/investigate→investigate; it never reduces scrutiny and never changes the calibrated fraud score."),
        ],
    )

    for name, nb in [
        ("01_data_audit_eda.ipynb", nb1),
        ("02_model_imbalance_comparison.ipynb", nb2),
        ("03_calibration_decision_policy.ipynb", nb3),
        ("04_fairness_xai.ipynb", nb4),
        ("05_multimodal_evidence.ipynb", nb5),
    ]:
        print(save_execute(name, nb))


if __name__ == "__main__":
    main()
