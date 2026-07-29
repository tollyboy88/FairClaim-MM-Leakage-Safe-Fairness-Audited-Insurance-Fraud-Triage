"""Enrich analysis outputs and generate the data-bound manuscript sources."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AOA = ROOT / "publication" / "aoa"
TABLES = AOA / "tables"
FIGURES = AOA / "figures"
MANUSCRIPT = AOA / "manuscript"
REPORTS = ROOT / "reports"


def percent(value: float, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}"


def number(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def wilson(successes: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - half, centre + half


def enrich_fairness(frame: pd.DataFrame) -> pd.DataFrame:
    if "approval_rate_ci_low" in frame.columns:
        return frame
    rows = []
    for _, row in frame.iterrows():
        item = row.to_dict()
        n = int(row["n"])
        approved = int(round(row["approval_rate"] * n))
        abstained = int(round(row["abstention_rate"] * n))
        investigated = n - approved - abstained
        fraud = int(round(row["fraud_prevalence"] * n))
        missed = int(row["fraud_missed_in_approval_n"])
        approved_legitimate = int(round(row["approval_region_npv"] * approved)) if approved else 0
        for prefix, successes, denominator in [
            ("approval_rate", approved, n),
            ("abstention_rate", abstained, n),
            ("investigation_rate", investigated, n),
            ("fraud_missed_in_approval_rate", missed, fraud),
            ("approval_region_npv", approved_legitimate, approved),
        ]:
            low, high = wilson(successes, denominator)
            item[f"{prefix}_ci_low"] = low
            item[f"{prefix}_ci_high"] = high
        rows.append(item)
    return pd.DataFrame(rows)


def cost_comparison_text(costs: pd.DataFrame) -> tuple[str, str]:
    comparisons = []
    lower_count = 0
    for scenario in costs["scenario"].drop_duplicates():
        subset = costs[costs["scenario"] == scenario].set_index("policy")
        binary = float(subset.loc["Binary investigate", "expected_cost_per_claim"])
        selective = float(subset.loc["Three-way selective", "expected_cost_per_claim"])
        difference = selective - binary
        if difference < 0:
            lower_count += 1
        comparisons.append(
            f"{scenario}: £{selective:,.2f} per claim for three-way versus "
            f"£{binary:,.2f} for binary ({difference:+,.2f})"
        )
    if lower_count == len(comparisons):
        direction = "lower illustrative expected cost"
    elif lower_count == 0:
        direction = "higher illustrative expected cost"
    else:
        direction = "scenario-dependent illustrative expected cost"
    return direction, "; ".join(comparisons) + "."


def escape_latex_text(text: str) -> str:
    placeholders = {}
    for index, match in enumerate(re.finditer(r"`([^`]+)`", text)):
        token = f"ZZCODE{index}ZZ"
        placeholders[token] = r"\texttt{" + match.group(1).replace("_", r"\_") + "}"
        text = text.replace(match.group(0), token, 1)
    text = text.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
    text = re.sub(r"\*\*(.*?)\*\*", r"\\textbf{\1}", text)
    for token, replacement in placeholders.items():
        text = text.replace(token, replacement)
    return text


def markdown_to_latex(markdown: str) -> str:
    preamble = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,array}
\usepackage[hidelinks]{hyperref}
\usepackage{setspace}
\usepackage{lineno}
\usepackage{newunicodechar}
\newunicodechar{£}{\pounds}
\doublespacing
\linenumbers
\title{Calibrated Selective Prediction for Leakage-Safe and Fairness-Audited Motor Insurance Fraud Triage}
\author{Anonymous manuscript}
\date{}
\begin{document}
\maketitle
"""
    output = [preamble]
    in_abstract = False
    in_enumerate = False
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            if in_enumerate:
                output.append(r"\end{enumerate}")
                in_enumerate = False
            continue
        if line == "{{REFERENCES}}":
            if in_abstract:
                output.append(r"\end{abstract}")
                in_abstract = False
            output.append(r"\section*{References}")
            output.append(r"\begin{enumerate}")
            for reference in (MANUSCRIPT / "references_harvard.md").read_text(encoding="utf-8").splitlines():
                if reference.strip() and not reference.startswith("#"):
                    output.append(r"\item " + escape_latex_text(reference.strip()))
            output.append(r"\end{enumerate}")
            continue
        if line == "# Abstract":
            output.append(r"\begin{abstract}")
            in_abstract = True
            continue
        if line.startswith("# "):
            if in_abstract:
                output.append(r"\end{abstract}")
                in_abstract = False
            heading = line[2:]
            command = "section*" if heading == "Declarations" else "section"
            output.append(fr"\{command}{{{escape_latex_text(heading)}}}")
            continue
        if line.startswith("## "):
            output.append(fr"\subsection{{{escape_latex_text(line[3:])}}}")
            continue
        if line.startswith("$$") and line.endswith("$$"):
            output.extend([r"\begin{equation}", line[2:-2].strip(), r"\end{equation}"])
            continue
        if line.startswith("{{TABLE:") and line.endswith("}}"):
            filename, caption, columns = [part.strip() for part in line[8:-2].split("|", 2)]
            frame = pd.read_csv(TABLES / filename)[[column.strip() for column in columns.split(",")]]
            latex = frame.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.4f}")
            output.extend(
                [
                    r"\begin{table}[htbp]",
                    r"\centering\scriptsize",
                    fr"\caption{{{escape_latex_text(caption)}}}",
                    r"\resizebox{\textwidth}{!}{%",
                    latex,
                    r"}",
                    r"\end{table}",
                ]
            )
            continue
        if line.startswith("{{FIGURE:") and line.endswith("}}"):
            filename, caption = [part.strip() for part in line[9:-2].split("|", 1)]
            output.extend(
                [
                    r"\begin{figure}[htbp]",
                    r"\centering",
                    fr"\includegraphics[width=0.92\textwidth]{{../figures/{filename}}}",
                    fr"\caption{{{escape_latex_text(caption)}}}",
                    r"\end{figure}",
                ]
            )
            continue
        if re.match(r"^\d+\. ", line):
            if not in_enumerate:
                output.append(r"\begin{enumerate}")
                in_enumerate = True
            output.append(r"\item " + escape_latex_text(re.sub(r"^\d+\. ", "", line)))
            continue
        output.append(escape_latex_text(line) + "\n")
    if in_enumerate:
        output.append(r"\end{enumerate}")
    if in_abstract:
        output.append(r"\end{abstract}")
    output.append(r"\end{document}")
    return "\n".join(output)


def main() -> None:
    tables = {
        path.name: pd.read_csv(path, keep_default_na=False)
        for path in TABLES.glob("*.csv")
    }
    expected = {f"table_{index:02d}" for index in range(1, 15)}
    present = {path.stem[:8] for path in TABLES.glob("table_*.csv")}
    if not expected.issubset(present):
        raise FileNotFoundError("Run scripts/run_aoa_revision_analyses.py before finalising.")

    fairness = enrich_fairness(tables["table_08_three_action_fairness.csv"])
    fairness.to_csv(TABLES / "table_08_three_action_fairness.csv", index=False)
    fairness.to_csv(ROOT / "reports" / "aoa_revision" / "table_08_three_action_fairness.csv", index=False)
    tables["table_08_three_action_fairness.csv"] = fairness

    importance = pd.read_csv(REPORTS / "permutation_importance.csv")
    importance.to_csv(TABLES / "table_15_permutation_importance.csv", index=False)
    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    top = importance.head(10).sort_values("importance_mean")
    axis.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#375a6d")
    axis.set_xlabel("Decrease in held-out PR-AUC")
    axis.set_title("Global permutation sensitivity")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_07_permutation_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    model = tables["table_01_model_family_no_balancing.csv"]
    imbalance = tables["table_02_hgb_imbalance_strategies.csv"]
    age = tables["table_03_age_proxy_sensitivity.csv"].set_index("specification")
    calibration = tables["table_04_calibration_comparison.csv"].set_index("method")
    decision = tables["table_05_decision_regions_with_ci.csv"].set_index("action")
    metric_ci = tables["table_06_metric_bootstrap_ci.csv"].set_index("metric")
    threshold_samples = tables["table_07_threshold_bootstrap_samples.csv"]
    capacity = tables["table_09_fixed_capacity.csv"].set_index("investigation_capacity")
    costs = tables["table_10_operational_cost_sensitivity.csv"]
    ablations = tables["table_11_framework_ablations.csv"].set_index("configuration")
    seeds = tables["table_13_repeated_seeds.csv"]
    temporal = tables["table_14_temporal_validation.csv"].iloc[0]

    best_model = model.sort_values("cv_pr_auc_mean", ascending=False).iloc[0]
    best_imbalance = imbalance.sort_values("validation_pr_auc", ascending=False).iloc[0]
    best_strategy = (
        "no balancing"
        if str(best_imbalance["imbalance_strategy"]).strip().lower() == "none"
        else str(best_imbalance["imbalance_strategy"])
    )
    retained = age.loc["Retain AgeOfPolicyHolder"]
    removed = age.loc["Exclude all direct age-derived fields"]
    cost_direction, cost_text = cost_comparison_text(costs)
    action_gaps = []
    for attribute, subset in fairness.groupby("attribute"):
        action_gaps.append(
            {
                "attribute": attribute,
                "approval": subset["approval_rate"].max() - subset["approval_rate"].min(),
                "abstention": subset["abstention_rate"].max() - subset["abstention_rate"].min(),
                "investigation": subset["investigation_rate"].max() - subset["investigation_rate"].min(),
            }
        )
    gaps = pd.DataFrame(action_gaps)
    max_inv = gaps.loc[gaps["investigation"].idxmax()]
    source_data = pd.read_csv(ROOT / "fraud_oracle.csv")

    pr_ci = metric_ci.loc["pr_auc"]
    recall_ci = metric_ci.loc["recall"]
    raw = ablations.loc["Uncalibrated binary"]
    complete = ablations.loc["Complete framework"]
    sigmoid = calibration.loc["sigmoid"]
    isotonic = calibration.loc["isotonic"]
    calibration_choice = (
        "sigmoid"
        if sigmoid["validation_brier"] <= isotonic["validation_brier"]
        else "isotonic"
    )
    temporal_delta = float(temporal["test_pr_auc"]) - float(pr_ci["estimate"])

    replacements = {
        "N_ROWS": f"{len(source_data):,}",
        "N_FRAUD": f"{int(source_data['FraudFound_P'].sum()):,}",
        "PREVALENCE_PCT": percent(source_data["FraudFound_P"].mean(), 2),
        "TEST_PR_AUC": number(pr_ci["estimate"]),
        "TEST_PR_AUC_LOW": number(pr_ci["ci_2_5"]),
        "TEST_PR_AUC_HIGH": number(pr_ci["ci_97_5"]),
        "TEST_ROC_AUC": number(metric_ci.loc["roc_auc", "estimate"]),
        "TEST_BRIER": number(metric_ci.loc["brier_score", "estimate"]),
        "TEST_PRECISION": number(metric_ci.loc["precision", "estimate"]),
        "TEST_RECALL": number(metric_ci.loc["recall", "estimate"]),
        "TEST_RECALL_LOW": number(recall_ci["ci_2_5"]),
        "TEST_RECALL_HIGH": number(recall_ci["ci_97_5"]),
        "APPROVE_THRESHOLD": f"{sigmoid['approve_threshold']:.6f}",
        "INVESTIGATE_THRESHOLD": f"{sigmoid['investigate_threshold']:.6f}",
        "APPROVE_N": f"{int(decision.loc['approve', 'claims']):,}",
        "ABSTAIN_N": f"{int(decision.loc['abstain', 'claims']):,}",
        "INVESTIGATE_N": f"{int(decision.loc['investigate', 'claims']):,}",
        "APPROVE_FRAUD_RATE": percent(decision.loc["approve", "fraud_rate"]) + "%",
        "ABSTAIN_FRAUD_RATE": percent(decision.loc["abstain", "fraud_rate"]) + "%",
        "INVESTIGATE_FRAUD_RATE": percent(decision.loc["investigate", "fraud_rate"]) + "%",
        "APPROVE_NPV": percent(decision.loc["approve", "npv_if_approved"], 2) + "%",
        "CAP5_CAPTURE": percent(capacity.loc[0.05, "fraud_capture_rate"]),
        "CAP10_CAPTURE": percent(capacity.loc[0.10, "fraud_capture_rate"]),
        "CAP20_CAPTURE": percent(capacity.loc[0.20, "fraud_capture_rate"]),
        "CAP30_CAPTURE": percent(capacity.loc[0.30, "fraud_capture_rate"]),
        "CAP5_PRECISION": percent(capacity.loc[0.05, "investigation_precision"]),
        "CAP10_PRECISION": percent(capacity.loc[0.10, "investigation_precision"]),
        "CAP30_PRECISION": percent(capacity.loc[0.30, "investigation_precision"]),
        "CAP10": "10",
        "COST_DIRECTION": cost_direction,
        "COST_RESULT_TEXT": cost_text,
        "RAW_BRIER": number(raw["brier_score"]),
        "CAL_BRIER": number(complete["brier_score"]),
        "SELECTIVE_COVERAGE": percent(complete["coverage"]) + "%",
        "SELECTIVE_RISK": percent(complete["selective_error_risk"]) + "%",
        "AGE_RETAIN_PR": number(retained["test_pr_auc"]),
        "AGE_REMOVE_PR": number(removed["test_pr_auc"]),
        "AGE_RETAIN_BRIER": number(retained["test_brier"]),
        "AGE_REMOVE_BRIER": number(removed["test_brier"]),
        "AGE_RETAIN_NPV": percent(retained["approval_npv"], 2) + "%",
        "AGE_REMOVE_NPV": percent(removed["approval_npv"], 2) + "%",
        "AGE_RETAIN_GAP": percent(retained["age_investigation_rate_gap"]) + "%",
        "AGE_REMOVE_GAP": percent(removed["age_investigation_rate_gap"]) + "%",
        "AGE_INTERPRETATION": (
            "Removal reduced the observed age disparity without a material PR-AUC loss."
            if removed["age_investigation_rate_gap"] < retained["age_investigation_rate_gap"]
            and removed["test_pr_auc"] >= retained["test_pr_auc"] - 0.01
            else "Removal exposed a measurable performance–disparity trade-off that requires governance judgement."
        ),
        "MAX_INV_GAP": percent(max_inv["investigation"]) + "%",
        "MAX_INV_ATTRIBUTE": str(max_inv["attribute"]),
        "MAX_APPROVAL_GAP": percent(gaps["approval"].max()) + "%",
        "MAX_ABSTAIN_GAP": percent(gaps["abstention"].max()) + "%",
        "IMPORTANCE_TEXT": ", ".join(
            f"{row.feature} ({row.importance_mean:.3f})" for row in importance.head(6).itertuples()
        ),
        "SIGMOID_BRIER": number(sigmoid["test_brier"]),
        "ISOTONIC_BRIER": number(isotonic["test_brier"]),
        "CALIBRATION_CHOICE": calibration_choice,
        "SEED_PR_MEAN": number(seeds["test_pr_auc"].mean()),
        "SEED_PR_SD": number(seeds["test_pr_auc"].std(ddof=1)),
        "APPROVE_BOOT_MEDIAN": number(threshold_samples["approve_below"].median()),
        "APPROVE_BOOT_LOW": number(threshold_samples["approve_below"].quantile(0.025)),
        "APPROVE_BOOT_HIGH": number(threshold_samples["approve_below"].quantile(0.975)),
        "INV_BOOT_MEDIAN": number(threshold_samples["investigate_at_or_above"].median()),
        "INV_BOOT_LOW": number(threshold_samples["investigate_at_or_above"].quantile(0.025)),
        "INV_BOOT_HIGH": number(threshold_samples["investigate_at_or_above"].quantile(0.975)),
        "TEMPORAL_PR_AUC": number(temporal["test_pr_auc"]),
        "TEMPORAL_ROC_AUC": number(temporal["test_roc_auc"]),
        "TEMPORAL_BRIER": number(temporal["test_brier"]),
        "TEMPORAL_PRECISION": number(temporal["test_precision"]),
        "TEMPORAL_RECALL": number(temporal["test_recall"]),
        "TEMPORAL_INTERPRETATION": (
            f"Temporal PR-AUC was {abs(temporal_delta):.3f} "
            + ("lower" if temporal_delta < 0 else "higher")
            + " than the primary random-split estimate, demonstrating sensitivity to evaluation period."
        ),
        "RQ1_TEXT": (
            f"{best_model['model']} ranked first without balancing on the prespecified family-selection "
            f"criterion, with mean five-fold training "
            f"PR-AUC {best_model['cv_pr_auc_mean']:.3f} (SD {best_model['cv_pr_auc_sd']:.3f}) and "
            f"validation PR-AUC {best_model['validation_pr_auc']:.3f}. Random forest had lower mean "
            f"cross-validated PR-AUC ({model.loc[model['model'].eq('Random forest'), 'cv_pr_auc_mean'].iloc[0]:.3f}) "
            f"but a fixed-validation PR-AUC only "
            f"{model.loc[model['model'].eq('Random forest'), 'validation_pr_auc'].iloc[0] - best_model['validation_pr_auc']:.6f} "
            f"higher. In the separate imbalance "
            f"experiment, {best_strategy} ranked first with validation PR-AUC "
            f"{best_imbalance['validation_pr_auc']:.3f}. The result supports {best_model['model']} "
            f"with {best_strategy.lower()} for this benchmark; it does not "
            "generalise to a claim that balancing is always ineffective."
        ),
        "BEST_MODEL": str(best_model["model"]),
        "AGE_DISCUSSION": (
            f"test PR-AUC changed from {retained['test_pr_auc']:.3f} to {removed['test_pr_auc']:.3f}, "
            f"while the age investigation-rate gap changed from {100*retained['age_investigation_rate_gap']:.1f}% "
            f"to {100*removed['age_investigation_rate_gap']:.1f}%."
        ),
        "ROBUSTNESS_DISCUSSION": (
            f"The temporal PR-AUC was {temporal['test_pr_auc']:.3f} versus the random-split "
            f"estimate of {pr_ci['estimate']:.3f}, so the operational policy should be revalidated by period."
        ),
        "CONCLUSION_RQ1": (
            f"RQ1 selected {best_model['model']} with {best_strategy.lower()}."
        ),
        "CONCLUSION_RQ2": (
            f"RQ2 quantified fraud capture from {percent(capacity.loc[0.05, 'fraud_capture_rate'])}% "
            f"at 5% capacity to {percent(capacity.loc[0.30, 'fraud_capture_rate'])}% at 30% capacity and "
            f"found {cost_direction} for three-way triage under the declared scenarios."
        ),
        "CONCLUSION_RQ3": (
            f"RQ3 showed that removing AgeOfPolicyHolder changed PR-AUC from {retained['test_pr_auc']:.3f} "
            f"to {removed['test_pr_auc']:.3f}, while subgroup differences remained across all actions."
        ),
        "CONCLUSION_RQ4": (
            f"RQ4 found mean repeated-seed PR-AUC {seeds['test_pr_auc'].mean():.3f} and temporal "
            f"PR-AUC {temporal['test_pr_auc']:.3f}, making uncertainty and period shift explicit."
        ),
    }

    template = (MANUSCRIPT / "main.md.template").read_text(encoding="utf-8")
    for key, value in replacements.items():
        template = template.replace(f"[[{key}]]", str(value))
    unresolved = sorted(set(re.findall(r"\[\[([A-Z0-9_]+)\]\]", template)))
    if unresolved:
        raise ValueError(f"Unresolved manuscript placeholders: {unresolved}")
    (MANUSCRIPT / "main.md").write_text(template, encoding="utf-8")
    (MANUSCRIPT / "main.tex").write_text(markdown_to_latex(template), encoding="utf-8")
    print(MANUSCRIPT / "main.md")
    print(MANUSCRIPT / "main.tex")


if __name__ == "__main__":
    main()
