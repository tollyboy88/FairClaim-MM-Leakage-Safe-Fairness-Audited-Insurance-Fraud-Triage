from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fairclaim.config import (
    DATABASE_PATH,
    DECISION_LABELS,
    IMAGE_MODEL_BUNDLE_PATH,
    MODEL_BUNDLE_PATH,
    REPORT_DIR,
)
from fairclaim.evidence import (
    analyse_image,
    assess_evidence_consistency,
    fuse_review_route,
    load_image_bundle,
)
from fairclaim.inference import score_frame, score_single
from fairclaim.metrics import fairness_table
from fairclaim.storage import load_history, save_batch_predictions, save_prediction

st.set_page_config(page_title="FairClaim-MM", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_model_bundle() -> dict:
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError("Run `python scripts/train_models.py` before opening the dashboard.")
    return joblib.load(MODEL_BUNDLE_PATH)


@st.cache_resource
def load_optional_image_bundle() -> dict | None:
    return load_image_bundle(IMAGE_MODEL_BUNDLE_PATH) if IMAGE_MODEL_BUNDLE_PATH.exists() else None


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def decision_banner(decision: str, probability: float) -> None:
    labels = {
        "approve": ("Approved for normal processing", st.success),
        "defer": ("Defer — obtain more evidence or human review", st.warning),
        "abstain": ("Defer — obtain more evidence or human review (legacy label)", st.warning),
        "investigate": ("Investigate — elevated fraud risk", st.error),
    }
    title, renderer = labels[decision]
    renderer(f"**{title}** · calibrated fraud probability: **{probability:.1%}**")
    st.caption(DECISION_LABELS[decision])


def schema_table(bundle: dict) -> pd.DataFrame:
    rows = []
    for category, fields in bundle["input_schema"].items():
        for name, spec in fields.items():
            rows.append(
                {
                    "field": name,
                    "role": category.replace("_", " "),
                    "type": spec["type"],
                    "required": spec.get("required", False),
                    "allowed/example": ", ".join(spec.get("options", [])[:8])
                    if spec["type"] == "category"
                    else spec.get("default", "optional reference"),
                }
            )
    return pd.DataFrame(rows)


def default_template(bundle: dict, rows: int = 3) -> pd.DataFrame:
    record = {}
    for area in ("model_features", "audit_fields"):
        for name, spec in bundle["input_schema"][area].items():
            record[name] = spec.get("default", "")
    record["PolicyNumber"] = "YOUR_REFERENCE"
    return pd.DataFrame([record.copy() for _ in range(rows)])


def overview_page(bundle: dict) -> None:
    st.title("FairClaim-MM decision support")
    st.write(
        "A fairness-aware insurance fraud research prototype. It produces calibrated risk and "
        "routes claims to **approve**, **defer**, or **investigate**; it does not make a final legal decision."
    )
    metrics = bundle["test_metrics"]
    cols = st.columns(5)
    cols[0].metric("Test PR-AUC", f"{metrics['pr_auc']:.3f}")
    cols[1].metric("Test ROC-AUC", f"{metrics['roc_auc']:.3f}")
    cols[2].metric("Fraud recall", f"{metrics['recall']:.1%}")
    cols[3].metric("Precision", f"{metrics['precision']:.1%}")
    cols[4].metric("Brier score", f"{metrics['brier_score']:.3f}")

    st.info(
        f"Selected model: **{bundle['selected_model']}** · version `{bundle['model_version']}`. "
        "Policy number and protected attributes are excluded from the fraud score."
    )
    with st.expander("Exact decision calculation", expanded=False):
        low = bundle["thresholds"]["approve_below"]
        high = bundle["thresholds"]["investigate_at_or_above"]
        st.code(
            f"if calibrated_probability < {low:.8f}: approve\n"
            f"elif calibrated_probability >= {high:.8f}: investigate\n"
            "else: defer"
        )
        st.write(
            "The investigation boundary maximises validation F2, where "
            "F2 = 5 × precision × recall / (4 × precision + recall). "
            "The approval boundary is the largest lower threshold retaining at least 99% validation NPV."
        )
        st.caption(
            "The calibrated probability is the mean of five sigmoid-calibrated gradient-boosting models. "
            "See docs/decision_calculation.md for the full equation."
        )
    left, right = st.columns(2)
    revision_tables = PROJECT_ROOT / "publication" / "aoa" / "tables"
    revised_comparison = revision_tables / "table_01_model_family_no_balancing.csv"
    comparison = load_csv(
        revised_comparison if revised_comparison.exists() else REPORT_DIR / "model_comparison.csv"
    )
    with left:
        st.subheader("Validation model comparison")
        fig = px.bar(
            comparison.sort_values("validation_pr_auc"),
            x="validation_pr_auc",
            y="model",
            orientation="h",
            labels={"validation_pr_auc": "Validation PR-AUC", "model": "Model"},
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Held-out decision distribution")
        predictions = load_csv(REPORT_DIR / "held_out_test_predictions.csv")
        counts = predictions["decision"].value_counts().rename_axis("decision").reset_index(name="claims")
        fig = px.pie(counts, names="decision", values="claims", hole=0.45)
        st.plotly_chart(fig, width="stretch")
    revised_imbalance = revision_tables / "table_02_hgb_imbalance_strategies.csv"
    imbalance_path = (
        revised_imbalance
        if revised_imbalance.exists()
        else REPORT_DIR / "imbalance_strategy_comparison.csv"
    )
    if imbalance_path.exists():
        st.subheader("Class-imbalance strategy comparison")
        imbalance = load_csv(imbalance_path)
        st.caption(
            "Resampling strategies were applied inside training folds only. No balancing achieved the strongest validation PR-AUC."
        )
        st.dataframe(imbalance, width="stretch", hide_index=True)
    revised_decisions = revision_tables / "table_05_decision_regions_with_ci.csv"
    decision_summary_path = (
        revised_decisions
        if revised_decisions.exists()
        else REPORT_DIR / "decision_policy_test_summary.csv"
    )
    if decision_summary_path.exists():
        st.subheader("Observed outcomes by decision route")
        st.dataframe(load_csv(decision_summary_path), width="stretch", hide_index=True)

    st.subheader("Evidence branches and limitations")
    st.markdown(
        """
        - **Current production score:** structured Fraud Oracle claim fields.
        - **Damage images:** a separate HOG image model identifies six front/rear damage classes; test accuracy is 55.4%, so only high-confidence contradictions can escalate review.
        - **Ecommerce evidence images:** synthetic and useful for controlled evidence experiments only.
        - **Graph data:** synthetic Memgraph branch for linked-claim/network research.

        Images are not randomly linked to tabular training rows. Image/note contradictions can conservatively
        escalate the workflow route, but they do not alter the calibrated tabular fraud probability.
        """
    )


def single_claim_page(bundle: dict) -> None:
    st.title("Assess one claim")
    st.write(
        "Complete the structured claim form. Required model fields determine the score; audit-only fields "
        "measure fairness and do not enter the model. Optional evidence is stored locally for an investigator."
    )
    with st.expander("What information is required?", expanded=False):
        st.dataframe(schema_table(bundle), width="stretch", hide_index=True)

    schema = bundle["input_schema"]
    image_bundle = load_optional_image_bundle()
    with st.form("single_claim"):
        claim_reference = st.text_input("Claim reference (optional)")
        values = {}
        st.subheader("Claim data used by the model")
        columns = st.columns(3)
        for index, (name, spec) in enumerate(schema["model_features"].items()):
            with columns[index % 3]:
                if spec["type"] == "number":
                    values[name] = st.number_input(
                        name,
                        min_value=float(spec["minimum"]),
                        max_value=float(spec["maximum"]),
                        value=float(spec["default"]),
                    )
                else:
                    default_index = spec["options"].index(spec["default"])
                    values[name] = st.selectbox(name, spec["options"], index=default_index)

        st.subheader("Audit-only information")
        st.caption("These values are saved for fairness monitoring but are not supplied to the fraud model.")
        audit = {}
        audit_columns = st.columns(3)
        for index, (name, spec) in enumerate(schema["audit_fields"].items()):
            with audit_columns[index % 3]:
                if spec["type"] == "number":
                    audit[name] = st.number_input(
                        f"{name} (audit only)",
                        min_value=float(spec["minimum"]),
                        max_value=float(spec["maximum"]),
                        value=float(spec["default"]),
                    )
                else:
                    audit[name] = st.selectbox(f"{name} (audit only)", spec["options"])
        st.subheader("Narrative and multimedia evidence")
        claim_note = st.text_area(
            "Claimant written note",
            help="Describe what happened and, where relevant, whether damage is front/rear and broken/crushed.",
        )
        witness_note = st.text_area("Witness written note (optional)")
        audio_transcript = st.text_area(
            "Audio-note transcript (optional)",
            help="Audio is stored locally. Automatic speech transcription is not enabled in this CPU prototype, so paste a verified transcript here.",
        )
        evidence_uploads = st.file_uploader(
            "Supporting evidence (optional)",
            type=["png", "jpg", "jpeg", "pdf", "txt", "csv", "wav", "mp3", "m4a", "ogg"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("Assess claim", type="primary")

    if submitted:
        if len(evidence_uploads) > 5 or any(upload.size > 10 * 1024 * 1024 for upload in evidence_uploads):
            st.error("Upload at most five evidence files, no larger than 10 MB each.")
            return
        result = score_single(values, bundle, explain=True)
        image_results = []
        if image_bundle is not None:
            for upload in evidence_uploads:
                if upload.type in {"image/png", "image/jpeg"} or Path(upload.name).suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    try:
                        analysed = analyse_image(upload.getvalue(), image_bundle)
                        analysed["filename"] = upload.name
                        image_results.append(analysed)
                    except Exception as exc:
                        st.warning(f"Could not analyse image {upload.name}: {exc}")
        combined_witness = " ".join(part for part in (witness_note, audio_transcript) if part)
        consistency = assess_evidence_consistency(image_results, claim_note, combined_witness)
        final_decision, fusion_reason = fuse_review_route(result["decision"], consistency["status"])
        decision_banner(final_decision, result["fraud_probability"])
        st.write(f"Tabular route: **{result['decision']}**. Evidence-fusion route: **{final_decision}**.")
        st.caption(fusion_reason)
        if image_results:
            st.subheader("Image evidence analysis")
            st.caption(
                f"Image model `{image_bundle['model_version']}` classifies damage, not fraud. "
                f"Held-out accuracy: {image_bundle['metrics']['accuracy']:.1%}."
            )
            st.dataframe(pd.DataFrame(image_results), width="stretch", hide_index=True)
        st.subheader("Cross-evidence consistency")
        if consistency["status"] == "contradiction":
            st.warning("Contradiction detected: " + " ".join(consistency["contradictions"]))
        elif consistency["status"] == "consistent":
            st.success("The extracted image and note facts are consistent within this limited prototype check.")
        else:
            st.info("There was not enough explicit front/rear or damage-type information for a consistency comparison.")
        for warning in consistency["warnings"]:
            st.caption(warning)
        explanation = pd.DataFrame(result["explanation"])
        st.subheader("Why the score moved")
        st.caption(
            "One-feature-at-a-time model sensitivity relative to typical training values. "
            "This is not a causal explanation or proof of fraud."
        )
        figure = px.bar(
            explanation.sort_values("score_change"),
            x="score_change",
            y="feature",
            orientation="h",
            color="direction",
            labels={"score_change": "Change in fraud probability"},
        )
        st.plotly_chart(figure, width="stretch")
        with st.expander("Explanation details"):
            st.dataframe(explanation, width="stretch", hide_index=True)

        evidence = [(item.name, item.type, item.getvalue()) for item in evidence_uploads]
        audit_record = {
            **audit,
            "claim_note": claim_note,
            "witness_note": witness_note,
            "audio_transcript": audio_transcript,
            "tabular_decision": result["decision"],
            "evidence_consistency": consistency,
            "image_results": image_results,
            "fusion_reason": fusion_reason,
        }
        prediction_id = save_prediction(
            DATABASE_PATH,
            source="single",
            model_version=bundle["model_version"],
            probability=result["fraud_probability"],
            decision=final_decision,
            inputs=values,
            audit=audit_record,
            explanation=result["explanation"],
            claim_reference=claim_reference or None,
            evidence=evidence,
        )
        st.caption(f"Saved locally with prediction ID `{prediction_id}`.")


def batch_page(bundle: dict) -> None:
    st.title("Assess multiple claims")
    st.write(
        "Upload CSV data with one claim per row. All model-feature columns are required. `PolicyNumber`, "
        "`Sex`, `Age`, and `MaritalStatus` are optional and used only for traceability/fairness reporting."
    )
    template = default_template(bundle)
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="fairclaim_batch_template.csv",
        mime="text/csv",
    )
    with st.expander("Column specification"):
        st.dataframe(schema_table(bundle), width="stretch", hide_index=True)
    upload = st.file_uploader("Upload claim CSV", type=["csv"], key="batch_csv")
    if upload is None:
        return
    try:
        frame = pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        return
    if frame.empty:
        st.error("The uploaded CSV has no rows.")
        return
    if len(frame) > 10_000:
        st.error("This prototype accepts at most 10,000 claims in one batch.")
        return
    st.write(f"Loaded **{len(frame):,}** claims.")
    st.dataframe(frame.head(20), width="stretch")
    if st.button("Validate and assess batch", type="primary"):
        try:
            scored = score_frame(frame, bundle)
        except ValueError as exc:
            st.error(str(exc))
            return
        batch_id = save_batch_predictions(
            DATABASE_PATH,
            model_version=bundle["model_version"],
            rows=scored,
            input_columns=bundle["model_features"],
            audit_columns=bundle["protected_attributes"],
        )
        st.success(f"Assessed and saved {len(scored):,} claims locally. Batch ID: `{batch_id}`")
        counts = scored["decision"].value_counts().rename_axis("decision").reset_index(name="claims")
        st.plotly_chart(px.bar(counts, x="decision", y="claims", color="decision"), width="stretch")
        st.dataframe(scored, width="stretch")
        st.download_button(
            "Download assessed claims",
            scored.to_csv(index=False).encode("utf-8"),
            file_name=f"fairclaim_results_{batch_id}.csv",
            mime="text/csv",
        )
        protected = set(bundle["protected_attributes"])
        if protected.issubset(scored.columns):
            st.subheader("Uploaded-batch decision rates by audit group")
            for attribute in bundle["protected_attributes"]:
                rates = (
                    scored.groupby(attribute, dropna=False)["decision"]
                    .value_counts(normalize=True)
                    .rename("rate")
                    .reset_index()
                )
                st.dataframe(rates, width="stretch", hide_index=True)
        if "FraudFound_P" in scored.columns and protected.issubset(scored.columns):
            detail, gaps = fairness_table(
                scored["FraudFound_P"],
                scored["fraud_probability"],
                bundle["thresholds"]["investigate_at_or_above"],
                scored[list(bundle["protected_attributes"])],
            )
            st.subheader("Labelled-batch fairness audit")
            st.dataframe(gaps, width="stretch", hide_index=True)
            st.dataframe(detail, width="stretch", hide_index=True)


def audit_page(bundle: dict) -> None:
    st.title("Fairness and explanation audit")
    st.warning(
        "Protected attributes are excluded from the score, but proxy variables and different base rates can "
        "still create unequal outcomes. Differences shown here require investigation; they are not proof of unlawful bias."
    )
    gaps = load_csv(REPORT_DIR / "fairness_gaps.csv")
    detail = load_csv(REPORT_DIR / "fairness_by_group.csv")
    st.subheader("Fairness gaps on the untouched test set")
    st.dataframe(gaps, width="stretch", hide_index=True)
    confidence_path = REPORT_DIR / "fairness_gap_confidence_intervals.csv"
    if confidence_path.exists():
        with st.expander("Bootstrap 95% confidence intervals"):
            st.caption(
                "Intervals are based on 500 resamples of the held-out test set. Wide intervals indicate unstable estimates."
            )
            st.dataframe(load_csv(confidence_path), width="stretch", hide_index=True)
    attribute = st.selectbox("Inspect protected grouping", detail["attribute"].unique())
    subset = detail[detail["attribute"] == attribute]
    metric = st.selectbox(
        "Metric",
        ["selection_rate", "recall_tpr", "false_positive_rate", "precision", "pr_auc", "mean_score"],
    )
    st.plotly_chart(px.bar(subset, x="group", y=metric, hover_data=["n"]), width="stretch")
    st.dataframe(subset, width="stretch", hide_index=True)

    st.subheader("Global model sensitivity")
    importance = load_csv(REPORT_DIR / "permutation_importance.csv").head(15)
    st.caption("Decrease in test PR-AUC when a feature is shuffled. Importance is predictive, not causal.")
    st.plotly_chart(
        px.bar(
            importance.sort_values("importance_mean"),
            x="importance_mean",
            y="feature",
            orientation="h",
            error_x="importance_std",
        ),
        width="stretch",
    )
    st.subheader("Governance controls")
    st.markdown(
        """
        - Use `investigate` as a referral signal, never as an automatic fraud finding.
        - Use `defer` to request more evidence or route an intermediate-risk claim to a trained reviewer.
        - Monitor false-positive and recall gaps after deployment with labelled outcomes.
        - Revalidate thresholds when prevalence, investigation capacity, or claim mix changes.
        - Keep a human appeal and override process and record the final adjudication separately.
        """
    )


def operational_research_page() -> None:
    st.title("Operational research evidence")
    st.write(
        "These results support the revised operations-research manuscript. They are held-out "
        "benchmark analyses, not estimates of realised insurer savings or production performance."
    )
    table_dir = PROJECT_ROOT / "publication" / "aoa" / "tables"
    figure_dir = PROJECT_ROOT / "publication" / "aoa" / "figures"
    required = table_dir / "table_09_fixed_capacity.csv"
    if not required.exists():
        st.info(
            "Run `python scripts/run_aoa_revision_analyses.py` to generate the operational "
            "evidence package."
        )
        return

    capacity = load_csv(required)
    costs = load_csv(table_dir / "table_10_operational_cost_sensitivity.csv")
    age_proxy = load_csv(table_dir / "table_03_age_proxy_sensitivity.csv")
    calibration = load_csv(table_dir / "table_04_calibration_comparison.csv")
    fairness = load_csv(table_dir / "table_08_three_action_fairness.csv")
    temporal = load_csv(table_dir / "table_14_temporal_validation.csv")

    st.subheader("Investigation capacity")
    st.caption("Fraud captured when investigators can review only the highest-risk claims.")
    st.plotly_chart(
        px.line(
            capacity,
            x="investigation_capacity",
            y="fraud_capture_rate",
            markers=True,
            hover_data=["claims_investigated", "investigation_precision"],
            labels={
                "investigation_capacity": "Share of claims investigated",
                "fraud_capture_rate": "Share of labelled fraud captured",
            },
        ),
        width="stretch",
    )
    st.dataframe(capacity, width="stretch", hide_index=True)

    st.subheader("Illustrative cost sensitivity")
    st.warning(
        "The GBP amounts are declared scenarios. They are not observed claim losses, prices, "
        "or guaranteed savings and must be replaced with an insurer's governed cost matrix."
    )
    st.plotly_chart(
        px.bar(
            costs,
            x="scenario",
            y="expected_cost_per_claim",
            color="policy",
            barmode="group",
            labels={"expected_cost_per_claim": "Illustrative expected cost per claim (£)"},
        ),
        width="stretch",
    )
    with st.expander("Cost assumptions and complete results"):
        st.dataframe(costs, width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Age-proxy sensitivity")
        st.dataframe(age_proxy, width="stretch", hide_index=True)
    with right:
        st.subheader("Calibration comparison")
        st.dataframe(calibration, width="stretch", hide_index=True)

    st.subheader("Three-action fairness")
    attribute = st.selectbox(
        "Protected grouping for action audit",
        fairness["attribute"].drop_duplicates().tolist(),
        key="operational_fairness_attribute",
    )
    subset = fairness[fairness["attribute"] == attribute]
    action_rates = subset.melt(
        id_vars=["group"],
        value_vars=["approval_rate", "defer_rate", "investigation_rate"],
        var_name="action",
        value_name="rate",
    )
    st.plotly_chart(
        px.bar(action_rates, x="group", y="rate", color="action", barmode="group"),
        width="stretch",
    )
    st.dataframe(subset, width="stretch", hide_index=True)

    st.subheader("Temporal stress test")
    st.caption("Train on 1994, select thresholds on 1995, and test on 1996; Year is excluded as a predictor.")
    st.dataframe(temporal, width="stretch", hide_index=True)
    with st.expander("Manuscript figures"):
        for path in sorted(figure_dir.glob("*.png")):
            st.image(str(path), caption=path.stem.replace("_", " ").title(), width="stretch")


def history_page(bundle: dict) -> None:
    st.title("Local decision history")
    st.write(f"Predictions are stored only in `{DATABASE_PATH}` on this computer.")
    history = load_history(DATABASE_PATH, limit=5000)
    if history.empty:
        st.info("No dashboard predictions have been saved yet.")
        return
    cols = st.columns(3)
    cols[0].metric("Saved predictions", f"{len(history):,}")
    cols[1].metric("Investigate rate", f"{(history['decision'] == 'investigate').mean():.1%}")
    deferred = history["decision"].isin(["defer", "abstain"])
    cols[2].metric("Defer rate", f"{deferred.mean():.1%}")
    st.dataframe(history, width="stretch", hide_index=True)
    st.download_button(
        "Export history",
        history.to_csv(index=False).encode("utf-8"),
        file_name="fairclaim_local_history.csv",
        mime="text/csv",
    )


def main() -> None:
    bundle = load_model_bundle()
    st.sidebar.title("FairClaim-MM")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Single claim",
            "Batch upload",
            "Fairness & explanations",
            "Operational research",
            "Local history",
        ],
    )
    st.sidebar.caption(f"Model: {bundle['model_version']}")
    if page == "Overview":
        overview_page(bundle)
    elif page == "Single claim":
        single_claim_page(bundle)
    elif page == "Batch upload":
        batch_page(bundle)
    elif page == "Fairness & explanations":
        audit_page(bundle)
    elif page == "Operational research":
        operational_research_page()
    else:
        history_page(bundle)


if __name__ == "__main__":
    main()
