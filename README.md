# FairClaim-MM

FairClaim-MM is a local, fairness-aware insurance fraud research prototype. It trains a calibrated tabular fraud model and provides a Streamlit dashboard for three-way claim routing:

- **approve**: low calibrated fraud risk; continue normal processing
- **investigate**: elevated fraud risk; refer to an authorised investigator
- **abstain**: uncertain score; obtain more evidence or human review

The output is decision support, not proof of fraud or an automatic claim denial.

## Current evidence contract

The production score uses the structured fields in `fraud_oracle.csv`. `PolicyNumber` is excluded as an identifier. `Sex`, `Age`, and `MaritalStatus` are excluded from model inputs and retained only for subgroup auditing.

The two image datasets and Memgraph project are separate research branches. They are not randomly linked to tabular claims, and uploaded evidence does not affect the current tabular score.

## Run from VS Code or PowerShell

```powershell
python -m pip install -r requirements.txt
python scripts/train_models.py
python -m streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Dashboard pages

1. **Overview** — held-out performance, model comparison, and decision distribution.
2. **Single claim** — guided form, optional locally stored evidence, three-way decision, and local explanation.
3. **Batch upload** — downloadable template, CSV validation, bulk scoring, local storage, and result download.
4. **Fairness & explanations** — subgroup gaps and global permutation importance.
5. **Local history** — recent decisions saved in the local SQLite database.

The SQLite database is created at `local_storage/fairclaim.db`. Evidence uploaded through the single-claim page is stored as a database BLOB and is not sent to an external service.

See `docs/input_data_guide.md` for required fields, optional audit columns, evidence limits, and decision meanings.
See `docs/decision_calculation.md` for the exact calibrated-probability, threshold, and evidence-fusion equations.
See `docs/multimodal_data_assessment.md` for the added INSURBIAS benchmark, rejected candidates, and the linked data needed for a genuine multimodal fraud model.

## Reproducibility and leakage controls

- Stratified 60/20/20 train/validation/test split.
- Preprocessing is fitted inside model pipelines.
- Five-fold cross-validation is performed only within training data.
- The strongest model is selected using validation PR-AUC.
- Probability calibration is fitted from training folds.
- Decision thresholds are selected on validation predictions.
- The test set is evaluated once and is not used for model or threshold selection.

Generated files are stored under `artifacts/` and `reports/`.
