from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "fraud_oracle.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"
LOCAL_STORAGE_DIR = PROJECT_ROOT / "local_storage"
MODEL_BUNDLE_PATH = ARTIFACT_DIR / "model_bundle.joblib"
IMAGE_MODEL_BUNDLE_PATH = ARTIFACT_DIR / "image_damage_bundle.joblib"
DATABASE_PATH = LOCAL_STORAGE_DIR / "fairclaim.db"

TARGET = "FraudFound_P"
ID_COLUMNS = ["PolicyNumber"]
PROTECTED_COLUMNS = ["Sex", "Age", "MaritalStatus"]
RANDOM_STATE = 42

# These exact fields are only used to report subgroup outcomes. The separate
# categorical field AgeOfPolicyHolder remains a model feature and should be
# treated as a possible age proxy during fairness evaluation.
AUDIT_ONLY_COLUMNS = PROTECTED_COLUMNS

DECISION_LABELS = {
    "approve": "Low calibrated fraud risk; suitable for normal processing.",
    "defer": "Intermediate calibrated risk; obtain more evidence or human review.",
    # Backward-compatible label for locally stored decisions created before
    # the terminology was updated.
    "abstain": "Legacy defer label; obtain more evidence or human review.",
    "investigate": "Elevated fraud risk; route to an authorised investigator.",
}
