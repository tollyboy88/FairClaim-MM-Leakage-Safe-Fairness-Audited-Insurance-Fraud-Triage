from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import SGDClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fairclaim.config import ARTIFACT_DIR, RANDOM_STATE, REPORT_DIR  # noqa: E402
from fairclaim.evidence import HOG_PARAMETERS, image_features  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "datasets" / "comprehensive-car-damage"
MODEL_PATH = ARTIFACT_DIR / "image_damage_bundle.joblib"
CLASSES = ["F_Breakage", "F_Crushed", "F_Normal", "R_Breakage", "R_Crushed", "R_Normal"]


def main() -> None:
    paths: list[Path] = []
    labels: list[str] = []
    for label in CLASSES:
        for path in sorted((DATA_ROOT / label).iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(path)
                labels.append(label)
    if len(paths) != 2300:
        raise ValueError(f"Expected 2,300 images, found {len(paths)}")
    print("Extracting HOG features from", len(paths), "images")
    X = np.vstack([image_features(path) for path in paths])
    y = np.asarray(labels)
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model = CalibratedClassifierCV(
        estimator=SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=0.0005,
            l1_ratio=0.05,
            class_weight="balanced",
            max_iter=3000,
            tol=1e-4,
            random_state=RANDOM_STATE,
        ),
        method="sigmoid",
        cv=cv,
    )
    model.fit(X[train_idx], y[train_idx])
    probability = model.predict_proba(X[test_idx])
    prediction = model.classes_[np.argmax(probability, axis=1)]
    metrics = {
        "test_rows": int(len(test_idx)),
        "accuracy": float(accuracy_score(y[test_idx], prediction)),
        "macro_f1": float(f1_score(y[test_idx], prediction, average="macro")),
        "classification_report": classification_report(y[test_idx], prediction, output_dict=True),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        confusion_matrix(y[test_idx], prediction, labels=CLASSES), index=CLASSES, columns=CLASSES
    ).to_csv(REPORT_DIR / "image_damage_confusion_matrix.csv")
    (REPORT_DIR / "image_damage_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    version = datetime.now(timezone.utc).strftime("image-damage-%Y%m%dT%H%M%SZ")
    joblib.dump(
        {
            "model": model,
            "classes": model.classes_.tolist(),
            "model_version": version,
            "feature_extractor": "HOG plus RGB mean/std",
            "hog_parameters": HOG_PARAMETERS,
            "image_size": [64, 64],
            "metrics": metrics,
            "limitations": "Damage class only; not a fraud classifier and not trained on uploaded claimant evidence.",
        },
        MODEL_PATH,
    )
    print(json.dumps({"model_version": version, **metrics}, indent=2))


if __name__ == "__main__":
    main()
