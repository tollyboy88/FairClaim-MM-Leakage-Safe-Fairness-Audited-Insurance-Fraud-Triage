from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog


HOG_PARAMETERS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}


def image_features(source: str | Path | bytes) -> np.ndarray:
    if isinstance(source, bytes):
        image = Image.open(io.BytesIO(source))
    else:
        image = Image.open(source)
    with image:
        rgb = image.convert("RGB").resize((64, 64))
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    gray = np.dot(array[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    descriptor = hog(gray, feature_vector=True, **HOG_PARAMETERS)
    colour_summary = np.r_[array.mean(axis=(0, 1)), array.std(axis=(0, 1))]
    return np.r_[descriptor, colour_summary].astype(np.float32)


def load_image_bundle(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if not {"model", "classes", "model_version"}.issubset(bundle):
        raise ValueError("Invalid image model bundle")
    return bundle


def analyse_image(content: bytes, bundle: dict[str, Any]) -> dict[str, Any]:
    features = image_features(content).reshape(1, -1)
    probabilities = bundle["model"].predict_proba(features)[0]
    order = np.argsort(probabilities)[::-1]
    label = str(bundle["classes"][order[0]])
    side, damage = label.split("_", 1)
    return {
        "label": label,
        "confidence": float(probabilities[order[0]]),
        "vehicle_side": "front" if side == "F" else "rear",
        "damage_type": damage.lower(),
        "top_predictions": [
            {"label": str(bundle["classes"][index]), "probability": float(probabilities[index])}
            for index in order[:3]
        ],
        "purpose": "damage evidence classification; not direct fraud detection",
    }


def extract_note_evidence(text: str) -> dict[str, list[str]]:
    normalised = re.sub(r"\s+", " ", text.lower())
    side = [word for word in ("front", "rear") if re.search(rf"\b{word}\b", normalised)]
    damage_terms = {
        "breakage": ("break", "broken", "breakage", "crack", "cracked", "shattered"),
        "crushed": ("crush", "crushed", "crumpled", "caved in"),
        "normal": ("no damage", "undamaged", "normal"),
    }
    damage = [
        label
        for label, terms in damage_terms.items()
        if any(term in normalised for term in terms)
    ]
    return {"vehicle_side": side, "damage_type": damage}


def assess_evidence_consistency(
    image_results: list[dict[str, Any]], claim_note: str, witness_note: str
) -> dict[str, Any]:
    combined_note = " ".join(part for part in (claim_note, witness_note) if part).strip()
    extracted = extract_note_evidence(combined_note) if combined_note else {"vehicle_side": [], "damage_type": []}
    contradictions: list[str] = []
    low_confidence: list[str] = []
    for index, result in enumerate(image_results, start=1):
        if result["confidence"] < 0.55:
            low_confidence.append(f"Image {index} has low classifier confidence ({result['confidence']:.1%}).")
        if result["confidence"] < 0.65:
            continue
        if extracted["vehicle_side"] and result["vehicle_side"] not in extracted["vehicle_side"]:
            contradictions.append(
                f"Image {index} appears {result['vehicle_side']}, while the notes mention {', '.join(extracted['vehicle_side'])}."
            )
        if extracted["damage_type"] and result["damage_type"] not in extracted["damage_type"]:
            contradictions.append(
                f"Image {index} suggests {result['damage_type']}, while the notes mention {', '.join(extracted['damage_type'])}."
            )
    if contradictions:
        status = "contradiction"
    elif image_results and combined_note and (extracted["vehicle_side"] or extracted["damage_type"]):
        status = "consistent"
    else:
        status = "insufficient_for_comparison"
    return {
        "status": status,
        "note_facts": extracted,
        "contradictions": contradictions,
        "warnings": low_confidence,
    }


def fuse_review_route(tabular_decision: str, consistency_status: str) -> tuple[str, str]:
    """Conservative workflow fusion; evidence never reduces a tabular referral."""
    if consistency_status != "contradiction":
        return tabular_decision, "No verified cross-evidence contradiction changed the tabular route."
    if tabular_decision == "approve":
        return "abstain", "A cross-evidence contradiction prevents automatic approval; human review is required."
    return "investigate", "A cross-evidence contradiction escalates an uncertain/high-risk claim for investigation."
