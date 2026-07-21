from pathlib import Path

from fairclaim.config import IMAGE_MODEL_BUNDLE_PATH
from fairclaim.evidence import (
    analyse_image,
    assess_evidence_consistency,
    extract_note_evidence,
    fuse_review_route,
    load_image_bundle,
)


def test_image_model_and_note_extraction():
    bundle = load_image_bundle(IMAGE_MODEL_BUNDLE_PATH)
    root = Path(__file__).resolve().parents[1]
    image_path = next((root / "datasets" / "comprehensive-car-damage" / "F_Normal").glob("*"))
    result = analyse_image(image_path.read_bytes(), bundle)
    assert result["label"] in bundle["classes"]
    assert 0 <= result["confidence"] <= 1
    assert extract_note_evidence("The front bumper was crushed")["damage_type"] == ["crushed"]


def test_contradiction_only_escalates_route():
    images = [{"confidence": 0.9, "vehicle_side": "rear", "damage_type": "breakage"}]
    consistency = assess_evidence_consistency(images, "The front was crushed", "")
    assert consistency["status"] == "contradiction"
    assert fuse_review_route("approve", "contradiction")[0] == "abstain"
    assert fuse_review_route("investigate", "consistent")[0] == "investigate"
