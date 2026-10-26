from pathlib import Path
import importlib.util


def load_training_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "train_models.py"
    spec = importlib.util.spec_from_file_location("train_models_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_model_family_comparison_has_no_hidden_class_weighting():
    module = load_training_module()
    candidates = module.candidates()
    assert candidates["logistic_regression"].class_weight is None
    assert candidates["random_forest"].class_weight is None
    assert candidates["extra_trees"].class_weight is None
    assert candidates["hist_gradient_boosting"].class_weight is None
