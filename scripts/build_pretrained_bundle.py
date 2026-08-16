from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd
import sklearn
import xgboost


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.history import load_bundled_history
from src.ensemble_models import set_xgboost_device
from src.model import (
    APPEARANCE_TARGETS,
    MINUTES_TARGETS,
    POINT_TARGETS,
    PRETRAINED_MODEL_PATH,
    START_TARGETS,
    _compact_model_bundle,
    _fit,
    _history_key,
    _normalise_history,
    _regressor,
    _target,
    engineer_features,
)


def main() -> None:
    history = _normalise_history(load_bundled_history())
    point_target = _target(history, POINT_TARGETS)
    if point_target is None:
        raise RuntimeError("Bundled history has no point target")

    training = engineer_features(history).reset_index(drop=True)
    training[point_target] = pd.to_numeric(training[point_target], errors="coerce")
    training = training.dropna(subset=[point_target]).reset_index(drop=True)
    bundle = _compact_model_bundle(
        training,
        point_target,
        _target(training, START_TARGETS),
        _target(training, APPEARANCE_TARGETS),
        _target(training, MINUTES_TARGETS),
    )

    price_training = training.copy()
    price_training["price"] = pd.to_numeric(price_training["price"], errors="coerce")
    price_training = price_training.dropna(subset=["price"]).reset_index(drop=True)
    price_model = _fit(
        _regressor(73, max_iter=55),
        price_training,
        price_training["price"],
    )

    # A Colab run may train XGBoost on CUDA. Store CPU inference settings so the
    # resulting artifact remains portable to Streamlit Community Cloud.
    set_xgboost_device(bundle.start_model, "cpu") if bundle.start_model is not None else None
    set_xgboost_device(bundle.appearance_model, "cpu") if bundle.appearance_model is not None else None
    set_xgboost_device(bundle.minutes_model, "cpu") if bundle.minutes_model is not None else None
    for group in (bundle.point_models, bundle.starter_point_models, bundle.substitute_point_models):
        for model in group.values():
            set_xgboost_device(model, "cpu")

    artifact = {
        "schema_version": 3,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "history_key": _history_key(history),
        "bundle": bundle,
        "price_model": price_model,
    }
    PRETRAINED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, PRETRAINED_MODEL_PATH, compress=3)
    print(
        f"Wrote {PRETRAINED_MODEL_PATH} with {bundle.training_rows:,} rows "
        f"using scikit-learn {sklearn.__version__} and XGBoost {xgboost.__version__}"
    )


if __name__ == "__main__":
    main()
