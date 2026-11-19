import math
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


MODEL_PATH = Path(__file__).resolve().with_name("pokemon_type1_extra_trees.joblib")
FEATURE_LABELS = {
    "height_dm": "Height (dm)",
    "weight_hg": "Weight (hg)",
    "base_experience": "Base experience",
    "hp": "HP",
    "attack": "Attack",
    "defense": "Defense",
}

model = joblib.load(MODEL_PATH)
FEATURE_NAMES = tuple(str(name) for name in model.feature_names_in_)
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]
transformed_feature_names = preprocessor.get_feature_names_out()

if len(transformed_feature_names) != len(classifier.feature_importances_):
    raise RuntimeError("Model feature names do not match its feature importances.")

feature_importances = sorted(
    (
        {
            "name": str(name).split("__", 1)[-1],
            "label": FEATURE_LABELS.get(
                str(name).split("__", 1)[-1],
                str(name).split("__", 1)[-1].replace("_", " ").title(),
            ),
            "importance": float(importance),
        }
        for name, importance in zip(
            transformed_feature_names, classifier.feature_importances_
        )
    ),
    key=lambda item: item["importance"],
    reverse=True,
)

analysis = {
    "model_name": type(classifier).__name__,
    "tree_count": classifier.n_estimators,
    "feature_count": len(FEATURE_NAMES),
    "class_count": len(classifier.classes_),
    "feature_importances": feature_importances,
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024


def parse_features(form):
    values = {}
    for feature in FEATURE_NAMES:
        raw_value = form.get(feature, "").strip()
        label = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(f"{label} must be a number.") from error
        if not math.isfinite(value):
            raise ValueError(f"{label} must be a finite number.")
        if value < 0:
            raise ValueError(f"{label} cannot be negative.")
        values[feature] = value
    return values


@app.get("/")
def home():
    return render_template(
        "index.html",
        features=FEATURE_NAMES,
        labels=FEATURE_LABELS,
        values={feature: "" for feature in FEATURE_NAMES},
        analysis=analysis,
    )


@app.post("/predict")
def predict():
    values = {feature: request.form.get(feature, "").strip() for feature in FEATURE_NAMES}
    try:
        parsed_values = parse_features(request.form)
    except ValueError as error:
        return (
            render_template(
                "index.html",
                features=FEATURE_NAMES,
                labels=FEATURE_LABELS,
                values=values,
                analysis=analysis,
                error=str(error),
            ),
            400,
        )

    sample = pd.DataFrame([parsed_values], columns=FEATURE_NAMES)
    predicted_type = str(model.predict(sample)[0])
    class_scores = model.predict_proba(sample)[0]
    ranked_scores = sorted(
        zip(model.classes_, class_scores),
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    results = [
        {"label": str(label).title(), "score": float(score)}
        for label, score in ranked_scores
    ]

    return render_template(
        "index.html",
        features=FEATURE_NAMES,
        labels=FEATURE_LABELS,
        values=values,
        analysis=analysis,
        prediction=predicted_type.title(),
        results=results,
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
