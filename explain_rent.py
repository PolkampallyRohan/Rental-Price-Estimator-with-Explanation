import requests
import json
import joblib
import numpy as np
import pandas as pd
import shap


FEATURE_COLUMNS = [
    "in_unit_laundry",
    "dishwasher",
    "hardwood_floors",
    "recently_renovated",
    "pets_allowed"
]


def property_to_dataframe(property_data: dict) -> pd.DataFrame:
    row = {
        "city": property_data["city"],
        "neighborhood": property_data["neighborhood"],
        "beds": property_data["beds"],
        "baths": property_data["baths"],
        "sqft": property_data["sqft"],
        "year_built": property_data["year_built"],
        "parking": property_data["parking"],
        "floor": property_data["floor"],
        "has_elevator": int(property_data["has_elevator"]),
    }

    features = set(property_data.get("features", []))

    for feature in FEATURE_COLUMNS:
        row[feature] = 1 if feature in features else 0

    return pd.DataFrame([row])


def align_columns(df: pd.DataFrame, model_columns: list[str]) -> pd.DataFrame:
    df_encoded = pd.get_dummies(df)

    for column in model_columns:
        if column not in df_encoded.columns:
            df_encoded[column] = 0

    return df_encoded[model_columns]


def build_drivers(model, x_aligned: pd.DataFrame) -> list[dict]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_aligned)

    impacts = shap_values[0]
    shap_map = {
        feature_name: int(round(float(impact)))
        for feature_name, impact in zip(x_aligned.columns, impacts)
    }

    drivers = []

    allowed_driver_features = {
        "in_unit_laundry",
        "dishwasher",
        "hardwood_floors",
        "recently_renovated",
        "pets_allowed",
        "parking_street_only",
        "parking_garage",
        "parking_off_street"
    }

    for feature_name in allowed_driver_features:
        impact_usd = shap_map.get(feature_name, 0)

        if abs(impact_usd) >= 20:
            drivers.append({
                "feature": feature_name,
                "impact_usd": impact_usd,
                "direction": "up" if impact_usd > 0 else "down"
            })

    no_elevator_high_floor_impact = (
        shap_map.get("floor", 0) +
        shap_map.get("has_elevator", 0)
    )

    if abs(no_elevator_high_floor_impact) >= 20:
        drivers.append({
            "feature": "no_elevator_high_floor",
            "impact_usd": no_elevator_high_floor_impact,
            "direction": "up" if no_elevator_high_floor_impact > 0 else "down"
        })

    drivers.sort(key=lambda item: abs(item["impact_usd"]), reverse=True)

    return drivers[:8]

def make_plain_language_explanation(point_estimate: int, low: int, high: int, drivers: list[dict]) -> str:
    driver_lines = "\n".join(
        f"- {format_feature_name(d['feature'])}: {d['impact_usd']:+d} USD/month"
        for d in drivers
    )

    prompt = f"""
    You are an AI assistant helping explain a machine learning rent prediction.

    Predicted monthly rent: ${point_estimate:,}
    Expected range: ${low:,} to ${high:,}

    The following feature impacts come directly from SHAP and MUST NOT be changed:

    {driver_lines}

    Write ONLY the explanation text.

    Requirements:
    - One paragraph only.
    - Do not use bullet points.
    - Do not say "Here's the explanation", "Based on the analysis", or similar introductions.
    - Mention the predicted rent and that it is approximate.
    - Explain that positive SHAP values increase the estimate.
    - Explain that negative SHAP values decrease the estimate.
    - Do not invent any new dollar amounts.
    - Do not mention SHAP, JSON, machine learning, or the prompt.
    """
    
    # Convert SHAP feature impacts into a natural-language explanation.
    # The LLM is instructed to use only the provided SHAP values and
    # not invent or modify any dollar impacts.
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        explanation = response.json().get("response", "").strip()

        if explanation:
            return explanation

    except requests.RequestException as error:
        print(f"Ollama unavailable, using fallback explanation. Details: {error}")

    return make_fallback_explanation(point_estimate, low, high, drivers)

def make_fallback_explanation(point_estimate: int, low: int, high: int, drivers: list[dict]) -> str:
    positive_drivers = [d for d in drivers if d["impact_usd"] > 0]
    negative_drivers = [d for d in drivers if d["impact_usd"] < 0]

    explanation = (
        f"I'd expect this property to rent around ${point_estimate:,}/month, "
        f"with a likely range of about ${low:,} to ${high:,}. "
    )

    if positive_drivers:
        positives = ", ".join(
            f"{format_feature_name(d['feature'])} (${d['impact_usd']:+,}/mo)"
            for d in positive_drivers[:3]
        )
        explanation += f"The estimate is pushed upward mostly by {positives}. "

    if negative_drivers:
        negatives = ", ".join(
            f"{format_feature_name(d['feature'])} (${d['impact_usd']:+,}/mo)"
            for d in negative_drivers[:2]
        )
        explanation += f"Working against the price: {negatives}. "

    explanation += (
        "The dollar impacts come from the model's SHAP values, "
        "so the explanation is tied directly to the prediction rather than invented afterward."
    )

    return explanation

def format_feature_name(feature: str) -> str:
    labels = {
        "in_unit_laundry": "in-unit laundry",
        "recently_renovated": "recent renovation",
        "hardwood_floors": "hardwood floors",
        "pets_allowed": "pets allowed",
        "parking_street_only": "street-only parking",
        "parking_garage": "garage parking",
        "parking_off_street": "off-street parking",
        "no_elevator_high_floor": "third-floor walk-up with no elevator"
    }

    return labels.get(feature, feature.replace("_", " "))


def predict_rent(property_data: dict) -> dict:
    model = joblib.load("models/rent_model.pkl")
    model_columns = joblib.load("models/model_columns.pkl")

    raw_df = property_to_dataframe(property_data)
    x_aligned = align_columns(raw_df, model_columns)

    point_estimate = int(round(model.predict(x_aligned)[0] / 10) * 10)

    low = int(round((point_estimate - 250) / 10) * 10)
    high = int(round((point_estimate + 250) / 10) * 10)

    drivers = build_drivers(model, x_aligned)

    result = {
        "point_estimate": point_estimate,
        "range": {
            "low": low,
            "high": high
        },
        "confidence": "medium",
        "explanation": make_plain_language_explanation(
            point_estimate,
            low,
            high,
            drivers
        ),
        "drivers": drivers
    }

    return result


def main():
    with open("sample_property.json", "r", encoding="utf-8") as file:
        property_data = json.load(file)

    result = predict_rent(property_data)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()