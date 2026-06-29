# Rent Price Explainer with SHAP and LLM

Predicts a fair monthly rent for a property using a machine learning model and explains the prediction in plain language using SHAP feature attributions and a local LLM.

## Features

- Trains a Random Forest regression model for rent prediction.
- Predicts a monthly rent estimate and confidence range.
- Uses SHAP to identify feature contributions.
- Generates a natural-language explanation using Ollama (Llama 3.2).
- Ensures the LLM only explains the SHAP outputs and does not invent feature impacts.

---

## Project Structure

```
rent-price-explainer/
│
├── data/
│   └── rentals.csv
│
├── models/
│
├── train_model.py
├── explain_rent.py
├── sample_property.json
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd rent-price-explainer
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Install Ollama

Download and install Ollama:

https://ollama.com/download

Pull the required model:

```bash
ollama pull llama3.2
```

If the Ollama desktop application is installed, ensure it is running. If using the command-line version, start the Ollama server before executing the application.

---

## Train the Model

Generate the synthetic rental dataset and train the model:

```bash
python train_model.py
```

This creates:

```
data/rentals.csv
models/rent_model.pkl
models/model_columns.pkl
```

---

## Run the Predictor

```bash
python explain_rent.py
```

Example output:

```json
{
  "point_estimate": 3520,
  "range": {
    "low": 3270,
    "high": 3770
  },
  "confidence": "medium",
  "explanation": "...",
  "drivers": [
    {
      "feature": "in_unit_laundry",
      "impact_usd": 69,
      "direction": "up"
    }
  ]
}
```

---

## How It Works

1. A Random Forest regression model predicts monthly rent.
2. SHAP computes feature-level contribution values.
3. These SHAP values are passed to a local LLM (Llama 3.2 via Ollama).
4. The LLM converts the feature impacts into a concise natural-language explanation without modifying the underlying SHAP values.

---

## Technologies

- Python
- pandas
- scikit-learn
- SHAP
- Ollama
- Llama 3.2
- joblib

---
