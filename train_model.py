import random
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

random.seed(42)

cities = {
    "Cambridge": {
        "neighborhoods": {
            "Inman Square": 3150,
            "Harvard Square": 3500,
            "Kendall Square": 3800,
            "Central Square": 3250
        }
    }
}

rows = []

for _ in range(5000):
    city = "Cambridge"
    neighborhood = random.choice(list(cities[city]["neighborhoods"].keys()))

    beds = random.randint(1, 4)
    baths = random.choice([1, 1, 2, 2, 3])
    sqft = random.randint(500, 1800)
    year_built = random.randint(1890, 2022)

    in_unit_laundry = random.choices([0, 1], weights=[65, 35])[0]
    dishwasher = random.choices([0, 1], weights=[55, 45])[0]
    hardwood_floors = random.choices([0, 1], weights=[50, 50])[0]
    recently_renovated = random.choices([0, 1], weights=[75, 25])[0]
    pets_allowed = random.choices([0, 1], weights=[60, 40])[0]

    parking = random.choice(["street_only", "garage", "off_street"])

    floor = random.randint(1, 5)
    has_elevator = random.randint(0, 1)

    rent = cities[city]["neighborhoods"][neighborhood]

    rent += (beds - 2) * 350
    rent += (baths - 1) * 175
    rent += (sqft - 900) * 1.35

    if year_built > 2000:
        rent += 120

    if in_unit_laundry:
        rent += 180

    if dishwasher:
        rent += 75

    if hardwood_floors:
        rent += 55

    if recently_renovated:
        rent += 160

    if pets_allowed:
        rent += 45

    if parking == "garage":
        rent += 175
    elif parking == "off_street":
        rent += 90

    if floor >= 3 and not has_elevator:
        rent -= 45

    rent += random.randint(-150, 150)

    rows.append({
        "city": city,
        "neighborhood": neighborhood,
        "beds": beds,
        "baths": baths,
        "sqft": sqft,
        "year_built": year_built,
        "in_unit_laundry": in_unit_laundry,
        "dishwasher": dishwasher,
        "hardwood_floors": hardwood_floors,
        "recently_renovated": recently_renovated,
        "pets_allowed": pets_allowed,
        "parking": parking,
        "floor": floor,
        "has_elevator": has_elevator,
        "rent": rent
    })

df = pd.DataFrame(rows)
df.to_csv("data/rentals.csv", index=False)

X = df.drop(columns=["rent"])
y = df["rent"]

X = pd.get_dummies(X)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "models/rent_model.pkl")
joblib.dump(X.columns.tolist(), "models/model_columns.pkl")

print("Model trained successfully.")
print(f"Training samples: {len(df)}")