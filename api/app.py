from typing import Union

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

description = """
# API de prix Getaround

Cette API suggère un prix journalier de location (en euros) à partir des caractéristiques d'une voiture.
Le modèle est une forêt aléatoire entraînée sur 4 840 voitures déjà en location sur Getaround.

## Endpoints

* `GET /` : message de bienvenue.
* `POST /predict` : prédiction de prix pour une ou plusieurs voitures.

## Format d'entrée du `/predict`

Un JSON avec une clé `input` contenant une liste de voitures, chaque voiture étant une liste de 13 valeurs
dans l'ordre suivant :

1. model_key (texte) : marque, par exemple "Citroën"
2. mileage (entier) : kilométrage
3. engine_power (entier) : puissance en chevaux
4. fuel (texte) : "diesel", "petrol", "hybrid_petrol" ou "electro"
5. paint_color (texte) : couleur, par exemple "black"
6. car_type (texte) : carrosserie, par exemple "sedan"
7. private_parking_available (booléen)
8. has_gps (booléen)
9. has_air_conditioning (booléen)
10. automatic_car (booléen)
11. has_getaround_connect (booléen)
12. has_speed_regulator (booléen)
13. winter_tires (booléen)

Exemple :

```
{"input": [["Citroën", 140411, 100, "diesel", "black", "convertible", true, true, false, false, true, true, true]]}
```

Réponse : un JSON avec une clé `prediction` contenant la liste des prix prédits, dans le même ordre.

```
{"prediction": [116.7]}
```
"""

tags_metadata = [
    {"name": "Introduction", "description": "Point d'entrée de l'API."},
    {"name": "Machine Learning", "description": "Prédiction du prix journalier de location."},
]

app = FastAPI(
    title="Getaround pricing API",
    description=description,
    version="1.0",
    openapi_tags=tags_metadata,
)

COLUMNS = [
    "model_key", "mileage", "engine_power", "fuel", "paint_color", "car_type",
    "private_parking_available", "has_gps", "has_air_conditioning", "automatic_car",
    "has_getaround_connect", "has_speed_regulator", "winter_tires",
]

model = joblib.load("model.joblib")


class PredictionInput(BaseModel):
    input: list[list[Union[str, int, float, bool]]]


class PredictionOutput(BaseModel):
    prediction: list[float]


@app.get("/", tags=["Introduction"])
async def index():
    """Message de bienvenue indiquant où trouver la documentation."""
    return {"message": "Bienvenue sur l'API de prix Getaround. La documentation est disponible sur /docs (Swagger UI) ou sur /redoc (ReDoc)"}


@app.post("/predict", tags=["Machine Learning"], response_model=PredictionOutput)
async def predict(data: PredictionInput):
    """Prédit le prix journalier de location pour chaque voiture de la liste `input`."""
    cars = pd.DataFrame(data.input, columns=COLUMNS)
    prediction = model.predict(cars)
    return {"prediction": prediction.round(1).tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)
