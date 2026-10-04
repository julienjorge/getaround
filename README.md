# Getaround : délai minimum entre deux locations et prix conseillé

## Le projet

Getaround est une plateforme de location de voitures entre particuliers. Quand un conducteur rend la voiture en retard, le conducteur suivant attend, ou annule. Getaround envisage d'imposer un délai minimum entre deux locations d'une même voiture. Le Product Manager doit en choisir la durée (le seuil) et les voitures concernées (le périmètre : toutes, ou seulement celles équipées du système Connect, qui s'ouvrent avec le smartphone sans rencontrer le propriétaire).

Nous avons construit trois choses :

- **une analyse des retards**, qui chiffre pour chaque seuil et chaque périmètre ce que la mesure coûte (locations refusées, part du revenu) et ce qu'elle rapporte (cas problématiques évités) ;
- **un tableau de bord web**, avec un seuil et un périmètre réglables, pour que le Product Manager compare les options ;
- **une API de prix**, qui suggère un prix de location journalier à partir des caractéristiques d'une voiture, grâce à un modèle de machine learning.

Le tableau de bord et l'API sont en ligne :

- tableau de bord : https://julienvmj-getaround-dashboard.hf.space
- API et sa documentation : https://julienvmj-getaround-api.hf.space/docs

## Les données

Deux fichiers sans lien entre eux, dans data/raw :

- **get_around_delay_analysis.xlsx** : 21 310 locations, avec le type de checkin (mobile ou connect), l'état (terminée ou annulée), le retard à la restitution et, quand la location précédente de la même voiture s'est terminée moins de 12 heures avant, son identifiant et l'écart prévu entre les deux. Base de l'analyse des retards et du tableau de bord.
- **get_around_pricing_project.csv** : 4 843 voitures en location, avec treize caractéristiques (marque, kilométrage, puissance, carburant, couleur, carrosserie, sept options) et leur prix par jour. Base du modèle de prix.

## Comment ça marche

**Analyse des retards** (notebook 01-Getaround_analysis) :

1. Description des données : 16 346 locations au retard connu, 1 841 locations précédées d'une autre dans les 12 heures.
2. Fréquence et ampleur des retards, par type de checkin.
3. Impact sur la location suivante : chaque location est jointe à sa précédente pour savoir si la voiture était revenue à l'heure, et si la location a été annulée.
4. Effet du délai minimum : pour chaque seuil de 0 à 12 heures et chaque périmètre, locations refusées et cas problématiques évités.
5. Part du revenu concernée, puis conclusion.

**Modèle de prix** (notebook 02-Getaround_pricing) : exploration et nettoyage, préparation des données dans un préprocesseur, comparaison d'une régression linéaire et d'une forêt aléatoire, validation croisée, choix du modèle.

**Mise en production** : le script src/train.py entraîne les modèles en les enregistrant dans MLflow (qui garde la trace des paramètres, métriques et modèles de chaque essai) et sauvegarde le modèle retenu dans le dossier de l'API. L'API (FastAPI) charge ce modèle et expose un endpoint /predict et une page /docs. Le tableau de bord (Streamlit) reprend les calculs du notebook. Chaque application a son Dockerfile et tourne sur un Space Hugging Face.

Trois choix à connaître :

- **Un cas problématique** est une location dont la voiture n'était pas revenue à l'heure du checkin. Il est compté résolu si le délai minimum aurait refusé la réservation, et toute réservation refusée est comptée perdue. C'est une borne haute : une partie aurait été reportée.
- **La part du revenu** est calculée sans les prix, absents du fichier : part des locations terminées qui auraient été refusées, en supposant un prix moyen identique pour les locations enchaînées.
- **Le modèle retenu** est une forêt aléatoire à profondeur limitée : presque aussi précise que la forêt par défaut, elle surapprend beaucoup moins et son fichier est plus léger. L'encodage accepte une marque jamais vue à l'entraînement sans erreur.

## Les résultats

**Les retards.** Plus d'un conducteur sur deux rend la voiture en retard (57,5 %), un quart dépasse une heure (26,8 %), un sur six deux heures (15,6 %). Le retard médian est de 9 minutes. Les conducteurs Connect sont plus ponctuels : 16,3 % dépassent une heure, contre 29,6 % en Mobile.

**L'impact sur le conducteur suivant.** Sur les locations enchaînées, 8,7 % des conducteurs suivants ont trouvé la voiture en retard en Connect, 15,9 % en Mobile. Mais en Connect, un tel retard fait doubler le taux d'annulation (de 14,1 % à 27,5 %) ; en Mobile il n'augmente que de 3,6 points (de 8,5 % à 12,1 %).

**L'effet du délai minimum.** Les cas résolus plafonnent vite, les locations refusées continuent de croître :

| Seuil | Périmètre | Locations refusées | Cas résolus | Part du revenu en jeu |
|---|---|---|---|---|
| 1 h | toutes | 401 | 146 sur 216 | 2,0 % |
| 2 h | toutes | 666 | 180 sur 216 | 3,3 % |
| 1 h | Connect | 181 | 48 sur 69 | 0,9 % |
| 2 h | Connect | 295 | 59 sur 69 | 1,4 % |

**Notre recommandation** : un seuil de deux heures, qui résout plus de 80 % des cas problématiques pour au plus 3,3 % du revenu. Sur le périmètre, les deux options se défendent : toutes les voitures pour résoudre le plus de cas, Connect seulement pour cibler les cas les plus graves au moindre coût.

**Le modèle de prix.** En validation croisée, pour un prix médian de 119 € par jour :

| Modèle | R² | Erreur moyenne |
|---|---|---|
| Régression linéaire | 0,703 | 12,3 € |
| Forêt aléatoire par défaut | 0,755 | 10,6 € |
| Forêt aléatoire, profondeur 10 (retenue) | 0,748 | 10,9 € |

## Exécuter le projet

```
01-Getaround_analysis.ipynb   analyse des retards
02-Getaround_pricing.ipynb    choix du modèle de prix
data/raw/                     les deux fichiers fournis
src/train.py                  entraînement, suivi MLflow, sauvegarde du modèle dans api/
api/                          API FastAPI, modèle, Dockerfile
dashboard/                    tableau de bord Streamlit, données, Dockerfile
.env.example                  variables d'environnement attendues
```

**Prérequis** : Python 3.13 avec pandas, plotly, scikit-learn 1.6.1, streamlit, fastapi, mlflow et joblib. Git LFS pour récupérer le modèle api/model.joblib. Docker seulement pour construire les images.

**Configuration** : copier .env.example en .env et renseigner le jeton Hugging Face, utile uniquement pour déployer.

**Lancement**, depuis la racine :

- `python src/train.py` entraîne les modèles et écrit api/model.joblib ; `mlflow ui --port 5001` ouvre le suivi ;
- `cd api && python app.py` lance l'API sur http://localhost:4000 (documentation sur /docs) ;
- `cd dashboard && streamlit run app.py` lance le tableau de bord sur http://localhost:8501.

Test de l'API en ligne :

```
curl -H "Content-Type: application/json" -X POST \
  -d '{"input": [["Citroën", 140411, 100, "diesel", "black", "convertible", true, true, false, false, true, true, true]]}' \
  https://julienvmj-getaround-api.hf.space/predict
```

## Limites et suites possibles

- **Les chiffres du délai minimum sont des bornes hautes** : une réservation refusée est comptée perdue, et 112 des 1 841 paires n'ont pas de retard connu pour la location précédente.
- **Le lien entre retard et annulation est une association**, pas une cause démontrée, sur des effectifs modestes.
- **Le modèle de prix connaît mal les marques rares**, et ne prédit pas au-delà des valeurs vues à l'entraînement.
- **Le suivi MLflow est local** ; en production, le modèle passerait par un registre partagé.
- **Le tableau de bord et l'API ne communiquent pas** : une page du tableau de bord appelant l'API serait l'évolution naturelle.
- **Les Spaces gratuits s'endorment** après une période sans visite et mettent jusqu'à deux minutes à se réveiller.
