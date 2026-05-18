# Architecture Locale

## Vue d'ensemble

Le projet repose sur une architecture volontairement simple:

1. `src/train.py` charge `data/student_dataset.csv`, entraine le pipeline ML et produit les artefacts locaux.
2. `src/predict.py` recharge le modele et centralise la logique de prediction.
3. `src/app.py` expose le modele via FastAPI et publie des metriques Prometheus.
4. `docker-compose.yml` orchestre l'API et Prometheus en local.
5. `.gitlab-ci.yml` automatise lint, tests, entrainement, build Docker et un push Harbor optionnel.

## Flux de donnees

- Dataset CSV local -> preprocessing -> modele RandomForest -> artefacts `artifacts/`
- Artefacts `artifacts/` -> API FastAPI -> predictions HTTP
- API FastAPI -> endpoint `/metrics` -> Prometheus

## Dossiers importants

- `data/` : dataset source
- `artifacts/` : modele entraine et metadonnees
- `mlruns/` : tracking MLflow local facultatif
- `monitoring/` : configuration Prometheus

## Choix techniques

- `scikit-learn` pour un pipeline simple et rapide a tester.
- `FastAPI` pour une exposition REST legere.
- `Prometheus` pour un monitoring facile a brancher localement.
- `Docker Compose` pour garder un deploiement local reproductible.
