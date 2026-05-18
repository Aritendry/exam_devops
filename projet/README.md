# Projet MLOps Local

Ce projet met en place une mini chaine MLOps 100% locale autour d'un modele de regression qui predit la `Final_Exam_Score` d'un etudiant a partir d'indicateurs d'etude, de sommeil, d'assiduite et de contexte scolaire.

L'implementation suit l'esprit du sujet donne dans `projet_mlops.txt` et des notes de `Devops.txt`, mais reste volontairement locale pour limiter la complexite inutile: dataset local, artefacts locaux, API FastAPI locale, monitoring Prometheus local, Docker Compose local et pipeline GitLab CI compatible avec un futur Harbor optionnel.

## Structure

```text
projet/
├── .env.example
├── .gitlab-ci.yml
├── README.md
├── data/student_dataset.csv
├── docker-compose.yml
├── docker/Dockerfile
├── docs/
├── monitoring/prometheus.yml
├── requirements.txt
├── src/
│   ├── app.py
│   ├── predict.py
│   └── train.py
└── tests/
    ├── test_api.py
    └── test_train.py
```

## Fonctionnalites

- Entrainement local d'un pipeline `scikit-learn` avec preprocessing numerique et categoriel.
- Sauvegarde locale du modele et des metadonnees dans `artifacts/`.
- Tracking MLflow avec UI locale sur port dedie et stockage des runs dans `mlruns/`.
- API FastAPI avec endpoints de sante, info modele, prediction simple, prediction batch et metriques Prometheus.
- Stack Docker Compose locale avec `mlflow`, `trainer`, `api` et `prometheus`.
- CI GitLab avec lint, tests, entrainement, build Docker et push Harbor optionnel.

## Demarrage local

Depuis le dossier `projet/` :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m src.train
uvicorn src.app:app --reload
```

Services locaux :

- API FastAPI: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- MLflow UI: `http://127.0.0.1:5000`

## Lancer avec Docker Compose

```bash
cp .env.example .env
docker compose up --build -d mlflow api prometheus
docker compose --profile training up trainer
```

Services attendus :

- MLflow: `http://127.0.0.1:5000`
- API FastAPI: `http://127.0.0.1:8000`
- Prometheus: `http://127.0.0.1:9090`

## Role des services

- `trainer` entraine le modele et enregistre les runs dans MLflow.
- `mlflow` permet de consulter les experiences, metriques et artefacts.
- `api` FastAPI charge les artefacts du modele pour servir les predictions.
- `prometheus` surveille l'API.

Autrement dit, oui: dans une logique MLOps, on travaille surtout sur l'entrainement et le suivi dans `MLflow`, alors que `FastAPI` sert principalement a exposer le resultat du modele entraine.

## Endpoints principaux

- `GET /health`
- `GET /model/info`
- `POST /predict`
- `POST /predict/batch`
- `GET /metrics`

Exemple de payload pour `POST /predict` :

```json
{
  "Hours_Studied": 22.0,
  "Attendance": 91.0,
  "Sleep_Hours": 7.4,
  "Previous_Scores": 74.0,
  "Tutoring_Sessions": 2,
  "Parental_Involvement": "Medium",
  "Access_to_Resources": "High",
  "Extracurricular_Activities": "Yes",
  "Motivation_Level": "High",
  "Internet_Access": "Yes"
}
```

## Tests

```bash
pytest -q
```

## Remarque projet

Le notebook `eda.ipynb` peut rester utile pour l'exploration, mais toute la chaine principale du projet est codee en Python classique pour rester plus legere, plus testable et plus simple a automatiser dans une logique MLOps.

Si ton environnement utilise le plugin Compose moderne, `docker compose up --build` fonctionne aussi.
