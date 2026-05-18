# Rapport Final

## Resume

Le projet implemente une chaine MLOps locale autour d'un cas de regression sur les performances etudiantes. L'objectif principal etait de disposer d'un flux complet et presentable sans dependre d'une infrastructure distante.

## Realisations

- Entrainement automatise du modele avec preprocessing et sauvegarde d'artefacts.
- API FastAPI de prediction simple et batch.
- Endpoint de sante et endpoint de metriques pour l'observabilite.
- Conteneurisation avec Docker et orchestration locale via Docker Compose.
- Pipeline GitLab CI prete pour lint, tests, train, build et push Harbor optionnel.

## Limites actuelles

- Le deploiement final reste local et non distant par choix de simplification.
- Le suivi MLflow est optionnel et en stockage fichier local.
- Aucun registre Harbor n'est impose pour l'execution courante.

## Evolutions possibles

- Ajouter une vraie phase de validation de donnees.
- Integrer un registre Harbor local dans `docker-compose`.
- Ajouter Grafana au-dessus de Prometheus.
- Versionner les datasets avec DVC ou MLflow artifacts.
