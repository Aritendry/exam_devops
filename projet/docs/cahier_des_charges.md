# Cahier Des Charges

## Objectif

Construire une chaine MLOps locale pour entrainer, servir et monitorer un modele de Machine Learning a partir d'un dataset deja disponible dans le projet.

## Perimetre fonctionnel

- Charger un dataset CSV local.
- Entrainer un modele de regression reproductible.
- Sauvegarder les artefacts du modele localement.
- Exposer une API FastAPI de prediction.
- Monitorer l'API avec Prometheus.
- Conteneuriser l'application avec Docker.
- Preparer une pipeline GitLab CI locale-first.

## Contraintes

- Tout doit fonctionner en local sans dependre d'un serveur distant.
- Le projet doit rester compatible avec une extension future vers GitLab + Harbor.
- Les notebooks ne doivent pas etre necessaires a l'execution normale.

## Livrables

- Code source Python de train et prediction.
- API FastAPI documentee.
- Tests automatises.
- Dockerfile et docker-compose.
- Documentation technique.
