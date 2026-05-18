## 🚀 CI/CD Pipeline

[![pipeline status](https://gitlab.com/TON_USER/TON_PROJET/badges/main/pipeline.svg)](https://gitlab.com/TON_USER/TON_PROJET/-/pipelines)

### Variables CI/CD requises (GitLab → Settings → CI/CD → Variables)

| Variable | Description | Masked | Protected |
|----------|-------------|--------|-----------|
| `HARBOR_REGISTRY` | URL du registre Harbor | ✅ | ✅ |
| `HARBOR_IMAGE` | Nom complet de l'image | ✅ | ✅ |
| `HARBOR_USERNAME` | User Harbor | ✅ | ✅ |
| `HARBOR_PASSWORD` | Password Harbor | ✅ | ✅ |
| `SSH_PRIVATE_KEY` | Clé SSH pour déploiement | ✅ | ✅ |
| `SSH_HOST` | Hôte du serveur de déploiement | ❌ | ✅ |
| `SSH_USER` | User SSH | ❌ | ✅ |
| `DEPLOY_PATH` | Chemin de déploiement sur le serveur | ❌ | ✅ |

### Test local rapide

```fish
cd projet
docker compose up --build
curl http://localhost:8000/health