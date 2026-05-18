pipeline {
    agent any

    triggers {
        githubPush()
    }

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    environment {
        PROJECT_DIR = "projet"
        VENV_DIR = "${WORKSPACE}/projet/.venv"
        COMPOSE_FILE = "${WORKSPACE}/projet/docker-compose.yml"
    }

    stages {
        stage("Checkout") {
            steps {
                checkout scm
            }
        }

        stage("Load Secrets") {
            steps {
                script {
                    if (!fileExists(".env")) {
                        error("Le fichier .env est manquant a la racine du depot.")
                    }

                    def dotenv = [:]
                    readFile(".env").split("\n").each { rawLine ->
                        def line = rawLine.trim()
                        if (!line || line.startsWith("#")) {
                            return
                        }

                        int separator = line.indexOf("=")
                        if (separator <= 0) {
                            return
                        }

                        def key = line.substring(0, separator).trim()
                        def value = line.substring(separator + 1).trim()
                        dotenv[key] = value
                    }

                    env.DOCKERHUB_USERNAME = dotenv["DOCKERHUB_USERNAME"] ?: ""
                    env.DOCKERHUB_TOKEN = dotenv["DOCKERHUB_TOKEN"] ?: ""
                    env.DOCKER_IMAGE_NAME = dotenv["DOCKER_IMAGE_NAME"] ?: "student-mlops-api"
                    env.DOCKER_IMAGE_TAG = dotenv["DOCKER_IMAGE_TAG"] ?: "latest"
                    env.GITHUB_BRANCH = dotenv["GITHUB_BRANCH"] ?: "main"
                    env.IMAGE_FULL_NAME = "${env.DOCKERHUB_USERNAME}/${env.DOCKER_IMAGE_NAME}"

                    if (!env.DOCKERHUB_USERNAME?.trim() || !env.DOCKERHUB_TOKEN?.trim()) {
                        error("DOCKERHUB_USERNAME et DOCKERHUB_TOKEN doivent etre definis dans .env.")
                    }
                }
            }
        }

        stage("Prepare Python") {
            steps {
                dir("${PROJECT_DIR}") {
                    sh """
                        python3 -m venv .venv
                        . .venv/bin/activate
                        pip install --upgrade pip
                        pip install -r requirements.txt
                    """
                }
            }
        }

        stage("Lint") {
            steps {
                dir("${PROJECT_DIR}") {
                    sh """
                        . .venv/bin/activate
                        black --check src tests
                        flake8 src tests
                    """
                }
            }
        }

        stage("Tests") {
            steps {
                dir("${PROJECT_DIR}") {
                    sh """
                        . .venv/bin/activate
                        pytest -q tests
                    """
                }
            }
        }

        stage("Train Model") {
            steps {
                dir("${PROJECT_DIR}") {
                    sh """
                        . .venv/bin/activate
                        python -m src.train --disable-mlflow
                    """
                }
            }
        }

        stage("Prepare Compose Env") {
            steps {
                dir("${PROJECT_DIR}") {
                    sh """
                        if [ ! -f .env ]; then
                            cp .env.example .env
                        fi
                    """
                }
            }
        }

        stage("Build Docker Image") {
            steps {
                dir("${PROJECT_DIR}") {
                    sh """
                        docker build -f docker/Dockerfile -t ${IMAGE_FULL_NAME}:${BUILD_NUMBER} .
                        docker tag ${IMAGE_FULL_NAME}:${BUILD_NUMBER} ${IMAGE_FULL_NAME}:${DOCKER_IMAGE_TAG}
                    """
                }
            }
        }

        stage("Login Docker Hub") {
            steps {
                sh '''
                    set +x
                    echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
                '''
            }
        }

        stage("Push Docker Hub") {
            steps {
                sh """
                    docker push ${IMAGE_FULL_NAME}:${BUILD_NUMBER}
                    docker push ${IMAGE_FULL_NAME}:${DOCKER_IMAGE_TAG}
                """
            }
        }

        stage("Deploy Local Stack") {
            steps {
                dir("${PROJECT_DIR}") {
                    sh """
                        docker-compose up -d mlflow
                        docker-compose --profile training up trainer
                        docker-compose up -d api prometheus
                    """
                }
            }
        }
    }

    post {
        always {
            dir("${PROJECT_DIR}") {
                sh """
                    docker-compose ps || true
                    docker logout || true
                """
            }
        }
        success {
            echo "Pipeline Jenkins terminee avec succes."
        }
        failure {
            echo "Pipeline Jenkins echouee. Verifiez les logs des stages."
        }
    }
}
