pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'telecom-churn-predictor:latest'
    }

    stages {
        stage('Checkout & Setup') {
            steps {
                echo 'Checking out source repository...'
                checkout scm
                sh 'python3 -m pip install -r requirements.txt'
            }
        }

        stage('Run Unit & Integration Tests') {
            steps {
                echo 'Executing pytest suite...'
                sh 'python3 -m pytest tests/'
            }
        }

        stage('Build Docker Image') {
            steps {
                echo 'Building and training the Docker image...'
                sh 'docker build -t $DOCKER_IMAGE .'
            }
        }
    }

    post {
        success {
            echo 'Telecom Churn Prediction tests and Docker image build completed successfully.'
        }
        failure {
            echo 'CI/CD Pipeline failed. Please check build logs.'
        }
    }
}
