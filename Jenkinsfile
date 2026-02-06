// CI: build, test, Docker image build, registry push (image:prediction-service:git-sha)
pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        stage('Test') {
            steps {
                sh 'pytest tests/ -v'
            }
        }
        stage('Docker Build') {
            steps {
                script {
                    def tag = env.GIT_COMMIT?.take(7) ?: 'latest'
                    sh "docker build -t investment-prediction-service:${tag} ."
                    // push to registry (예: ghcr.io/owner/investment-prediction-service:${tag})
                }
            }
        }
    }
}
