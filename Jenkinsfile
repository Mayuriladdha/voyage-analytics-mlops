pipeline {
  agent any
  environment {
    IMAGE = 'mayuriladdha/voyage-flight-price'
    TAG   = "${env.BUILD_NUMBER}"
    REGISTRY_CREDS = credentials('dockerhub-creds')   // Jenkins > Credentials (username/password)
  }
  stages {
    stage('Checkout') { steps { checkout scm } }

    stage('Install & Test') {
      steps {
        sh '''
          python3 -m venv .venv && . .venv/bin/activate
          pip install -r requirements-api.txt pytest
          python -m pytest -q tests
        '''
      }
    }

    stage('Build Image') {
      steps { sh 'docker build -t $IMAGE:$TAG -t $IMAGE:latest .' }
    }

    stage('Push Image') {
      steps {
        sh '''
          echo $REGISTRY_CREDS_PSW | docker login -u $REGISTRY_CREDS_USR --password-stdin
          docker push $IMAGE:$TAG
          docker push $IMAGE:latest
        '''
      }
    }

    stage('Deploy to Kubernetes') {
      steps {
        sh '''
          kubectl apply -f k8s/deployment.yml -f k8s/service.yml -f k8s/hpa.yml
          kubectl set image deployment/flight-price-api api=$IMAGE:$TAG
          kubectl rollout status deployment/flight-price-api --timeout=120s
        '''
      }
    }
  }
  post {
    success { echo "Deployed $IMAGE:$TAG" }
    failure { echo 'Pipeline failed - check the stage logs above' }
  }
}
