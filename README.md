# Voyage Analytics: Integrating MLOps in Travel

End-to-end MLOps capstone (AlmaBetter) by **Mayuri Laddha**. Three travel datasets (users, flights, hotels) feed three models; the **flight price regression** model is productionised with Flask, Docker, Kubernetes, Airflow, Jenkins and MLflow.

| Model | Method | Test result |
|---|---|---|
| Flight price (regression) | Random Forest in a sklearn Pipeline | RMSE ≈ 2.38, MAE ≈ 0.23, R² ≈ 0.9999 (linear baseline RMSE ≈ 103) |
| Gender (classification) | Logistic Regression / Random Forest | ≈ 33% accuracy = chance level (no signal in data, see notebook) |
| Hotel recommendation | Item-based collaborative filtering | hit@3 ≈ 0.61 vs random ≈ 0.53, popularity ≈ 0.63 |

> The price model's near-perfect score reflects how this (synthetic-looking) dataset is built, not what real fares would give. The notebook discusses this.

## Repository layout
```
data/               users.csv, flights.csv, hotels.csv
notebooks/          Voyage_Analytics_Capstone.ipynb  (Colab notebook, project template format)
src/                common.py, train_regression.py, train_classification.py, train_recommender.py
models/             flight_price_model.joblib, gender_model.joblib, recommender.joblib, routes.json, *_metrics.json
api/app.py          Flask REST API
streamlit_app/      Streamlit app (recommendations, insights, price estimator)
Dockerfile          container image for the API
k8s/                deployment.yml, service.yml, hpa.yml
airflow/dags/       flight_price_pipeline.py (weekly retraining DAG)
Jenkinsfile         CI/CD: test -> build -> push -> deploy to Kubernetes
mlflow_tracking/    how experiments/registry are used (code is in src/train_regression.py)
tests/              API tests
docs/               documentation template and video script
```

## Quick start
```bash
pip install -r requirements.txt
export PYTHONPATH=src
python src/train_regression.py          # trains 3 models, logs to MLflow, writes models/*.joblib
python src/train_classification.py
python src/train_recommender.py
python api/app.py                       # API on http://localhost:5000
streamlit run streamlit_app/app.py      # app on http://localhost:8501
python -m pytest -q tests
```

## REST API
| Endpoint | Description |
|---|---|
| `GET /health` | probe used by Kubernetes |
| `GET /routes` | valid origin → destination pairs |
| `POST /predict` | flight price; one JSON object or a list |
| `POST /predict_gender` | gender class + probabilities (`age`, `company` required) |
| `GET /recommend?userCode=0&k=3` | hotel suggestions |

```bash
curl -X POST localhost:5000/predict -H "Content-Type: application/json" \
  -d '{"from":"Recife (PE)","to":"Florianopolis (SC)","flightType":"firstClass","agency":"FlyingDrops","date":"2023-05-17"}'
```
The distance is looked up from the route, so callers never send it. Invalid input returns HTTP 400 with a message.

## Docker and Kubernetes
```bash
docker build -t mayuriladdha/voyage-flight-price:latest .
docker run -p 5000:5000 mayuriladdha/voyage-flight-price:latest
kubectl apply -f k8s/
kubectl get pods,svc,hpa
```
Two replicas with readiness/liveness probes on `/health`, resource limits, and a CPU-based HorizontalPodAutoscaler (2-6 pods at 60% CPU).

## Airflow
`flight_price_retraining` runs weekly: `validate_data → train_and_track → quality_gate → promote_model | skip_promotion`. The gate promotes only if test RMSE is below `RMSE_THRESHOLD` (default 10). See `airflow/README.md`.

## Jenkins (CI/CD)
`Jenkinsfile` stages: Checkout → Install & Test → Build Image → Push Image (credentials id `dockerhub-creds`) → Deploy to Kubernetes (`kubectl set image` + rollout status). Update `IMAGE` in the Jenkinsfile and `k8s/deployment.yml` if your Docker Hub name differs.

## MLflow
Experiment `voyage-flight-price`; every candidate model logs params, RMSE/MAE/R², training time and the pipeline; the winner is registered as `flight-price-regressor` (a new version per retraining). See `mlflow_tracking/README.md`.

## Limitations and next steps
Gender prediction is at chance level and the recommender only matches a popularity baseline on this data; the API has no authentication; there is no drift monitoring yet. Next: monitoring and drift alerts, auth, richer features, cloud deployment.
