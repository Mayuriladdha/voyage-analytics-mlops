# Voyage Analytics - MLOps Workflow Documentation
*(Copy into a Google Doc. Add a screenshot where marked. Do not include tool installation steps.)*

## 1. REST API (Flask)
What it does, endpoints (`/health`, `/routes`, `/predict`, `/predict_gender`, `/recommend`), request/response flow, input validation.
- 📸 Terminal running `python api/app.py`
- 📸 Postman/curl call to `/predict` with the JSON response
- 📸 A rejected request (HTTP 400) showing validation

## 2. Streamlit App
The three tabs (recommendations, insights, price estimator) and which model each uses.
- 📸 Recommendations tab for one user
- 📸 Insights tab
- 📸 Price estimator result

## 3. Docker Deployment
Why a container (same environment everywhere), what the Dockerfile does line by line (base image, dependencies, code + model copy, gunicorn, healthcheck).
- 📸 `docker build` finishing
- 📸 `docker images` and `docker run`, then `/predict` against the container

## 4. Kubernetes Deployment
Deployment (2 replicas, probes, limits), Service (NodePort), HPA (2-6 pods at 60% CPU).
- 📸 `kubectl get pods,svc,hpa`
- 📸 `/predict` through the service
- 📸 `kubectl describe hpa` or pods scaling under load

## 5. Scheduling in Airflow
DAG graph and why each task exists (validate → train → RMSE gate → promote/skip); schedule `@weekly`.
- 📸 DAG graph view
- 📸 A successful run (grid view) and a task log

## 6. CI/CD Pipeline (Jenkins)
Each stage of the Jenkinsfile and what triggers it.
- 📸 Pipeline stage view, all green
- 📸 Console output of the deploy stage (rollout status)

## 7. MLflow Tracking
Experiment `voyage-flight-price`, what is logged, the comparison of the three models, registered model versions.
- 📸 Runs table with RMSE/MAE/R²
- 📸 Run comparison chart
- 📸 Model registry page (`flight-price-regressor`)

## Results summary
Random Forest: RMSE ≈ 2.38, MAE ≈ 0.23, R² ≈ 0.9999 vs linear baseline RMSE ≈ 103.
