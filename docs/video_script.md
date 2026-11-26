# Video script - regression model only (target ≈ 15 min; extend each part if your course requires 40 min)

**1. Introduction (1.5 min)** I built Voyage Analytics, an end-to-end ML system for travel and tourism. It predicts flight prices (the focus of this video), classifies user gender and recommends hotels. Stack: Python, Flask, Docker, Kubernetes, Airflow, Jenkins, MLflow, Streamlit. Show the repo layout.

**2. Problem understanding (2 min)** A platform must quote prices instantly and keep the model fresh. A notebook cannot serve real-time requests, scale with traffic, retrain automatically or tell you which model version is live. That is why the model becomes a deployable system.

**3. Data understanding (2 min)** Users (1,340), flights (≈272k), hotels (≈40.5k). Join keys: `userCode` and `travelCode`. Flights → regression; users + aggregates → classification; hotels → recommender. Mention the EDA findings: flight type and distance drive price, FlyingDrops sells only first class, `time` duplicates `distance`.

**4. Model development and serving (2.5 min)** Features and the sklearn Pipeline (one-hot + Random Forest). Compare linear (RMSE ≈ 103), boosting (≈ 16) and Random Forest (≈ 2.4). Be upfront that the near-perfect R² reflects the dataset. Then the Flask API: request JSON → validation → route lookup for distance → date features → pipeline → price. Demo a good and a bad request.

**5. MLOps pipeline (2 min)** Docker packages code + model. Kubernetes runs 2 replicas, probes and an autoscaler. Jenkins: test → build → push → rollout. Airflow: weekly validate → train → RMSE gate → promote. MLflow: params, metrics and registered versions per run.

**6. App interface and monitoring (2 min)** Streamlit tabs. Briefly how the recommender works. Monitoring ideas: track prediction error against actual fares, input drift, API latency and errors.

**7. Challenges and reliability (1.5 min)** Multiple datasets and join logic, keeping training and serving preprocessing identical (one Pipeline), choosing honest evaluation for the recommender, deployment complexity. Reliability: Git, repeatable pipeline, versioned models.

**8. Learnings and future work (1.5 min)** Training a model ≠ running a system. Future: drift monitoring, automated retraining triggers, API authentication, richer features, cloud deployment.

## Likely follow-up questions
- *Productionise real-time pricing?* API behind a load balancer, model in a container, autoscaling, monitoring, versioned rollouts.
- *Why Docker and Kubernetes?* Same environment everywhere; Kubernetes adds replicas, self-healing, autoscaling and rolling updates.
- *How does MLflow help?* Every run's params/metrics/model are stored and the registry gives numbered versions to promote or roll back.
- *Retraining in production?* Scheduled/triggered Airflow DAG, validation, RMSE gate, registry, Jenkins rollout.
- *Scale for a large platform?* More replicas/nodes, caching of routes, batch endpoint, a feature store, cloud-managed Kubernetes.
