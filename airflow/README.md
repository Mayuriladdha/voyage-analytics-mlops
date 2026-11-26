# Airflow

DAG: `dags/flight_price_pipeline.py` (`flight_price_retraining`, weekly).
Run Airflow with Docker Compose and mount this repo, e.g. add to the `volumes` of the
scheduler/worker services:

    - ./voyage-analytics-mlops:/opt/airflow/project
    - ./voyage-analytics-mlops/airflow/dags:/opt/airflow/dags

and install `requirements-api.txt` + `mlflow` in the Airflow image. The RMSE gate is set with
the `RMSE_THRESHOLD` environment variable.
