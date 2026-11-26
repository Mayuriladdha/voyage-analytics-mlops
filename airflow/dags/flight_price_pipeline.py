"""Airflow DAG: weekly retraining of the flight price model.

validate_data -> train_and_track -> quality_gate -> (promote_model | skip_promotion)

Mount the repo at /opt/airflow/project (see airflow/README.md) so the DAG can import src/.
"""
import json
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

PROJECT = os.getenv("PROJECT_DIR", "/opt/airflow/project")
sys.path.insert(0, os.path.join(PROJECT, "src"))
os.environ.setdefault("DATA_DIR", os.path.join(PROJECT, "data"))
os.environ.setdefault("MODEL_DIR", os.path.join(PROJECT, "models"))

RMSE_THRESHOLD = float(os.getenv("RMSE_THRESHOLD", 10.0))   # promote only if RMSE is below this


def validate_data():
    import pandas as pd
    df = pd.read_csv(os.path.join(os.environ["DATA_DIR"], "flights.csv"))
    required = {"from", "to", "flightType", "price", "distance", "agency", "date"}
    assert required <= set(df.columns), f"missing columns: {required - set(df.columns)}"
    assert len(df) > 1000, "too few rows"
    assert df["price"].notna().all() and (df["price"] > 0).all(), "bad prices"


def train_and_track(ti):
    from train_regression import main
    best, results = main()                       # logs every run to MLflow
    ti.xcom_push(key="best_rmse", value=results[best]["rmse"])
    ti.xcom_push(key="best_model", value=best)


def quality_gate(ti):
    rmse = ti.xcom_pull(task_ids="train_and_track", key="best_rmse")
    return "promote_model" if rmse < RMSE_THRESHOLD else "skip_promotion"


def promote_model(ti):
    # The new joblib is already written to MODEL_DIR; a real setup would now trigger the
    # Jenkins job (webhook) so the image is rebuilt and rolled out to Kubernetes.
    print("Promoted:", ti.xcom_pull(task_ids="train_and_track", key="best_model"))


with DAG(
    dag_id="flight_price_retraining",
    description="Validate data, retrain, track in MLflow, gate on RMSE",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    default_args={"owner": "mayuri", "retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["voyage", "mlops"],
) as dag:
    t1 = PythonOperator(task_id="validate_data", python_callable=validate_data)
    t2 = PythonOperator(task_id="train_and_track", python_callable=train_and_track)
    t3 = BranchPythonOperator(task_id="quality_gate", python_callable=quality_gate)
    t4 = PythonOperator(task_id="promote_model", python_callable=promote_model)
    t5 = EmptyOperator(task_id="skip_promotion")
    t1 >> t2 >> t3 >> [t4, t5]
