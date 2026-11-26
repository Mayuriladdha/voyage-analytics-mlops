# MLflow tracking

Tracking code lives in `src/train_regression.py` (experiment `voyage-flight-price`, registered
model `flight-price-regressor`). To browse runs:

    mlflow ui --backend-store-uri ./mlruns          # file store
    # or, with the model registry:
    export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
    python src/train_regression.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db

Each of the 3 candidate models logs params, RMSE/MAE/R2, training time and the fitted pipeline;
the winner is logged again as `champion_<model>` and registered, giving a new model version
on every retraining run.
