import numpy as np
import pandas as pd
import os
import mlflow
import mlflow.lightgbm
import pickle
import lightgbm as lgb
import warnings
import mlflow.sklearn
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from mlflow.models import infer_signature
from urllib.parse import urlparse
import logging

def train_model():
    rand_seed = 1234
    np.random.seed(rand_seed)


    current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    traindata = os.path.join(current_directory, "data/train_val_data.pkl")
    testdata = os.path.join(current_directory, "data/test_val_data.pkl")
    scaledpkl = os.path.join(current_directory, "data/scaler.pkl")
    runidpkl = os.path.join(current_directory, "data/runidpkl.pkl")
    trainingLog = os.path.join(current_directory, "ml/training.log")


    logging.basicConfig(filename=trainingLog, level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.warning("Trainging new model")

    with open(traindata, 'rb') as file:
        train = pickle.load(file)

    with open(testdata, 'rb') as file:
        test = pickle.load(file)

    logging.info("data loaded")

    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]

    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]

    scaler = StandardScaler()
    scaler.fit(X_train)

    # print(X_train.head(2))

    for col in X_train.columns:
        print(col)

    with open(scaledpkl, 'wb') as file:
        train = pickle.dump(scaler, file)

    scaled_train = scaler.transform(X_train)
    scaled_test = scaler.transform(X_test)

    logging.info("data scaled")


    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 10,
        "max_depth": 3,
        "random_state": rand_seed,
        "n_jobs": 1,
    }

    logging.info(f"params: {lgb_params}")
    
    


    with mlflow.start_run() as run:
        
        model = lgb.LGBMClassifier(**lgb_params)
        # Train the model
        model.fit(scaled_train, y_train)
        print("model fit done")
        # Log parameters
        mlflow.log_params(lgb_params)

        # Make predictions
        y_pred = model.predict(scaled_test)

        logging.info("model pridiction done with scaled test data")
        print("model predicted")

        # Log metrics
        auc = roc_auc_score(y_test, y_pred)
        logging.info(f"AUC: {auc}")
        print(auc)
        mlflow.log_metric("auc", roc_auc_score(y_test, y_pred))

        mlflow.set_tracking_uri("http://127.0.0.1:5001") 

        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Accurary: {accuracy}")
        print(accuracy)

        signature = infer_signature(scaled_train, y_pred)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        mlflow.sklearn.log_model(model, "model", signature=signature)
        logging.info(f"model logged")

        run_id = run.info.run_id

        print(run_id)
        logging.info(f"run_id for latest run {run_id}")

        
        with open(runidpkl, 'wb') as file:
            pickle.dump(run_id, file)


if __name__ == "__main__":
    train_model()
