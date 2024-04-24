import numpy as np
import pandas as pd
import os
import mlflow
import mlflow.lightgbm
import pickle
import lightgbm as lgb
import mlflow.sklearn
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from mlflow.models import infer_signature
from urllib.parse import urlparse
import logging

"""
    train_model() Trains a LightGBM model using data stored in pickle files, log metrics and model to MLflow, and save the run ID.

    This function loads training and test data from pickle files, scales the features using StandardScaler,
    trains a LightGBM model, logs relevant metrics and the model to MLflow, and saves the run ID to a pickle file.

    Note: This function assumes MLflow is properly configured and running.
"""

def train_model():
    rand_seed = 1234
    np.random.seed(rand_seed)

    # set paths to dump and load data in pickle formats
    current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    traindata = os.path.join(current_directory, "data/train_val_data.pkl")
    testdata = os.path.join(current_directory, "data/test_val_data.pkl")
    scaledpkl = os.path.join(current_directory, "data/scaler.pkl")
    runidpkl = os.path.join(current_directory, "data/runidpkl.pkl")
    trainingLog = os.path.join(current_directory, "ml/training.log")

    # initialize logging
    logging.basicConfig(filename=trainingLog, level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.warning("Trainging new model")

    # load train data
    with open(traindata, 'rb') as file:
        train = pickle.load(file)

    # load test data
    with open(testdata, 'rb') as file:
        test = pickle.load(file)

    logging.info("data loaded")

    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]

    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]

    # initalize standardscaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    # print(X_train.head(2))

    for col in X_train.columns:
        print(col)

    # Dump scaler .fit() for future use into prediction pipeline
    with open(scaledpkl, 'wb') as file:
        train = pickle.dump(scaler, file)

    # Scale training and testing data
    scaled_train = scaler.transform(X_train)
    scaled_test = scaler.transform(X_test)

    logging.info("data scaled")

    # Define params for LightGBM model
    # lgb_params = {
    #     "objective": "binary",
    #     "metric": "auc",
    #     "learning_rate": 0.03,
    #     "num_leaves": 10,
    #     "max_depth": 3,
    #     "random_state": rand_seed,
    #     "n_jobs": 1,
    # }

    # lgb_params = {'learning_rate': [0.01, 0.001,0.05,0.5],
    #             'max_depth': [2, 3, 4, 5, 8, 10]}
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.01,
        "num_leaves": 8,
        "max_depth": 6,
        "random_state": rand_seed,
        "n_jobs": 1,
    }

    logging.info(f"params: {lgb_params}")
    
    

    # start mlflow run
    with mlflow.start_run() as run:
        # initialize model with above params
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

        # calculate and log Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Accurary: {accuracy}")
        print(accuracy)

        signature = infer_signature(scaled_train, y_pred)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Log model
        mlflow.sklearn.log_model(model, "model", signature=signature)
        logging.info(f"model logged")

        run_id = run.info.run_id

        # print run id of the mlflow run
        print(run_id)
        logging.info(f"run_id for latest run {run_id}")

        # dump run id for loading the model in prediction pipeline
        with open(runidpkl, 'wb') as file:
            pickle.dump(run_id, file)


if __name__ == "__main__":
    train_model()
