import pandas as pd
import pickle
import os
import mlflow
import logging
from datetime import datetime

"""
    Predict using a pre-trained model and scaler on new processed data and log the predictions.

    This function loads a pre-trained model and scaler from MLflow, scales the new processed data,
    makes predictions using the model, logs the predictions, and saves the predicted data to a CSV file.

    Returns:
        None
"""

def predict_data():
    current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PredictionLog = os.path.join(current_directory, "ml/Prediction.log")
    runidpkl = os.path.join(current_directory, "data/runidpkl.pkl")


    mlflow.set_tracking_uri("http://127.0.0.1:5001")

    logging.basicConfig(filename=PredictionLog, level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.warning("starting a new run")
    
    # get run_id of the ml run from mlflow
    with open(runidpkl, 'rb') as file:
        run_id = pickle.load(file)

    # Load Model using the latest run id
    model_path = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_path) # Load model

    logging.info("model loaded!")

    # read processed data
    testpath = os.path.join(current_directory, "data/new_processed_data.pkl")
    with open(testpath, 'rb') as file:
        df = pickle.load(file)

    # load scaler
    scaledpkl = os.path.join(current_directory, "data/scaler.pkl")
    with open(scaledpkl, 'rb') as file:
        scaler = pickle.load(file)
    logging.info("scaler loaded!")

    # scale data
    scaled_data = scaler.transform(df)
    logging.info("new data scaled")

    # make predictions
    predictions = model.predict(scaled_data)
    logging.info("model predicted")

    # generate a csv file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    predictionFile = f"predicted/predicted_data_{timestamp}.csv"
    Predicted_file = os.path.join(current_directory, predictionFile)

    # save the predictions csv file
    df['preds'] = predictions
    df.to_csv(Predicted_file, index=False)

    
    

if __name__ == "__main__":
    predict_data()