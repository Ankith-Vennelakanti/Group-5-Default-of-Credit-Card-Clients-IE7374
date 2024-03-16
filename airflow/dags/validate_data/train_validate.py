import tensorflow_data_validation as tfdv
from sklearn.model_selection import train_test_split
import pickle
import os
import logging

"""
    Split the processed data into training and validation sets, dump them as pickle files,
    infer schema from training data, and validate the statistics of the validation data against the inferred schema.

    This function loads processed data from a pickle file, splits it into training and validation sets,
    dumps them into separate pickle files, infers schema from the training data, writes the inferred schema to a file,
    generates statistics from the validation data, validates the statistics against the inferred schema, and logs any anomalies.

    Returns:
        None
"""


def train_data_val():
    # Get the current directory
    current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # File paths for processed data and log file
    trainpath = os.path.join(current_directory, "data/train_processed_data.pkl")

    trainValidateLog = os.path.join(current_directory, "validate_data/trainValidate.log")

    # Configure logging
    logging.basicConfig(filename=trainValidateLog, level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.warning("starting new run")

    # Load processed data from pickle file
    with open(trainpath, 'rb') as file:
        df = pickle.load(file)

    # Split data into training and validation sets
    X_train, X_test = train_test_split(df, test_size=0.25, random_state=42)
    logging.info("train test split done")

    # Dump training and validation data into pickle files
    traindata = os.path.join(current_directory, "data/train_val_data.pkl")
    testdata = os.path.join(current_directory, "data/test_val_data.pkl")

    with open(traindata, 'wb') as file:
        pickle.dump(X_train, file)
    with open(testdata, 'wb') as file:
        pickle.dump(X_test, file)

    logging.info("training and validation data dunmped as pkl")

    # Reset index for training data and generate statistics
    X_train.reset_index(drop=True, inplace=True)
    stats = tfdv.generate_statistics_from_dataframe(X_train)

    # Infer schema from training data statistics
    schema = tfdv.infer_schema(stats)
    logging.info("schema infer done from train data")

    # current_directory = os.path.dirname(__file__)
    schemaPath = os.path.join(current_directory, "data/schema.pbtxt")

    # Write inferred schema to a file
    tfdv.write_schema_text(schema, schemaPath)
    logging.info("infered schema store for future use")

    # Reset index for validation data and generate statistics
    X_test.reset_index(drop=True, inplace=True)
    stats_test = tfdv.generate_statistics_from_dataframe(X_test)

    # Validate statistics of validation data against the inferred schema
    anomalies = tfdv.validate_statistics(stats_test, schema)
    logging.info("generate anomalies from test data")

    # Check for anomalies and log them
    if anomalies.anomaly_info:
        print("Anomalies detected in the new data:")
        logging.warning("infered schema store for future use")
        for feature_name, anomaly_info in anomalies.anomaly_info.items():
            print(f"Feature: {feature_name}")
            logging.warning(f"Feature: {feature_name}")
            print(f"  Anomaly severity: {anomaly_info.severity}")
            logging.warning(f"  Anomaly severity: {anomaly_info.severity}")
            print(f"  Anomaly short description: {anomaly_info.short_description}")
            logging.warning(f"  Anomaly short description: {anomaly_info.short_description}")
            print("\n")
    else:
        print("No anomalies detected in new data")
        logging.info("No anomalies detected in new data")
        
    
if __name__ == "__main__":
    train_data_val()
