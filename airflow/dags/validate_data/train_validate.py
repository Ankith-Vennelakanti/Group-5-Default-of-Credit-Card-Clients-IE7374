import tensorflow_data_validation as tfdv
from sklearn.model_selection import train_test_split
import pickle
import os
import logging


def train_data_val():

    current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trainpath = os.path.join(current_directory, "data/train_processed_data.pkl")

    trainValidateLog = os.path.join(current_directory, "validate_data/trainValidate.log")


    logging.basicConfig(filename=trainValidateLog, level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.warning("starting new run")

    with open(trainpath, 'rb') as file:
        df = pickle.load(file)

    X_train, X_test = train_test_split(df, test_size=0.25, random_state=42)
    logging.info("train test split done")

    traindata = os.path.join(current_directory, "data/train_val_data.pkl")
    testdata = os.path.join(current_directory, "data/test_val_data.pkl")

    with open(traindata, 'wb') as file:
        pickle.dump(X_train, file)
    with open(testdata, 'wb') as file:
        pickle.dump(X_test, file)

    logging.info("training and validation data dunmped as pkl")

    X_train.reset_index(drop=True, inplace=True)
    stats = tfdv.generate_statistics_from_dataframe(X_train)
    schema = tfdv.infer_schema(stats)
    logging.info("schema infer done from train data")

    # current_directory = os.path.dirname(__file__)
    schemaPath = os.path.join(current_directory, "data/schema.pbtxt")
    tfdv.write_schema_text(schema, schemaPath)
    logging.info("infered schema store for future use")


    X_test.reset_index(drop=True, inplace=True)
    stats_test = tfdv.generate_statistics_from_dataframe(X_test)
    anomalies = tfdv.validate_statistics(stats_test, schema)
    logging.info("generate anomalies from test data")


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
