import tensorflow_data_validation as tfdv
import pandas as pd
import os
import pickle
import logging



def new_data_val():
    current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    schemaPath = os.path.join(current_directory, "data/schema.pbtxt")
    schema = tfdv.load_schema_text(schemaPath)
    
    testValidateLog = os.path.join(current_directory, "validate_data/testValidate.log")


    logging.basicConfig(filename=testValidateLog, level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    testpath = os.path.join(current_directory, "data/new_processed_data.pkl")

    with open(testpath, 'rb') as file:
        df = pickle.load(file)

    logging.info("data loaded")

    schema.default_environment.append('SERVING')
    tfdv.get_feature(schema, 'default payment next month').not_in_environment.append('SERVING')
    logging.info("checking for target in new data")
    
    stats_test = tfdv.generate_statistics_from_dataframe(df)
    anomalies = tfdv.validate_statistics(stats_test, schema,environment='SERVING')
    logging.info("generate anomalies from new data")
    
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
    new_data_val()
