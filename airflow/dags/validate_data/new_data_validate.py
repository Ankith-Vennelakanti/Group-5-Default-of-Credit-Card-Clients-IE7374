import tensorflow_data_validation as tfdv
import pandas as pd
import os
import pickle
import logging

"""
    Validate newly processed data against a predefined schema and log any anomalies.

    This function loads a predefined schema from a file, verifies the non-presence of the target feature in the new data,
    generates statistical summaries from the new data, validates these statistics against the predefined schema,
    and logs any detected anomalies.

    Returns:
        None
"""


def new_data_val():
    # Obtain the path of the parent directory of the current script
    current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construct the path to the schema file
    schemaPath = os.path.join(current_directory, "data/schema.pbtxt")
    
    # Load the schema from the specified path
    schema = tfdv.load_schema_text(schemaPath)
    
    # Set up the logging file path
    testValidateLog = os.path.join(current_directory, "validate_data/testValidate.log")

    # Configure logging
    logging.basicConfig(filename=testValidateLog, level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Define the path to the new processed data
    testpath = os.path.join(current_directory, "data/new_processed_data.pkl")

    # Load the new processed data from the pickle file
    with open(testpath, 'rb') as file:
        df = pickle.load(file)

    # Log the completion of data loading
    logging.info("data loaded")

    # Configure the schema for the serving environment
    schema.default_environment.append('SERVING')
    
    # Ensure the target feature is not used during serving
    tfdv.get_feature(schema, 'default payment next month').not_in_environment.append('SERVING')
    
    # Log the check for the target feature in new data
    logging.info("checking for target in new data")
    
    df2= df.head(5)
    print(df2)
    # Generate statistics from the new data
    stats_test = tfdv.generate_statistics_from_dataframe(df)
    
    # Validate the generated statistics against the schema
    anomalies = tfdv.validate_statistics(stats_test, schema, environment='SERVING')
    
    # Log the generation of anomalies
    logging.info("generate anomalies from new data")
    anomalies_detected = False
    # Check if any anomalies were detected
    if anomalies.anomaly_info:
        print("Anomalies detected in the new data:")
        anomalies_detected = True
        # Log a message about storing the inferred schema
        logging.warning("inferred schema stored for future use")
        
        # Iterate over each anomaly and log the details
        for feature_name, anomaly_info in anomalies.anomaly_info.items():
            print(f"Feature: {feature_name}")
            logging.warning(f"Feature: {feature_name}")
            print(f"  Anomaly severity: {anomaly_info.severity}")
            logging.warning(f"  Anomaly severity: {anomaly_info.severity}")
            print(f"  Anomaly short description: {anomaly_info.short_description}")
            logging.warning(f"  Anomaly short description: {anomaly_info.short_description}")
            print("\n")
    else:
        # Log and print the absence of anomalies
        print("No anomalies detected in new data")
        logging.info("No anomalies detected in new data")
    
    return anomalies_detected

if __name__ == "__main__":
    new_data_val()
