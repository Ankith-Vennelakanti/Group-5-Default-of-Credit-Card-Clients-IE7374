import tensorflow_data_validation as tfdv
import pandas as pd
import os
import pickle



def new_data_val():
    current_directory = os.path.dirname(__file__)
    schemaPath = os.path.join(current_directory, "schema.pbtxt")
    schema = tfdv.load_schema_text(schemaPath)

    with open('new_processed_data.pkl', 'rb') as file:
        df = pickle.load(file)

    schema.default_environment.append('SERVING')
    tfdv.get_feature(schema, 'default payment next month').not_in_environment.append('SERVING')
    
    stats_test = tfdv.generate_statistics_from_dataframe(df)
    anomalies = tfdv.validate_statistics(stats_test, schema,environment='SERVING')
    
    if anomalies.anomaly_info:
        print("Anomalies detected in the new data:")
        for feature_name, anomaly_info in anomalies.anomaly_info.items():
            print(f"Feature: {feature_name}")
            print(f"  Anomaly severity: {anomaly_info.severity}")
            print(f"  Anomaly short description: {anomaly_info.short_description}")
            print("\n")
    else:
        print("No anomalies detected in new data")


new_data_val()