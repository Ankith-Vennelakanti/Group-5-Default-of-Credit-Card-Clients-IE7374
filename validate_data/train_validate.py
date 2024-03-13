import tensorflow_data_validation as tfdv
from sklearn.model_selection import train_test_split
import pickle
import os


def train_data_val():
    with open('processed_data.pkl', 'rb') as file:
        df = pickle.load(file)



    X_train, X_test = train_test_split(df, test_size=0.25, random_state=42)


    X_train.reset_index(drop=True, inplace=True)
    stats = tfdv.generate_statistics_from_dataframe(X_train)
    schema = tfdv.infer_schema(stats)

    current_directory = os.path.dirname(__file__)
    schemaPath = os.path.join(current_directory, "schema.pbtxt")
    tfdv.write_schema_text(schema, schemaPath)


    X_test.reset_index(drop=True, inplace=True)
    stats_test = tfdv.generate_statistics_from_dataframe(X_test)
    anomalies = tfdv.validate_statistics(stats_test, schema)


    if anomalies.anomaly_info:
        print("Anomalies detected in the new data:")
        for feature_name, anomaly_info in anomalies.anomaly_info.items():
            print(f"Feature: {feature_name}")
            print(f"  Anomaly severity: {anomaly_info.severity}")
            print(f"  Anomaly short description: {anomaly_info.short_description}")
            print("\n")
    else:
        print("No anomalies detected in new data")
        
    
if __name__ == "__main__":
    train_data_val()
