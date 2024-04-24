import pandas as pd
import os
import pickle
import logging
from openpyxl.workbook import Workbook
from airflow.api.client.local_client import Client

def retrain_data(anomalies_detected):
    print(anomalies_detected)
    if anomalies_detected == 'True':
        print("Anomalies detected. Retraining required")
    else:
        print("No anomalies detected. Skipping retraining.")



    