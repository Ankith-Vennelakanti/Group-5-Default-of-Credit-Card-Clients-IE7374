import os
from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from preprocess_data.preprocess import process
from validate_data.new_data_validate import new_data_val
from ml.predict import predict_data


"""
    DAG to preprocess, validate, and predict data.

    This DAG defines tasks to preprocess incoming data, validate its integrity,
    and make predictions using a machine learning model.

    Tasks:
    1. pre_process: Preprocess the incoming data.
    2. validate_test: Validate the preprocessed data.
    3. predict: Make predictions using the preprocessed data.

    Parameters:
    - default_args: Default arguments for the DAG.
    - schedule_interval: Interval at which the DAG runs.
    - is_paused_upon_creation: Whether the DAG should be paused upon creation.
    - catchup: Whether to catch up the DAG runs for the interval between start_date and the current date.

    Returns:
        None
"""


current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
testpath = os.path.join(current_directory, "dags/data/test_data.xlsx")
type = 'test'


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['youremail@email.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'start_date': datetime(2023,1,6),
}

with DAG('predict_data',
         default_args = default_args,
         schedule_interval = '@daily',
         is_paused_upon_creation=True,
         catchup = False
         ) as dag:
    
        pre_process = PythonOperator(
                task_id = 'pre_process',
                python_callable = process,
                # params={"path": trainpath},
                provide_context=True,
                op_args = [testpath,type]
        )
        validate_test = PythonOperator(
                task_id = 'validate_test',
                python_callable = new_data_val
        )
        predict = PythonOperator(
                task_id = 'predict',
                python_callable = predict_data
        )
        

        pre_process >> validate_test >> predict