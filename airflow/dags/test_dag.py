import os
from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from preprocess_data.preprocess import process
from validate_data.new_data_validate import new_data_val
from ml.predict import predict_data


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
    
        # split_data = PythonOperator(
        #         task_id = 'split_data',
        #         python_callable = split
        # )
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