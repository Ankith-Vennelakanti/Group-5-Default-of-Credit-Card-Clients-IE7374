from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from dataSplit.dataSplit import split
from preprocess_data.train_preprocess import train_process
from validate_data.train_validate import train_data_val

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['youremail@email.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'start_date': datetime(2023,1,8),
}

with DAG('weather_dag',
         default_args = default_args,
         schedule_interval = '@daily',
         is_paused_upon_creation=True,
         catchup = False
         ) as dag:
    
        split_data = PythonOperator(
                task_id = 'split_data',
                python_callable = split
        )
        pre_process = PythonOperator(
                task_id = 'pre_process',
                python_callable = train_process
        )
        validate_train = PythonOperator(
                task_id = 'validate_train',
                python_callable = train_data_val
        )
        

        split_data >> pre_process >> validate_train