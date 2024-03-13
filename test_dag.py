from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from dataSplit.dataSplit import split
from preprocess_data.new_data_preprocess import new_data
from validate_data.new_data_validate import new_data_val

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['youremail@email.com'],
    'start_date': datetime(2023,1,8),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
}

with DAG('weather_dag',
         default_args = default_args,
         schedule_interval = '@daily',
         is_paused_upon_creation=True,
         catchup = False
         ) as dag:
    
    
        pre_process = PythonOperator(
                task_id = 'pre_process',
                python_callable = new_data
        )
        validate_new_data = PythonOperator(
                task_id = 'validate_new_data',
                python_callable = new_data_val
        )
        

        pre_process >> validate_new_data