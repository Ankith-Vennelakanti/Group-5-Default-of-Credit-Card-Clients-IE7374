from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from dataSplit.dataSplit import split
from preprocess_data.preprocess import process
from validate_data.train_validate import train_data_val
from ml.train_model import train_model
import os

current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
trainpath = os.path.join(current_directory, "dags/data/train_val_data.xlsx")
type = 'train'


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['youremail@email.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'start_date': datetime(2023,1,6),
}

with DAG('train_data',
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
                python_callable = process,
                # params={"path": trainpath},
                provide_context=True,
                op_args = [trainpath,type]
        )
        validate_train = PythonOperator(
                task_id = 'validate_train',
                python_callable = train_data_val
        )
        Train_model = PythonOperator(
                task_id = 'Train_model',
                python_callable = train_model
        )
        

        split_data >> pre_process >> validate_train >> Train_model