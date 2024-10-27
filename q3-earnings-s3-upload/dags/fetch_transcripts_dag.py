# dags/fetch_transcripts_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from main import main as fetch_and_store_transcripts

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 28),
    'email': ['youremail@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'fetch_transcripts',
    default_args=default_args,
    description='Fetch and store earnings call transcripts',
    schedule_interval=timedelta(days=1),  # Adjust as needed
)

fetch_task = PythonOperator(
    task_id='fetch_and_store_transcripts',
    python_callable=fetch_and_store_transcripts,
    dag=dag,
)

fetch_task
