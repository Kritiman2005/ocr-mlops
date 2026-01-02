from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="ocr_retraining_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@weekly",
    catchup=False
):

    train = BashOperator(
        task_id="train_model",
        bash_command="python training/train.py"
    )

    register = BashOperator(
        task_id="register_model",
        bash_command="python registry/register_model.py"
    )

    train >> register
