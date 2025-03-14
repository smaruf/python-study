from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Define the DAG
dag = DAG('example_dag', start_date=datetime(2023, 1, 1), schedule_interval='@daily')

# Define tasks
def task_a():
    print("Task A runs")

def task_b():
    print("Task B runs")

run_task_a = PythonOperator(task_id='task_a', python_callable=task_a, dag=dag)
run_task_b = PythonOperator(task_id='task_b', python_callable=task_b, dag=dag)

# Set dependencies
run_task_a >> run_task_b
