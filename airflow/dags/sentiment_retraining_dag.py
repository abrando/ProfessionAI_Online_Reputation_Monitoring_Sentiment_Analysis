# airflow/dags/sentiment_retraining_dag.py
"""
Airflow DAG for scheduled sentiment model retraining checks.

This DAG:
- Runs once a week (configurable via schedule_interval).
- Executes the same logic as the GitHub Actions `retrain.yml` workflow:
  it calls `src.retrain.main()` which:
    - checks for new labeled data in data/new/*.csv
    - prints a plan for retraining

This shows how Airflow orchestrates the ML lifecycle in a
production-like environment, while GitHub Actions covers CI/CD.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def retrain_entrypoint():
    """
    Entrypoint wrapper for Airflow.

    It simply imports and calls `src.retrain.main()`. The assumption is that
    the Airflow environment sees the project code in its PYTHONPATH
    """
    from src.retrain import main as retrain_main

    retrain_main()


default_args = {
    "owner": "machineinnovators",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="sentiment_retraining_check",
    default_args=default_args,
    description="Weekly retraining check for the sentiment model",
    schedule_interval="0 3 * * 1",  # Every Monday at 03:00
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["sentiment", "mlops", "retraining"],
) as dag:

    retrain_task = PythonOperator(
        task_id="check_and_plan_retraining",
        python_callable=retrain_entrypoint,
    )

    retrain_task