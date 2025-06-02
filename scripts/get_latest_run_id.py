# scripts/get_latest_run_id.py
import mlflow
import os
import sys # Untuk exit codes

try:
    tracking_uri_py = os.environ.get('MLFLOW_TRACKING_URI')
    if not tracking_uri_py: # Cek apakah None atau string kosong
        print('Python script Error: MLFLOW_TRACKING_URI environment variable not set or empty.')
        sys.exit(1)
    mlflow.set_tracking_uri(tracking_uri_py)
    print(f'Python script: MLFLOW_TRACKING_URI set to {tracking_uri_py}')

    experiment_name_py = os.environ.get('EXPERIMENT_NAME_FOR_PYTHON')
    if not experiment_name_py: # Cek apakah None atau string kosong
        print('Python script Error: EXPERIMENT_NAME_FOR_PYTHON environment variable not set or empty.')
        sys.exit(1)
    print(f'Python script: Searching for experiment named "{experiment_name_py}"')

    experiment = mlflow.get_experiment_by_name(experiment_name_py)
    if experiment is None:
        print(f'Python script Error: Experiment "{experiment_name_py}" not found via API.')
        # Untuk debugging, list semua eksperimen jika tidak ditemukan:
        # print('Python script: Available experiments:')
        # for exp_obj in mlflow.search_experiments():
        #     print(f'- Name: "{exp_obj.name}", ID: "{exp_obj.experiment_id}"')
        sys.exit(1)
    experiment_id = experiment.experiment_id
    print(f'Python script: Found Experiment ID: {experiment_id}')

    runs_df = mlflow.search_runs(experiment_ids=[experiment_id], order_by=['attributes.start_time DESC'], max_results=1)

    if runs_df.empty:
        print(f'Python script Error: No runs found in experiment "{experiment_name_py}" (ID: "{experiment_id}").')
        sys.exit(1)

    latest_run_id_py = runs_df.iloc[0]['run_id']
    print(latest_run_id_py) # Output ini akan ditangkap oleh bash

except Exception as e:
    print(f'Python script execution error: {e}')
    sys.exit(1)