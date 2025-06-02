import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import numpy as np
import os
import time

# HAPUS SEMUA BLOK KODE YANG BERKAITAN DENGAN:
# EXPERIMENT_NAME = "..."
# mlflow.create_experiment(...)
# mlflow.get_experiment_by_name(...)
# mlflow.set_experiment(...)
# if not mlflow.active_run(): ... (dan blok kondisional terkait)

def load_data(file_path):
    """Memuat data dari file CSV."""
    print(f"Memuat data dari: {file_path}")
    df = pd.read_csv(file_path, index_col='date', parse_dates=['date'])
    print("Data berhasil dimuat.")
    return df

def tune_and_log_model(df):
    """Melakukan hyperparameter tuning dan manual logging ke MLflow."""
    if df.empty:
        print("DataFrame kosong, tuning dibatalkan.")
        return

    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3,
                               scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

    print("\nMemulai GridSearchCV untuk RandomForestRegressor...")
    start_time_grid_search = time.time()
    grid_search.fit(X_train, y_train)
    end_time_grid_search = time.time()
    grid_search_duration = end_time_grid_search - start_time_grid_search
    print(f"GridSearchCV selesai dalam {grid_search_duration:.2f} detik.")

    print(f"\nParameter terbaik ditemukan oleh GridSearchCV: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    # --- MLflow Manual Logging ---
    # Langsung log, karena 'mlflow run' sudah membuat run.

    # (Opsional) Dapatkan info run aktif untuk dicetak jika diperlukan
    active_run = mlflow.active_run()
    if active_run:
        run_id_for_print = active_run.info.run_id
        experiment_id_for_print = active_run.info.experiment_id
        print(f"\nMelakukan logging ke MLflow run ID: {run_id_for_print} dalam experiment ID: {experiment_id_for_print}")
    else:
        # Ini seharusnya tidak terjadi sekarang.
        print("Peringatan kritis: Tidak ada run MLflow yang aktif terdeteksi oleh skrip.")


    print("Mencatat parameter terbaik...")
    for param_name, param_value in grid_search.best_params_.items():
        mlflow.log_param(param_name, param_value)
    mlflow.log_param("cv_folds", grid_search.cv)

    print("Mencatat metrik evaluasi...")
    predictions = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    mlflow.log_metric("rmse_test", rmse)
    mlflow.log_metric("mae_test", mae)
    mlflow.log_metric("r2_score_test", r2)
    mlflow.log_metric("grid_search_duration_seconds", grid_search_duration)
    mlflow.log_metric("n_features_in", best_model.n_features_in_)
    best_cv_score = grid_search.best_score_
    mlflow.log_metric("best_cv_neg_mse", best_cv_score)
    mlflow.log_metric("best_cv_mse", -best_cv_score)

    print("\nMetrik evaluasi pada data uji (model terbaik):")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R2 Score: {r2:.4f}")

    print("Mencatat model terbaik...")
    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_train, best_model.predict(X_train))
    
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best-randomforest-model",
        signature=signature,
        input_example=X_train.head(5)
    )
    
    print("Model terbaik dan metrik telah dicatat secara manual ke MLflow.")
    print("Untuk melihat hasil, jalankan 'python -m mlflow ui' di terminal.")


def main():
    """Fungsi utama untuk menjalankan alur pemodelan dengan tuning."""
    print("--- Memulai Proses Pemodelan dengan Hyperparameter Tuning (Skilled) ---")
    data_path = os.path.join("namadataset_preprocessing", "bitcoin_idr_daily_processed.csv")
    df_processed = load_data(data_path)

    if not df_processed.empty:
        tune_and_log_model(df_processed)
    else:
        print("Tidak dapat melanjutkan karena data tidak berhasil dimuat.")

    print("\n--- Proses Pemodelan dengan Hyperparameter Tuning (Skilled) Selesai ---")

if __name__ == "__main__":
    main()