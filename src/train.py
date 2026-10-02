import os
import sys
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import mlflow

# Add current dir and parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_processing import ChurnDataPreprocessor


def train_and_evaluate_models(data_path, artifacts_dir):
    os.makedirs(artifacts_dir, exist_ok=True)
    
    print("--- Step 1: Loading Data ---")
    df = pd.read_csv(data_path)
    if 'Churn' not in df.columns:
        raise ValueError("Training data must include a 'Churn' target column.")

    dedupe_columns = [column for column in df.columns if column != 'customerID']
    df = df.drop_duplicates(subset=dedupe_columns).reset_index(drop=True)
    print(f"Loaded dataset with shape: {df.shape}")
    
    target = ChurnDataPreprocessor.extract_target(df)
    if target.nunique() != 2:
        raise ValueError("Training data must contain both churn classes.")

    print("--- Step 2: Split Data Before Preprocessing ---")
    train_df, holdout_df = train_test_split(
        df, test_size=0.40, random_state=42, stratify=target
    )
    validation_df, test_df = train_test_split(
        holdout_df,
        test_size=0.50,
        random_state=42,
        stratify=ChurnDataPreprocessor.extract_target(holdout_df),
    )

    print("--- Step 3: Fit Preprocessing on Training Data Only ---")
    preprocessor = ChurnDataPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_validation = preprocessor.transform(validation_df)
    y_validation = ChurnDataPreprocessor.extract_target(validation_df)
    X_test = preprocessor.transform(test_df)
    y_test = ChurnDataPreprocessor.extract_target(test_df)

    print(
        f"Train rows: {len(train_df)}, validation rows: {len(validation_df)}, "
        f"test rows: {len(test_df)}"
    )
    
    # Define models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=1),
        'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, eval_metric='logloss', random_state=42, n_jobs=1),
        'LightGBM': lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, random_state=42, verbose=-1, n_jobs=1),
        'CatBoost': cb.CatBoostClassifier(iterations=200, depth=6, learning_rate=0.08, verbose=0, random_seed=42, thread_count=1)
    }
    
    print("--- Step 4: Model Selection on Validation Data ---")
    mlflow.set_experiment("Telecom_Customer_Churn_Prediction")
    
    results = {}
    best_model_name = None
    best_auc = -1.0
    best_model = None
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"Training {name}...")
            
            if name == 'XGBoost':
                # Hyperparameter tuning on XGBoost
                param_grid = {
                    'n_estimators': [150, 250],
                    'max_depth': [4, 6],
                    'learning_rate': [0.05, 0.1]
                }
                grid = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', n_jobs=1)
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                mlflow.log_params(grid.best_params_)
            elif name == 'Random Forest':
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [8, 12]
                }
                grid = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', n_jobs=1)
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                mlflow.log_params(grid.best_params_)
            else:
                model.fit(X_train, y_train)
                
            y_pred = model.predict(X_validation)
            y_proba = (
                model.predict_proba(X_validation)[:, 1]
                if hasattr(model, "predict_proba")
                else y_pred
            )
            
            acc = accuracy_score(y_validation, y_pred)
            prec = precision_score(y_validation, y_pred, zero_division=0)
            rec = recall_score(y_validation, y_pred, zero_division=0)
            f1 = f1_score(y_validation, y_pred, zero_division=0)
            auc = roc_auc_score(y_validation, y_proba)
            
            print(
                f"[{name}] Validation — Accuracy: {acc:.4f} | "
                f"Precision: {prec:.4f} | Recall: {rec:.4f} | "
                f"F1: {f1:.4f} | ROC-AUC: {auc:.4f}"
            )
            
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)
            
            results[name] = {
                "accuracy": round(float(acc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
                "roc_auc": round(float(auc), 4)
            }
            
            if auc > best_auc:
                best_auc = auc
                best_model_name = name
                best_model = model

    print(f"\n[BEST MODEL] Best Performing Model: {best_model_name} with ROC-AUC = {best_auc:.4f}")
    
    print("--- Step 5: Final Holdout Evaluation ---")
    final_training_df = pd.concat([train_df, validation_df], ignore_index=True)
    final_preprocessor = ChurnDataPreprocessor()
    X_final_train, y_final_train = final_preprocessor.fit_transform(final_training_df)
    X_test = final_preprocessor.transform(test_df)
    best_model.fit(X_final_train, y_final_train)
    preprocessor = final_preprocessor

    final_pred = best_model.predict(X_test)
    final_proba = best_model.predict_proba(X_test)[:, 1]
    test_metrics = {
        "accuracy": round(float(accuracy_score(y_test, final_pred)), 4),
        "precision": round(float(precision_score(y_test, final_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, final_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, final_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, final_proba)), 4),
    }
    results["test_metrics"] = test_metrics
    print(f"Untouched test metrics: {test_metrics}")

    # Save the final preprocessor fitted on training and validation data only.
    preprocessor_path = os.path.join(artifacts_dir, 'preprocessor.pkl')
    preprocessor.save(preprocessor_path)
    print(f"Saved preprocessor to {preprocessor_path}")

    best_model_path = os.path.join(artifacts_dir, 'best_model.pkl')
    joblib.dump(best_model, best_model_path)
    print(f"Saved best model ({best_model_name}) to {best_model_path}")
    
    metrics_path = os.path.join(artifacts_dir, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            "best_model": best_model_name,
            "validation_results": {
                name: metrics for name, metrics in results.items()
                if name != "test_metrics"
            },
            "test_metrics": test_metrics,
        }, f, indent=4)
        
    print(f"Saved metrics summary to {metrics_path}")
    return best_model_name, results

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, 'data', 'telecom_churn_data.csv')
    artifacts_folder = os.path.join(base_dir, 'artifacts')
    
    if not os.path.exists(data_file):
        from data.generate_data import generate_telecom_churn_data
        df = generate_telecom_churn_data(5000)
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        df.to_csv(data_file, index=False)
        
    train_and_evaluate_models(data_file, artifacts_folder)
