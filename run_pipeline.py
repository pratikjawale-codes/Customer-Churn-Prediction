import os
import sys

def main():
    print("==================================================================")
    print("Telecom Customer Churn Prediction End-to-End Pipeline")
    print("==================================================================")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(base_dir)
    
    data_dir = os.path.join(base_dir, 'data')
    data_file = os.path.join(data_dir, 'telecom_churn_data.csv')
    artifacts_dir = os.path.join(base_dir, 'artifacts')
    
    # Step 1: Generate Data
    print("\n[1/3] Generating Telecom Customer Dataset...")
    from data.generate_data import generate_telecom_churn_data
    df = generate_telecom_churn_data(n_samples=5000)
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(data_file, index=False)
    print(f"Data generated successfully: {data_file} ({len(df)} rows)")
    
    # Step 2: Train & Evaluate Models
    print("\n[2/3] Executing Model Training, Hyperparameter Tuning & Evaluation...")
    from src.train import train_and_evaluate_models
    best_model, results = train_and_evaluate_models(data_file, artifacts_dir)
    
    # Step 3: Summary & Verification
    print("\n[3/3] Pipeline Execution Summary:")
    print("-" * 50)
    print(f"Best Model Selected: {best_model}")
    test_metrics = results["test_metrics"]
    print(f"Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
    print("-" * 50)
    print("Pipeline Completed Successfully. Model artifacts ready for deployment!")

if __name__ == '__main__':
    main()
