import numpy as np
import pandas as pd
import os

def generate_telecom_churn_data(n_samples=5000, seed=42):
    rng = np.random.default_rng(seed)
    
    customer_ids = [f"CUST-{i:05d}" for i in range(1, n_samples + 1)]
    genders = rng.choice(['Male', 'Female'], n_samples)
    senior_citizens = rng.choice([0, 1], n_samples, p=[0.84, 0.16])
    partners = rng.choice(['Yes', 'No'], n_samples, p=[0.48, 0.52])
    dependents = rng.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70])
    
    tenure = rng.integers(1, 73, n_samples)
    phone_service = rng.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10])
    
    multiple_lines = []
    for ps in phone_service:
        if ps == 'No':
            multiple_lines.append('No phone service')
        else:
            multiple_lines.append(rng.choice(['Yes', 'No'], p=[0.45, 0.55]))
            
    internet_service = rng.choice(['Fiber optic', 'DSL', 'No'], n_samples, p=[0.44, 0.34, 0.22])
    
    online_security = []
    online_backup = []
    device_protection = []
    tech_support = []
    streaming_tv = []
    streaming_movies = []
    
    for iserv in internet_service:
        if iserv == 'No':
            online_security.append('No internet service')
            online_backup.append('No internet service')
            device_protection.append('No internet service')
            tech_support.append('No internet service')
            streaming_tv.append('No internet service')
            streaming_movies.append('No internet service')
        else:
            online_security.append(rng.choice(['Yes', 'No'], p=[0.30, 0.70]))
            online_backup.append(rng.choice(['Yes', 'No'], p=[0.35, 0.65]))
            device_protection.append(rng.choice(['Yes', 'No'], p=[0.35, 0.65]))
            tech_support.append(rng.choice(['Yes', 'No'], p=[0.29, 0.71]))
            streaming_tv.append(rng.choice(['Yes', 'No'], p=[0.38, 0.62]))
            streaming_movies.append(rng.choice(['Yes', 'No'], p=[0.38, 0.62]))
            
    contracts = rng.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.24, 0.21])
    paperless_billing = rng.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41])
    payment_methods = rng.choice([
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ], n_samples, p=[0.34, 0.22, 0.22, 0.22])
    
    monthly_charges = []
    for iserv in internet_service:
        if iserv == 'No':
            monthly_charges.append(np.round(rng.uniform(18.0, 25.0), 2))
        elif iserv == 'DSL':
            monthly_charges.append(np.round(rng.uniform(40.0, 85.0), 2))
        else:
            monthly_charges.append(np.round(rng.uniform(70.0, 118.0), 2))
            
    monthly_charges = np.array(monthly_charges)
    total_charges = np.round(monthly_charges * tenure + rng.uniform(-10, 10, n_samples), 2)
    total_charges = np.maximum(total_charges, monthly_charges)
    
    # Generate a noisy synthetic target from customer and service characteristics.
    churn_prob = np.zeros(n_samples)
    
    # Strong predictors of churn
    churn_prob += np.where(contracts == 'Month-to-month', 0.40, 0.05)
    churn_prob += np.where(internet_service == 'Fiber optic', 0.25, 0.05)
    churn_prob += np.where(np.array(tech_support) == 'No', 0.20, -0.10)
    churn_prob += np.where(np.array(online_security) == 'No', 0.15, -0.10)
    churn_prob += np.where(payment_methods == 'Electronic check', 0.15, 0.0)
    churn_prob += np.where(tenure < 12, 0.25, -0.20)
    churn_prob += np.where(tenure > 36, -0.25, 0.0)
    churn_prob += np.where(monthly_charges > 80, 0.15, -0.05)
    
    # Convert to probabilities using sigmoid
    churn_prob = 1 / (1 + np.exp(-churn_prob))
    churn = rng.binomial(1, churn_prob)
    
    df = pd.DataFrame({
        'customerID': customer_ids,
        'gender': genders,
        'SeniorCitizen': senior_citizens,
        'Partner': partners,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contracts,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_methods,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': np.where(churn == 1, 'Yes', 'No')
    })
    
    return df

if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'telecom_churn_data.csv')
    df = generate_telecom_churn_data(5000)
    df.to_csv(file_path, index=False)
    print(f"Generated {len(df)} customer churn records at {file_path}")
