import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

class ChurnDataPreprocessor:
    def __init__(self):
        self.preprocessor = None
        self.feature_names = []
        self.num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charge_per_month_ratio', 'total_services_count']
        self.cat_cols = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group'
        ]

    def engineer_features(self, df):
        data = df.copy()
        
        # Handle TotalCharges numeric conversion
        if 'TotalCharges' in data.columns:
            data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
            data['TotalCharges'] = data['TotalCharges'].fillna(data['MonthlyCharges'] * data['tenure'])
            
        # Feature 1: Tenure Grouping
        def get_tenure_group(t):
            if t <= 12:
                return '0-12_months'
            elif t <= 24:
                return '12-24_months'
            elif t <= 48:
                return '24-48_months'
            elif t <= 60:
                return '48-60_months'
            else:
                return '60+_months'
                
        data['tenure_group'] = data['tenure'].apply(get_tenure_group)
        
        # Feature 2: Charge Per Month Ratio
        expected_total = data['MonthlyCharges'] * data['tenure']
        expected_total = np.where(expected_total == 0, 1.0, expected_total)
        data['charge_per_month_ratio'] = np.round(data['TotalCharges'] / expected_total, 4)
        
        # Feature 3: Total Add-on Services Count
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        data['total_services_count'] = 0
        for col in service_cols:
            if col in data.columns:
                data['total_services_count'] += (data[col] == 'Yes').astype(int)
                
        # Ensure SeniorCitizen is treated as categorical
        if 'SeniorCitizen' in data.columns:
            data['SeniorCitizen'] = data['SeniorCitizen'].astype(str)
            
        return data

    def clean_data(self, df):
        data = df.copy()

        # Drop ID column if present
        if 'customerID' in data.columns:
            data = data.drop(columns=['customerID'])
            
        return data

    @staticmethod
    def extract_target(df):
        if 'Churn' not in df.columns:
            raise ValueError("Training data must include a 'Churn' target column.")

        values = df['Churn'].astype(str).str.strip().str.lower()
        accepted_values = {'yes', 'no', '1', '0', 'true', 'false'}
        invalid_values = set(values.unique()) - accepted_values
        if invalid_values:
            raise ValueError(f"Unsupported Churn labels: {sorted(invalid_values)}")

        return values.isin({'yes', '1', 'true'}).astype(int)

    def fit_transform(self, df):
        df_clean = self.clean_data(df)
        df_engineered = self.engineer_features(df_clean)
        
        if 'Churn' in df_engineered.columns:
            y = self.extract_target(df_engineered)
            X = df_engineered.drop(columns=['Churn'])
        else:
            y = None
            X = df_engineered

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.num_cols),
                ('cat', categorical_transformer, self.cat_cols)
            ]
        )

        X_transformed = self.preprocessor.fit_transform(X)
        
        # Extract feature names
        cat_onehot_cols = list(
            self.preprocessor.named_transformers_['cat']
            .named_steps['onehot']
            .get_feature_names_out(self.cat_cols)
        )
        self.feature_names = self.num_cols + cat_onehot_cols
        
        return X_transformed, y

    def transform(self, df):
        df_clean = self.clean_data(df)
        df_engineered = self.engineer_features(df_clean)
        if 'Churn' in df_engineered.columns:
            X = df_engineered.drop(columns=['Churn'])
        else:
            X = df_engineered
            
        return self.preprocessor.transform(X)

    def save(self, filepath):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)
