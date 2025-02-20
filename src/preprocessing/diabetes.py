import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_diabetes_data():
    try:
        # Load the dataset from local datasets folder
        data_path = Path(__file__).resolve().parent.parent.parent / "datasets" / "diabetes.csv"
        df = pd.read_csv(data_path)
        
        feature_names = [
            'Pregnancies',      # Number of times pregnant
            'Glucose',          # Plasma glucose concentration (mg/dL)
            'BloodPressure',    # Diastolic blood pressure (mm Hg)
            'SkinThickness',    # Triceps skin fold thickness (mm)
            'Insulin',          # 2-Hour serum insulin (mu U/ml)
            'BMI',              # Body mass index
            'DiabetesPedigreeFunction',  # Diabetes pedigree function
            'Age'               # Age in years
        ]
        
        # Handle missing values (0 values in certain columns)
        zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for column in zero_not_accepted:
            mask = df[column] != 0
            df.loc[~mask, column] = df.loc[mask, column].median()
        
        # Add some derived features
        df['GlucoseBMI'] = df['Glucose'] * df['BMI'] / 1000
        df['GlucoseAge'] = df['Glucose'] * df['Age'] / 100
        feature_names.extend(['GlucoseBMI', 'GlucoseAge'])
        
        # Separate features and target
        X = df[feature_names]
        y = df['Outcome']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
        
        return X_scaled, y, scaler
        
    except Exception as e:
        logger.error(f"Error in diabetes data preprocessing: {str(e)}")
        raise 