import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_heart_data():
    try:
        # Load the dataset from local datasets folder
        data_path = Path(__file__).resolve().parent.parent.parent / "datasets" / "heart.csv"
        df = pd.read_csv(data_path)
        
        feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        # Handle missing values if any
        df = df.replace('?', pd.NA).dropna()
        
        # Separate features and target
        X = df[feature_names]
        y = df['target']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
        
        return X_scaled, y, scaler
        
    except Exception as e:
        logger.error(f"Error in heart disease data preprocessing: {str(e)}")
        raise 