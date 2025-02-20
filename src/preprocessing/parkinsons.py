import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_parkinsons_data():
    try:
        # Load the dataset from local datasets folder
        data_path = Path(__file__).resolve().parent.parent.parent / "datasets" / "parkinsons.csv"
        df = pd.read_csv(data_path)
        
        # Drop the 'name' column if it exists
        if 'name' in df.columns:
            df = df.drop('name', axis=1)
        
        # Rename 'status' to match our convention (1 for disease, 0 for healthy)
        if 'status' in df.columns:
            df['status'] = df['status'].map({0: 1, 1: 0})
        
        # Separate features and target
        X = df.drop('status', axis=1)
        y = df['status']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X_scaled, y, scaler
        
    except Exception as e:
        logger.error(f"Error in Parkinson's data preprocessing: {str(e)}")
        raise 