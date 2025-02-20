import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
from sklearn.datasets import load_breast_cancer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data():
    """Load and preprocess the breast cancer data."""
    try:
        # Load data from sklearn
        dataset = load_breast_cancer()
        feature_names = dataset.feature_names
        
        # Create DataFrame
        df = pd.DataFrame(dataset.data, columns=feature_names)
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
        
        return X_scaled, dataset.target, scaler
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise 