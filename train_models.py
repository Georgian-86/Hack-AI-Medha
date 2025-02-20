import os
from src.model import BreastCancerModel
from src.models.diabetes import DiabetesModel
from src.models.heart_disease import HeartDiseaseModel
from src.preprocessing.diabetes import load_and_preprocess_diabetes_data
from src.preprocessing.heart_disease import load_and_preprocess_heart_data
from src.data_preprocessing import load_and_preprocess_data
from src.config import MODEL_DIR
from src.models.parkinsons import ParkinsonsModel
from src.preprocessing.parkinsons import load_and_preprocess_parkinsons_data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_model_dir():
    """Ensure the models directory exists"""
    os.makedirs(MODEL_DIR, exist_ok=True)

def train_breast_cancer():
    print("Training Breast Cancer Model...")
    try:
        # Load and preprocess data
        X, y, scaler = load_and_preprocess_data()
        
        # Initialize and train model
        model = BreastCancerModel()
        model.scaler = scaler
        train_acc, test_acc = model.train(X, y)
        
        print(f"Breast Cancer Model - Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}\n")
        model.save_model()
    except Exception as e:
        logging.error(f"Error in training Breast Cancer model: {str(e)}")
        raise

def train_diabetes():
    print("Training Diabetes Model...")
    try:
        # Load and preprocess data
        X, y, scaler = load_and_preprocess_diabetes_data()
        
        # Initialize and train model
        model = DiabetesModel()
        model.scaler = scaler
        train_acc, test_acc = model.train(X, y)
        
        print(f"Diabetes Model - Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}\n")
        model.save_model()
    except Exception as e:
        logging.error(f"Error in training Diabetes model: {str(e)}")
        raise

def train_heart_disease():
    print("Training Heart Disease Model...")
    try:
        # Load and preprocess data
        X, y, scaler = load_and_preprocess_heart_data()
        
        # Initialize and train model
        model = HeartDiseaseModel()
        model.scaler = scaler
        train_acc, test_acc = model.train(X, y)
        
        print(f"Heart Disease Model - Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}\n")
        model.save_model()
    except Exception as e:
        logging.error(f"Error in training Heart Disease model: {str(e)}")
        raise

def train_parkinsons():
    print("Training Parkinson's Disease Model...")
    try:
        # Load and preprocess data
        X, y, scaler = load_and_preprocess_parkinsons_data()
        
        # Initialize and train model
        model = ParkinsonsModel()
        model.scaler = scaler
        train_acc, test_acc = model.train(X, y)
        
        print(f"Parkinson's Disease Model - Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}\n")
        model.save_model()
    except Exception as e:
        logging.error(f"Error in training Parkinson's model: {str(e)}")
        raise

def main():
    ensure_model_dir()
    train_breast_cancer()
    train_diabetes()
    train_heart_disease()
    train_parkinsons()

if __name__ == "__main__":
    main()
    
    # Add other model training here as you implement them
    # print("\nTraining Diabetes Model...")
    # train_diabetes()
    # etc. 