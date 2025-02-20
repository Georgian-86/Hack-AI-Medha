from abc import ABC, abstractmethod
import joblib
import logging
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.X_train = None
        self.y_train = None
    
    @abstractmethod
    def train(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    def save_model(self):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'X_train': self.X_train,
            'y_train': self.y_train
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls):
        instance = cls()
        with open(instance.model_path, 'rb') as f:
            model_data = pickle.load(f)
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.X_train = model_data['X_train']
        instance.y_train = model_data['y_train']
        return instance 