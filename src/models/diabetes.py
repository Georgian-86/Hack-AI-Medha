from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .base_model import BaseModel
from ..config import DIABETES_MODEL_PATH, RANDOM_STATE, TEST_SIZE
import numpy as np

class DiabetesModel(BaseModel):
    def __init__(self):
        super().__init__(DIABETES_MODEL_PATH)
        self.model = KNeighborsClassifier(
            n_neighbors=7,  # Increased neighbors for more robust prediction
            weights='distance'  # Weight points by distance
        )
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
            'GlucoseBMI', 'GlucoseAge'  # Added derived features
        ]
        self.X_train = None
        self.y_train = None
        
        # Define risk thresholds
        self.high_risk_threshold = 0.6
    
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=y  # Ensure balanced split
        )
        
        self.X_train = X_train
        self.y_train = y_train
        
        self.model.fit(X_train, y_train)
        return self.evaluate(X_train, X_test, y_train, y_test)
    
    def predict(self, X):
        if self.scaler:
            X = self.scaler.transform(X)
            
        # Get distances and indices of nearest neighbors
        distances, indices = self.model.kneighbors(X)
        
        # Get similar cases
        similar_cases = self.X_train.iloc[indices[0]]
        similar_outcomes = self.y_train.iloc[indices[0]]
        
        # Calculate weighted probability
        weights = 1 / (distances[0] + 1e-6)  # Add small constant to avoid division by zero
        weighted_prob = np.sum(similar_outcomes * weights) / np.sum(weights)
        
        # Make prediction based on probability threshold
        prediction = np.array([1 if weighted_prob >= self.high_risk_threshold else 0])
        
        return prediction, similar_cases, similar_outcomes, distances[0]
    
    def evaluate(self, X_train, X_test, y_train, y_test):
        train_accuracy = accuracy_score(y_train, self.model.predict(X_train))
        test_accuracy = accuracy_score(y_test, self.model.predict(X_test))
        return train_accuracy, test_accuracy 