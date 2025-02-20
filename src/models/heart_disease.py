from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .base_model import BaseModel
from ..config import HEART_DISEASE_MODEL_PATH, RANDOM_STATE, TEST_SIZE
import numpy as np
import pandas as pd

class HeartDiseaseModel(BaseModel):
    def __init__(self):
        super().__init__(HEART_DISEASE_MODEL_PATH)
        self.model = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',  # Weight by distance for better local sensitivity
            metric='manhattan'   # Manhattan distance for better feature importance
        )
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        self.X_train = None
        self.y_train = None
        
        # Define risk thresholds
        self.high_risk_threshold = 0.5
        
        # Feature importance weights
        self.feature_weights = {
            'age': 1.5,         # Age is important
            'cp': 2.0,          # Chest pain type is very important
            'trestbps': 1.2,    # Blood pressure
            'chol': 1.2,        # Cholesterol
            'thalach': 1.5,     # Max heart rate
            'oldpeak': 1.8,     # ST depression
            'ca': 2.0,          # Number of vessels
            'thal': 1.5         # Thalassemia
        }
    
    def train(self, X, y):
        X = X[self.feature_names]
        
        # Apply feature weights
        for feature, weight in self.feature_weights.items():
            if feature in X.columns:
                X[feature] = X[feature] * weight
        
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
        X = pd.DataFrame(X, columns=self.feature_names)
        
        # Apply feature weights
        for feature, weight in self.feature_weights.items():
            if feature in X.columns:
                X[feature] = X[feature] * weight
        
        # Get nearest neighbors
        distances, indices = self.model.kneighbors(X)
        
        # Get similar cases
        similar_cases = self.X_train.iloc[indices[0]]
        similar_outcomes = self.y_train.iloc[indices[0]]
        
        # Calculate risk score based on weighted voting
        weights = 1 / (distances[0] + 1e-6)
        weighted_prob = np.sum(similar_outcomes * weights) / np.sum(weights)
        
        # Calculate additional risk factors
        risk_factors = []
        
        # Convert X back to original scale if scaler exists
        if self.scaler:
            X_orig = pd.DataFrame(self.scaler.inverse_transform(X), columns=self.feature_names)
        else:
            X_orig = X
        
        # Check various risk factors
        if X_orig['age'].iloc[0] > 60:
            weighted_prob += 0.1
        if X_orig['cp'].iloc[0] >= 2:  # Non-typical chest pain
            weighted_prob += 0.1
        if X_orig['trestbps'].iloc[0] > 140:  # High blood pressure
            weighted_prob += 0.1
        if X_orig['chol'].iloc[0] > 240:  # High cholesterol
            weighted_prob += 0.1
        if X_orig['thalach'].iloc[0] < 120:  # Low max heart rate
            weighted_prob += 0.1
        if X_orig['oldpeak'].iloc[0] > 2:  # High ST depression
            weighted_prob += 0.15
        if X_orig['ca'].iloc[0] > 0:  # Presence of vessels colored by fluoroscopy
            weighted_prob += 0.15 * X_orig['ca'].iloc[0]
        
        # Make final prediction based on threshold
        prediction = np.array([1 if weighted_prob >= self.high_risk_threshold else 0])
        
        return prediction, similar_cases, similar_outcomes, distances[0]
    
    def evaluate(self, X_train, X_test, y_train, y_test):
        train_accuracy = accuracy_score(y_train, self.model.predict(X_train))
        test_accuracy = accuracy_score(y_test, self.model.predict(X_test))
        return train_accuracy, test_accuracy 