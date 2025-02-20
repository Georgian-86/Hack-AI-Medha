from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .models.base_model import BaseModel
from .config import BREAST_CANCER_MODEL_PATH, RANDOM_STATE, TEST_SIZE
import numpy as np
import pandas as pd

class BreastCancerModel(BaseModel):
    def __init__(self):
        super().__init__(BREAST_CANCER_MODEL_PATH)
        self.model = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        )
        self.feature_names = [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
            'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
            'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
            'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
            'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
            'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
        ]
        self.X_train = None
        self.y_train = None
        
        # Define risk thresholds
        self.high_risk_threshold = 0.5
        
        # Feature importance weights
        self.feature_weights = {
            'mean radius': 1.5,
            'mean texture': 1.2,
            'mean perimeter': 1.5,
            'mean area': 1.5,
            'mean concave points': 2.0,
            'worst radius': 1.8,
            'worst perimeter': 1.8,
            'worst area': 1.8,
            'worst concave points': 2.0
        }
    
    def train(self, X, y):
        # Convert input to DataFrame if it's not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
            
        # Apply feature weights
        X_weighted = X.copy()
        for feature, weight in self.feature_weights.items():
            if feature in X.columns:
                X_weighted[feature] = X_weighted[feature] * weight
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_weighted, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=y
        )
        
        # Store training data as DataFrame/Series
        self.X_train = pd.DataFrame(X_train, columns=self.feature_names)
        self.y_train = pd.Series(y_train)
        
        self.model.fit(X_train, y_train)
        return self.evaluate(X_train, X_test, y_train, y_test)
    
    def predict(self, X):
        # Convert input to DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
            
        if self.scaler:
            X = pd.DataFrame(self.scaler.transform(X), columns=self.feature_names)
        
        # Apply feature weights
        for feature, weight in self.feature_weights.items():
            if feature in X.columns:
                X[feature] = X[feature] * weight
        
        # Get nearest neighbors
        distances, indices = self.model.kneighbors(X)
        
        # Ensure X_train and y_train are DataFrame/Series
        if isinstance(self.X_train, np.ndarray):
            self.X_train = pd.DataFrame(self.X_train, columns=self.feature_names)
        if isinstance(self.y_train, np.ndarray):
            self.y_train = pd.Series(self.y_train)
        
        # Get similar cases
        similar_cases = self.X_train.iloc[indices[0]]
        similar_outcomes = self.y_train.iloc[indices[0]]
        
        # Calculate weighted probability
        weights = 1 / (distances[0] + 1e-6)
        weighted_prob = np.sum(similar_outcomes * weights) / np.sum(weights)
        
        # Check risk factors
        if self.scaler:
            X_orig = pd.DataFrame(self.scaler.inverse_transform(X), columns=self.feature_names)
        else:
            X_orig = X
            
        # Add risk based on key measurements
        if X_orig['mean radius'].iloc[0] > 15:
            weighted_prob += 0.1
        if X_orig['mean concave points'].iloc[0] > 0.05:
            weighted_prob += 0.15
        if X_orig['worst radius'].iloc[0] > 20:
            weighted_prob += 0.15
        if X_orig['worst concave points'].iloc[0] > 0.15:
            weighted_prob += 0.15
            
        # Make prediction based on threshold
        prediction = np.array([0 if weighted_prob >= self.high_risk_threshold else 1])
        
        return prediction, similar_cases, similar_outcomes, distances[0]
    
    def evaluate(self, X_train, X_test, y_train, y_test):
        train_accuracy = accuracy_score(y_train, self.model.predict(X_train))
        test_accuracy = accuracy_score(y_test, self.model.predict(X_test))
        return train_accuracy, test_accuracy 