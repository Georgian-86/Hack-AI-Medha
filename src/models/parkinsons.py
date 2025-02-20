from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .base_model import BaseModel
from ..config import PARKINSONS_MODEL_PATH, RANDOM_STATE, TEST_SIZE
import numpy as np
import pandas as pd

class ParkinsonsModel(BaseModel):
    def __init__(self):
        super().__init__(PARKINSONS_MODEL_PATH)
        self.model = KNeighborsClassifier(
            n_neighbors=5,  # Increased for more robust predictions
            weights='distance',
            metric='euclidean'  # Changed to euclidean for better distance measurement
        )
        self.feature_names = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
            'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
            'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
            'spread1', 'spread2', 'D2', 'PPE'
        ]
        self.X_train = None
        self.y_train = None
        self.scaler = None
        
        # Feature ranges from dataset analysis
        self.feature_ranges = {
            'MDVP:Fo(Hz)': (88.333, 260.105),
            'MDVP:Fhi(Hz)': (102.145, 592.030),
            'MDVP:Flo(Hz)': (65.476, 239.170),
            'MDVP:Jitter(%)': (0.001, 0.033),
            'MDVP:Shimmer': (0.009, 0.119),
            'HNR': (8.441, 33.047),
            'RPDE': (0.256, 0.685),
            'DFA': (0.574, 0.825),
            'spread1': (-7.968984, -2.434031),
            'spread2': (0.006, 0.527),
            'PPE': (0.044, 0.527)
        }
        
        # Add feature weights
        self.feature_weights = {
            'MDVP:Fo(Hz)': 1.0,
            'MDVP:Fhi(Hz)': 1.0,
            'MDVP:Flo(Hz)': 1.0,
            'MDVP:Jitter(%)': 2.0,
            'MDVP:Jitter(Abs)': 1.5,
            'MDVP:RAP': 1.5,
            'MDVP:PPQ': 1.5,
            'Jitter:DDP': 1.5,
            'MDVP:Shimmer': 2.0,
            'MDVP:Shimmer(dB)': 1.5,
            'Shimmer:APQ3': 1.5,
            'Shimmer:APQ5': 1.5,
            'MDVP:APQ': 1.5,
            'Shimmer:DDA': 1.5,
            'NHR': 1.8,
            'HNR': 1.8,
            'RPDE': 1.5,
            'DFA': 1.5,
            'spread1': 1.2,
            'spread2': 1.2,
            'D2': 1.2,
            'PPE': 1.8
        }
        
    def is_input_valid(self, X):
        """Check if input values are within expected ranges"""
        X_df = pd.DataFrame(X, columns=self.feature_names)
        for feature, (min_val, max_val) in self.feature_ranges.items():
            if feature in X_df.columns:
                value = X_df[feature].iloc[0]
                # Extend the acceptable range by 20% on both sides
                range_width = max_val - min_val
                extended_min = min_val - (range_width * 0.2)
                extended_max = max_val + (range_width * 0.2)
                if value < extended_min or value > extended_max:
                    return False, f"{feature} value ({value:.3f}) is outside expected range ({min_val:.3f} - {max_val:.3f})"
        return True, ""

    def predict(self, X):
        # Validate input
        is_valid, message = self.is_input_valid(X)
        if not is_valid:
            raise ValueError(f"Invalid input: {message}")
            
        if self.scaler:
            X = self.scaler.transform(X)
        
        X = pd.DataFrame(X, columns=self.feature_names)
        
        # Apply feature weights
        for feature, weight in self.feature_weights.items():
            if feature in X.columns:
                X[feature] = X[feature] * weight
        
        # Get nearest neighbors
        distances, indices = self.model.kneighbors(X)
        
        # Convert X_train to DataFrame if it's a numpy array
        if isinstance(self.X_train, np.ndarray):
            self.X_train = pd.DataFrame(self.X_train, columns=self.feature_names)
        
        # Get similar cases
        similar_cases = self.X_train.iloc[indices[0]]
        similar_outcomes = pd.Series(self.y_train).iloc[indices[0]]  # Convert y_train to Series
        
        # Calculate confidence score based on distances
        max_distance = np.max(distances)
        confidence_scores = 1 - (distances[0] / max_distance)
        
        # Weight the predictions by confidence
        weighted_pred = np.average(similar_outcomes, weights=confidence_scores)
        
        # Make final prediction
        prediction = np.array([1 if weighted_pred >= 0.5 else 0])
        
        return prediction, similar_cases, similar_outcomes, distances[0]
    
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
        
        # Store as DataFrames/Series
        self.X_train = pd.DataFrame(X_train, columns=self.feature_names)
        self.y_train = pd.Series(y_train)
        
        self.model.fit(X_train, y_train)
        return self.evaluate(X_train, X_test, y_train, y_test)
    
    def evaluate(self, X_train, X_test, y_train, y_test):
        train_accuracy = accuracy_score(y_train, self.model.predict(X_train))
        test_accuracy = accuracy_score(y_test, self.model.predict(X_test))
        return train_accuracy, test_accuracy 