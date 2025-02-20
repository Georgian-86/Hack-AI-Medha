from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from .base_model import BaseModel
from ..config import BREAST_CANCER_MODEL_PATH, RANDOM_STATE, TEST_SIZE

class BreastCancerModel(BaseModel):
    def __init__(self):
        super().__init__(BREAST_CANCER_MODEL_PATH)
        self.model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        self.model.fit(X_train, y_train)
        return self.evaluate(X_train, X_test, y_train, y_test)
    
    def predict(self, X):
        if self.scaler:
            X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def evaluate(self, X_train, X_test, y_train, y_test):
        train_accuracy = accuracy_score(y_train, self.model.predict(X_train))
        test_accuracy = accuracy_score(y_test, self.model.predict(X_test))
        return train_accuracy, test_accuracy 