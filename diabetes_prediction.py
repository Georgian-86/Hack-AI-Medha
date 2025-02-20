import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

def load_data():
    # Load the dataset
    diabetes_dataset = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\HackAI\diabetes.csv')
    return diabetes_dataset

def preprocess_data(diabetes_dataset):
    # Prepare data
    X = diabetes_dataset.drop(columns='Outcome', axis=1)
    Y = diabetes_dataset['Outcome']
    
    # Standardize features
    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    X = standardized_data
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    return X_train, X_test, Y_train, Y_test, scaler

def train_model(X_train, Y_train):
    # Train model
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)
    return classifier

def evaluate_model(classifier, X_train, X_test, Y_train, Y_test):
    # Evaluate model
    X_train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    print('Accuracy score of the training data:', training_data_accuracy)

    X_test_prediction = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    print('Accuracy score of the test data:', test_data_accuracy)

def predict_diabetes(input_data, classifier, scaler):
    # Convert input to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # Reshape for single prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Standardize the input
    std_data = scaler.transform(input_data_reshaped)
    
    # Make prediction
    prediction = classifier.predict(std_data)
    
    # Return result
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

if __name__ == "__main__":
    # Load and preprocess data
    diabetes_dataset = load_data()
    X_train, X_test, Y_train, Y_test, scaler = preprocess_data(diabetes_dataset)
    
    # Train model
    classifier = train_model(X_train, Y_train)
    
    # Evaluate model
    evaluate_model(classifier, X_train, X_test, Y_train, Y_test)
    
    # Example prediction
    input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
    result = predict_diabetes(input_data, classifier, scaler)
    print(result)
