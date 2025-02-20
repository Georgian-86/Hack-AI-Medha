from src.model import BreastCancerModel
from src.data_preprocessing import load_and_preprocess_data

def train():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, scaler = load_and_preprocess_data()
    
    # Initialize and train model
    print("Training model...")
    model = BreastCancerModel()
    model.scaler = scaler
    train_accuracy, test_accuracy = model.train(X, y)
    
    # Save the trained model
    print("Saving model...")
    model.save_model()
    
    print(f"Training completed!\nTrain accuracy: {train_accuracy:.4f}\nTest accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    train() 