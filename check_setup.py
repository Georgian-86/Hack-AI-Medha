import os

def check_project_setup():
    # Check directory structure
    directories = ['app', 'models', 'src', 'data']
    for dir in directories:
        if not os.path.exists(dir):
            print(f"Missing directory: {dir}")
            os.makedirs(dir)
            print(f"Created directory: {dir}")
    
    # Check required files
    required_files = [
        'app/streamlit_app.py',
        'src/config.py',
        'src/data_preprocessing.py',
        'src/model.py',
        'src/__init__.py',
        'train_model.py'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"Missing file: {file}")
        else:
            print(f"Found file: {file}")
    
    # Check if model exists
    if not os.path.exists('models/breast_cancer_model.pkl'):
        print("Model file not found. Please run train_model.py first")
    else:
        print("Model file found")

if __name__ == "__main__":
    check_project_setup() 