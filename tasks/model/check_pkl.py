import joblib

model_path = 'tasks/model/final_rf_model.pkl'

try:
    # Load the model
    model = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load the model: {e}")
