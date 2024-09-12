import joblib

# Load the model
MODEL_PATH = 'tasks/model/final_rf_model.pkl'
model = joblib.load(MODEL_PATH)

# Check if the model has feature importances (available for RandomForest, GradientBoosting, etc.)
if hasattr(model, 'feature_importances_'):
    print("Feature Importances Available:")
    print(model.feature_importances_)

# Check if the model has feature names (available if you used column names during training)
if hasattr(model, 'feature_names_in_'):
    print("Feature Names Used During Training:")
    print(model.feature_names_in_)
