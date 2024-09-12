import joblib
import os
from django.conf import settings

def predict_priority(task_data):
    # Load the saved model from tasks/model/final_rf_model.pkl
    model_path = os.path.join(settings.BASE_DIR, 'tasks', 'model', 'final_rf_model.pkl')
    
    # Load the trained model
    model = joblib.load(model_path)
    
    # Make sure task_data is in the correct format for prediction
    # task_data should be a list or numpy array with the same structure as the features used in training
    # Example format: [task_type, current_status, business_impact, days_until_deadline, impact_effort, log_estimated_effort]
    
    prediction = model.predict([task_data])
    return prediction[0]  # Return the predicted priority level
