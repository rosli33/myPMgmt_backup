import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
from django.conf import settings

def train_model():
    print("Loading data...")
    
    # Construct the correct file path
    data_path = os.path.join(settings.BASE_DIR, 'tasks', 'data', 'cleaned_tasks.csv')
    
    try:
        # Load the data
        tasks_df = pd.read_csv(data_path)

        # Convert dates to datetime format
        tasks_df['Creation Date'] = pd.to_datetime(tasks_df['Creation Date'], format='%d/%m/%Y')
        tasks_df['Deadline'] = pd.to_datetime(tasks_df['Deadline'], format='%d/%m/%Y')

        # Create a new feature: 'Days Until Deadline'
        tasks_df['Days Until Deadline'] = (tasks_df['Deadline'] - tasks_df['Creation Date']).dt.days

        # Encode categorical variables
        label_encoders = {}
        for column in ['Task Type', 'Current Status', 'Assignee Name', 'Business Impact']:
            le = LabelEncoder()
            tasks_df[column] = le.fit_transform(tasks_df[column])
            label_encoders[column] = le

        # Encode 'Priority Level' which will be our target variable
        priority_le = LabelEncoder()
        tasks_df['Priority Level Encoded'] = priority_le.fit_transform(tasks_df['Priority Level'])

        # Feature Engineering
        tasks_df['Impact_Effort'] = tasks_df['Business Impact'] * tasks_df['Estimated Effort (Hours)']
        tasks_df['Log_Estimated_Effort'] = np.log1p(tasks_df['Estimated Effort (Hours)'])

        # Select features for model training
        X = tasks_df[['Task Type', 'Current Status', 'Business Impact', 'Days Until Deadline', 'Impact_Effort', 'Log_Estimated_Effort']]
        y = tasks_df['Priority Level Encoded']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Apply SMOTE to balance the dataset
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Define a RandomForest model
        rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)

        # Train the RandomForest model
        rf_model.fit(X_train_balanced, y_train_balanced)

        # Make predictions on the test set
        y_pred = rf_model.predict(X_test)

        # Generate a classification report
        report = classification_report(y_test, y_pred, target_names=priority_le.classes_)
        print(report)

        # Save the model
        model_path = os.path.join(settings.BASE_DIR, 'tasks', 'model', 'final_rf_model.pkl')
        joblib.dump(rf_model, model_path)
        print(f"Model trained and saved to {model_path} successfully!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
