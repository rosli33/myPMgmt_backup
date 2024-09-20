import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tasks.forms import TASK_TYPE_CHOICES, STATUS_CHOICES, PRIORITY_CHOICES, BUSINESS_IMPACT_CHOICES
from io import BytesIO
import base64
from django.shortcuts import render
from tasks.models import Task, ManualTask
from tasks.forms import TaskInputForm, UploadTaskForm
from django.conf import settings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
from django.http import HttpResponseBadRequest
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import os

# Predict Task Priority
def predict_task_priority(df_task):
    # Load the ANN model
    MODEL_PATH = os.path.join(settings.BASE_DIR, 'tasks', 'model', 'best_ann_model.h5')
    model = load_model(MODEL_PATH)

    # Compile the model to add metrics for evaluation
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Predefined LabelEncoders for each column used during training
    task_type_le = LabelEncoder()
    current_status_le = LabelEncoder()
    business_impact_le = LabelEncoder()

    # Load the original training label encoders from the model training phase
    task_type_le.classes_ = np.array(['coa', 'portal', 'auth_control', 'finalization_integration', 'crm', 'data_storage', 
                                      'comm_collab', 'workflow_management', 'contract_documentation', 'tok', 
                                      'training_management', 'hrm', 'inventory_management', 'project_management', 
                                      'reporting_analytics', 'tot', 'ea', 'payment_gateway', 'change_management', 
                                      'notification_system', 'audit_compliance', 'devops', 'management_plans', 
                                      'monitoring_reports', 'ui_ux', 'data_lake_warehouse', 'data_services', 
                                      'closure_report', 'bcp', 'office_renovation', 'portal_lms', 'pmis'])  # Task Types
    current_status_le.classes_ = np.array(['todo', 'progress', 'completed'])  # Statuses
    business_impact_le.classes_ = np.array(['Low', 'Medium', 'High'])  # Business Impact

    # Rename columns to match what the model was trained on
    df_task = df_task.rename(columns={
        'task_type': 'Task_Type',
        'current_status': 'Current_Status',
        'business_impact': 'Business_Impact',
        'estimated_effort': 'Estimated Effort (Hours)',
        'priority': 'Impact_Effort',
        'deadline': 'Days Until Deadline'
    })

    # Handle unseen labels in your LabelEncoder
    df_task['Task_Type'] = df_task['Task_Type'].apply(lambda x: handle_unseen_labels(task_type_le, x))
    df_task['Current_Status'] = df_task['Current_Status'].apply(lambda x: handle_unseen_labels(current_status_le, x))
    df_task['Business_Impact'] = df_task['Business_Impact'].apply(lambda x: handle_unseen_labels(business_impact_le, x))

    # Calculate Days Until Deadline
    def calculate_days_until_deadline(deadline):
        # Handle date parsing explicitly
        try:
            deadline = pd.to_datetime(deadline, format='%Y-%m-%d')
            return (deadline - pd.to_datetime(datetime.now())).days
        except Exception as e:
            logger.error(f"Error parsing deadline: {e}")
            return None  # Or handle this more appropriately
        
    df_task['Days Until Deadline'] = df_task['Days Until Deadline'].apply(calculate_days_until_deadline)

    # Log-transform Estimated Effort (apply log1p)
    df_task['Log_Estimated_Effort'] = np.log1p(df_task['Estimated Effort (Hours)'])

    # Reorder columns to match the order used during training
    df_task = df_task[['Task_Type', 'Current_Status', 'Business_Impact', 'Days Until Deadline', 'Log_Estimated_Effort']]

    # Convert DataFrame to numpy array for model input
    input_data = df_task.values

    # Make predictions
    predictions = model.predict(input_data)
    predicted_classes = np.argmax(predictions, axis=1)

    return predicted_classes

logger = logging.getLogger(__name__)

# Task Prioritization View
def task_prioritization(request):
    if request.method == 'POST':
        task_form = TaskInputForm(request.POST)
        upload_form = UploadTaskForm(request.POST, request.FILES)

        # Manual task input
        if 'submit_task' in request.POST and task_form.is_valid():
            try:
                task_data = {
                    'task_title': task_form.cleaned_data['task_title'],
                    'task_type': dict(TASK_TYPE_CHOICES).get(task_form.cleaned_data['task_type']),  # Get display value
                    'current_status': dict(STATUS_CHOICES).get(task_form.cleaned_data['current_status']),  # Get display value
                    'business_impact': dict(BUSINESS_IMPACT_CHOICES).get(task_form.cleaned_data['business_impact']),  # Get display value
                    'estimated_effort': task_form.cleaned_data['estimated_effort'],
                    'priority_level': dict(PRIORITY_CHOICES).get(task_form.cleaned_data['priority']),  # Get display value
                    'deadline': task_form.cleaned_data['deadline'] if 'deadline' in task_form.cleaned_data else None
                }

                # Save the task to the ManualTask table
                ManualTask.objects.create(**task_data)

                # Create a DataFrame for prediction
                df_task = pd.DataFrame([task_data])
                prediction = predict_task_priority(df_task)

                # Generate visualizations from the ManualTask table
                manual_tasks = ManualTask.objects.all().values()  # Fetch all data from ManualTask
                manual_tasks_df = pd.DataFrame(manual_tasks)
                image4_base64, image5_base64, image6_base64 = generate_visualizations(manual_tasks_df)

                # Pass the predicted priority and visualizations to the template
                context = {
                    'priority': prediction[0],  # Ensure you handle predictions properly
                    'image4': image4_base64,
                    'image5': image5_base64,
                    'image6': image6_base64
                }

                return render(request, 'task_result.html', context)

            except Exception as e:
                logger.error(f"Error processing task input: {e}")
                return HttpResponseBadRequest(f"Error processing task input: {e}")
        
        # CSV file upload for bulk task input
        elif 'upload_csv' in request.POST and upload_form.is_valid():
            try:
                csv_file = request.FILES['csv_file']
                df = pd.read_csv(csv_file)

                # Handle date conversion for the 'deadline' column
                if 'deadline' in df.columns:
                    try:
                        df['deadline'] = pd.to_datetime(df['deadline'], dayfirst=True, errors='coerce')

                        # Check if there are any invalid dates
                        invalid_dates = df['deadline'].isna().sum()
                        if invalid_dates > 0:
                            return HttpResponseBadRequest(f"{invalid_dates} invalid date(s) found. Please check your CSV.")
                    except Exception as e:
                        return HttpResponseBadRequest(f"Error processing dates in CSV: {e}")

                # If 'deadline' column is missing, set default values
                if 'deadline' not in df.columns:
                    df['deadline'] = None

                # Make predictions
                predictions = predict_task_priority(df)
                df['Predicted Priority'] = predictions

                # Save uploaded tasks to the ManualTask table
                for _, row in df.iterrows():
                    ManualTask.objects.create(
                        task_title=row['task_title'],
                        task_type=row['task_type'],
                        current_status=row['current_status'],
                        business_impact=row['business_impact'],
                        estimated_effort=row['estimated_effort'],
                        priority_level=row['priority_level'],
                        deadline=row['deadline']
                    )

                # Generate visualizations from the ManualTask table
                manual_tasks = ManualTask.objects.all().values()
                manual_tasks_df = pd.DataFrame(manual_tasks)
                image4_base64, image5_base64, image6_base64 = generate_visualizations(manual_tasks_df)

                return render(request, 'task_result.html', {
                    'df': df.to_html(),
                    'image4': image4_base64,
                    'image5': image5_base64,
                    'image6': image6_base64
                })

            except Exception as e:
                return HttpResponseBadRequest(f"Error processing CSV upload: {e}")

    else:
        task_form = TaskInputForm()
        upload_form = UploadTaskForm()

    return render(request, 'task_prioritization.html', {'task_form': task_form, 'upload_form': upload_form})

# Helper function to generate visualizations as base64 images
def generate_visualizations(df_task):

    if df_task['priority_level'].isnull().sum() > 0:
        logger.warning("Missing priority levels in task data.")

    # Visualization 1: Distribution of Priority Levels
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    df_task['priority_level'].value_counts().plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Distribution of Task Priorities')
    ax1.set_xlabel('Priority')
    ax1.set_ylabel('Number of Tasks')

    # Save the figure to a BytesIO object
    buffer1 = BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    image4_base64 = base64.b64encode(buffer1.read()).decode('utf-8')
    plt.close(fig1)

    # Visualization 2: Business Impact vs Estimated Effort
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    df_task.groupby('business_impact')['estimated_effort'].mean().plot(kind='bar', ax=ax2, color='green')
    ax2.set_title('Average Estimated Effort by Business Impact')
    ax2.set_xlabel('Business Impact')
    ax2.set_ylabel('Average Estimated Effort')

    # Save the second figure to a BytesIO object
    buffer2 = BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    image5_base64 = base64.b64encode(buffer2.read()).decode('utf-8')
    plt.close(fig2)

    # Visualization 3: Estimated Effort by Priority Level
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    df_task.groupby('priority_level')['estimated_effort'].mean().plot(kind='bar', ax=ax3, color='green')
    ax3.set_title('Estimated Effort by Priority Level')
    ax3.set_xlabel('Priority Level')
    ax3.set_ylabel('Average Estimated Effort')

    # Save the third figure to a BytesIO object
    buffer3 = BytesIO()
    plt.savefig(buffer3, format='png')
    buffer3.seek(0)
    image6_base64 = base64.b64encode(buffer3.read()).decode('utf-8')
    plt.close(fig3)

    return image4_base64, image5_base64, image6_base64

def visualization(request):
    try:
        # Generate visualizations
        image1_base64, image2_base64, image3_base64 = generate_visualization_images()

        # Pass base64 images to the template
        context = {
            'image1': image1_base64,
            'image2': image2_base64,
            'image3': image3_base64,
        }
        return render(request, 'visualization.html', context)
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return render(request, 'visualization.html', {'error': str(e)})

# Handle unseen labels by returning a default or fallback index
def handle_unseen_labels(le, value):
    # Check if value is in classes_, if not, return a default class index (e.g., 0)
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        # You can decide on the default class, such as the first one (e.g., 'coa')
        return le.transform([le.classes_[0]])[0]
