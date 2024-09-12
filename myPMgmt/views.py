import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web rendering
from tasks.forms import TASK_TYPE_CHOICES, STATUS_CHOICES, PRIORITY_CHOICES, BUSINESS_IMPACT_CHOICES
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
import base64
from django.shortcuts import render
from tasks.models import Task, ManualTask
from tasks.forms import TaskInputForm, UploadTaskForm
import joblib
from django.conf import settings
from sklearn.preprocessing import LabelEncoder
from django.http import HttpResponseBadRequest

def generate_visualization_images():  # For visualization.html
    # Retrieve the task data from the database
    tasks = Task.objects.all().values()
    new_tasks_df = pd.DataFrame(tasks)

    # Visualization 1: Distribution of Priority Levels
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    new_tasks_df['priority_level'].value_counts().plot(kind='bar', color='skyblue', ax=ax1)
    ax1.set_title('Distribution of Priority Levels')
    ax1.set_xlabel('Priority Level')
    ax1.set_ylabel('Number of Tasks')
    ax1.set_xticks(ax1.get_xticks())
    plt.xticks(rotation=45)

    # Save the first plot to a buffer
    buffer1 = BytesIO()
    fig1.savefig(buffer1, format='png')
    buffer1.seek(0)
    image1_base64 = base64.b64encode(buffer1.read()).decode('utf-8')
    plt.close(fig1)

    # Visualization 2: Task Status vs. Business Impact
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    impact_status_counts = new_tasks_df.groupby(['current_status', 'business_impact']).size().unstack()
    impact_status_counts.plot(kind='bar', stacked=True, colormap='viridis', ax=ax2)
    ax2.set_title('Task Status vs. Business Impact')
    ax2.set_xlabel('Task Status')
    ax2.set_ylabel('Number of Tasks')
    plt.xticks(rotation=45)

    # Save the second plot to a buffer
    buffer2 = BytesIO()
    fig2.savefig(buffer2, format='png')
    buffer2.seek(0)
    image2_base64 = base64.b64encode(buffer2.read()).decode('utf-8')
    plt.close(fig2)

    # Visualization 3: Estimated Effort by Priority Level
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    new_tasks_df.boxplot(column='estimated_effort', by='priority_level', grid=False, patch_artist=True, ax=ax3)
    ax3.set_title('Estimated Effort by Priority Level')
    ax3.set_xlabel('Priority Level')
    ax3.set_ylabel('Estimated Effort (Hours)')
    plt.suptitle('')

    # Save the third plot to a buffer
    buffer3 = BytesIO()
    fig3.savefig(buffer3, format='png')
    buffer3.seek(0)
    image3_base64 = base64.b64encode(buffer3.read()).decode('utf-8')
    plt.close(fig3)

    return image1_base64, image2_base64, image3_base64

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
        
# Handle unseen labels dynamically
def handle_unseen_labels(label_encoder, values):
    unseen_values = [val for val in values if val not in label_encoder.classes_]
    if unseen_values:
        # Dynamically add the unseen labels to the encoder's classes
        label_encoder.classes_ = np.append(label_encoder.classes_, unseen_values)
    return label_encoder.transform(values)

# Function to load the model and make predictions
def predict_task_priority(df_task):

    # Load the model
    MODEL_PATH = 'tasks/model/final_rf_model.pkl'
    model = joblib.load(MODEL_PATH)

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
        'task_type': 'Task Type',
        'current_status': 'Current Status',
        'business_impact': 'Business Impact',
        'estimated_effort': 'Estimated Effort (Hours)',
        'priority': 'Impact_Effort',
        'deadline': 'Days Until Deadline'
    })

    # Encode categorical columns using LabelEncoders
    df_task['Task Type'] = handle_unseen_labels(task_type_le, df_task['Task Type'])
    df_task['Current Status'] = handle_unseen_labels(current_status_le, df_task['Current Status'])
    df_task['Business Impact'] = handle_unseen_labels(business_impact_le, df_task['Business Impact'])

    # Calculate Days Until Deadline
    def calculate_days_until_deadline(deadline):
        # Handle date parsing explicitly
        deadline = pd.to_datetime(deadline, format='%d/%m/%Y', dayfirst=True)
        return (deadline - pd.to_datetime(datetime.now())).days


    df_task['Days Until Deadline'] = df_task['Days Until Deadline'].apply(calculate_days_until_deadline)

    # Log-transform Estimated Effort (apply log1p)
    df_task['Log_Estimated_Effort'] = np.log1p(df_task['Estimated Effort (Hours)'])

    # Feature engineering - create the Impact_Effort feature
    df_task['Impact_Effort'] = df_task['Business Impact'] * df_task['Estimated Effort (Hours)']

    # Reorder columns to match the order used during training
    df_task = df_task[['Task Type', 'Current Status', 'Business Impact', 'Days Until Deadline', 'Impact_Effort', 'Log_Estimated_Effort']]

    # Make predictions
    predictions = model.predict(df_task)
    return predictions

# Task Prioritization View
def task_prioritization(request):
    if request.method == 'POST':
        task_form = TaskInputForm(request.POST)
        upload_form = UploadTaskForm(request.POST, request.FILES)

        # Manual task input
        if 'submit_task' in request.POST and task_form.is_valid():
            try:
                                # Get the displayed values (index 1) for choice fields
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
                    'priority': prediction[0],
                    'image4': image4_base64,
                    'image5': image5_base64,
                    'image6': image6_base64
                }

                return render(request, 'task_result.html', context)

            except Exception as e:
                return HttpResponseBadRequest(f"Error processing task input: {e}")


        # CSV file upload for bulk task input
        elif 'upload_csv' in request.POST and upload_form.is_valid():
            try:
                csv_file = request.FILES['csv_file']
                df = pd.read_csv(csv_file)

                # Handle date conversion for the 'deadline' column
                if 'deadline' in df.columns:
                    try:
                        # Explicitly handle dates in day/month/year format using dayfirst=True
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
                manual_tasks = ManualTask.objects.all().values()  # Fetch all data from ManualTask
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

def home(request):
    try:
        # Generate visualizations
        image4_base64, image5_base64, image6_base64 = generate_visualization_images()
        
        # Pass base64 images to the template
        context = {
            'image4': image4_base64,
            'image5': image5_base64,
            'image6': image6_base64,
        }
        return render(request, 'home.html', context)
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return render(request, 'home.html', {'error': str(e)})

# def generate_manualtask_visualizations():
#     # Fetch data from the ManualTask table
#     manual_tasks = ManualTask.objects.all().values()
#     manual_tasks_df = pd.DataFrame(manual_tasks)

#     # Visualization 1: Distribution of Priority Levels
#     fig1, ax1 = plt.subplots(figsize=(10, 6))
#     manual_tasks_df['priority_level'].value_counts().plot(kind='bar', ax=ax1, color='skyblue')
#     ax1.set_title('Distribution of Task Priorities (Manual Tasks)')
#     ax1.set_xlabel('Priority')
#     ax1.set_ylabel('Number of Tasks')

#     # Save the figure to a BytesIO object
#     buffer1 = BytesIO()
#     plt.savefig(buffer1, format='png')
#     buffer1.seek(0)
#     image7_base64 = base64.b64encode(buffer1.read()).decode('utf-8')
#     plt.close(fig1)

#     # Visualization 2: Business Impact vs Estimated Effort
#     fig2, ax2 = plt.subplots(figsize=(10, 6))
#     manual_tasks_df.groupby('business_impact')['estimated_effort'].mean().plot(kind='bar', ax=ax2, color='green')
#     ax2.set_title('Average Estimated Effort by Business Impact (Manual Tasks)')
#     ax2.set_xlabel('Business Impact')
#     ax2.set_ylabel('Average Estimated Effort')

#     # Save the second figure to a BytesIO object
#     buffer2 = BytesIO()
#     plt.savefig(buffer2, format='png')
#     buffer2.seek(0)
#     image8_base64 = base64.b64encode(buffer2.read()).decode('utf-8')
#     plt.close(fig2)

#     # Visualization 3: Estimated Effort by Priority Level
#     fig3, ax3 = plt.subplots(figsize=(10, 6))
#     manual_tasks_df.groupby('priority_level')['estimated_effort'].mean().plot(kind='bar', ax=ax3, color='blue')
#     ax3.set_title('Estimated Effort by Priority Level (Manual Tasks)')
#     ax3.set_xlabel('Priority Level')
#     ax3.set_ylabel('Average Estimated Effort')

#     # Save the third figure to a BytesIO object
#     buffer3 = BytesIO()
#     plt.savefig(buffer3, format='png')
#     buffer3.seek(0)
#     image9_base64 = base64.b64encode(buffer3.read()).decode('utf-8')
#     plt.close(fig3)

#     return image7_base64, image8_base64, image9_base64

# def manual_task_visualization(request):
#     try:
#         # Generate visualizations from ManualTask data
#         image7_base64, image8_base64, image9_base64 = generate_manualtask_visualizations()

#         # Pass base64 images to the template
#         context = {
#             'image7': image7_base64,
#             'image8': image8_base64,
#             'image9': image9_base64,
#         }
#         return render(request, 'manual_task_visualization.html', context)
#     except Exception as e:
#         print(f"Error generating visualizations: {e}")
#         return render(request, 'manual_task_visualization.html', {'error': str(e)})
