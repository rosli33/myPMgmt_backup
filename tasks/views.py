import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tasks.forms import TASK_TYPE_CHOICES, STATUS_CHOICES, BUSINESS_IMPACT_CHOICES
from io import BytesIO
import base64
from django.shortcuts import render, redirect
from tasks.models import Task, ManualTask
from tasks.forms import TaskInputForm, UploadTaskForm
from django.conf import settings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
from django.http import HttpResponse, FileResponse
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import os
from django.contrib import messages  # Import messages framework
from .utils import generate_visualizations  # Assuming you have these functions
import csv
from io import BytesIO
from django.http import HttpResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.units import inch
from reportlab.lib import utils

# Predict Task Priority
def predict_task_priority(df_task):
    # Load the ANN model
    MODEL_PATH = os.path.join(settings.BASE_DIR, 'tasks', 'model', 'final_ann_model.h5')
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
                    'task_type': dict(TASK_TYPE_CHOICES).get(task_form.cleaned_data['task_type']),
                    'current_status': dict(STATUS_CHOICES).get(task_form.cleaned_data['current_status']),
                    'business_impact': dict(BUSINESS_IMPACT_CHOICES).get(task_form.cleaned_data['business_impact']),
                    'estimated_effort': task_form.cleaned_data['estimated_effort'],
                    'deadline': task_form.cleaned_data['deadline']
                }

                # Create a DataFrame for prediction
                df_task = pd.DataFrame([task_data])
                prediction = predict_task_priority(df_task)

                # Ensure priority_level is a native Python int
                task_data['priority_level'] = int(prediction[0])

                # Save the task to the ManualTask table
                ManualTask.objects.create(**task_data)

                # Store priority as int in session
                request.session['priority'] = int(prediction[0])
                request.session['task_title'] = task_data['task_title']
                request.session['task_type'] = task_data['task_type']

                # Add insights
                business_impact = task_data['business_impact']
                insight = (
                    "High business impact tasks tend to receive higher priority."
                    if business_impact == 'High'
                    else "Medium business impact tasks often have a balanced priority."
                    if business_impact == 'Medium'
                    else "Low business impact tasks may have lower priority."
                )
                status_insight = (
                    "Tasks in progress may receive higher attention for completion."
                    if task_data['current_status'] == 'In Progress'
                    else "Tasks in 'To Do' are scheduled but may be prioritized based on urgency."
                    if task_data['current_status'] == 'To Do'
                    else "Completed tasks typically have the lowest priority."
                )
                effort_insight = (
                    "Tasks with higher estimated effort may need splitting into smaller sub-tasks."
                    if task_data['estimated_effort'] > 20
                    else "Medium-sized tasks are manageable and may receive higher priority."
                    if task_data['estimated_effort'] > 10
                    else "Small tasks can be quick wins and might be prioritized for faster completion."
                )

                request.session['insight'] = insight
                request.session['status_insight'] = status_insight
                request.session['effort_insight'] = effort_insight

                return redirect('task_result')  # Redirect to result page
            except Exception as e:
                logger.error(f"Error processing task input: {e}")
                messages.error(request, f"Error processing task input: {e}")
                return redirect('task_prioritization')

        # CSV file upload for bulk task input
        elif 'upload_csv' in request.POST and upload_form.is_valid():
            try:
                csv_file = request.FILES['csv_file']
                df = pd.read_csv(csv_file)

                # Handle date conversion for the 'deadline' column
                if 'deadline' in df.columns:
                    df['deadline'] = pd.to_datetime(df['deadline'], dayfirst=True, errors='coerce')

                # If 'deadline' column is missing, set default values
                if 'deadline' not in df.columns:
                    df['deadline'] = None

                # Make predictions
                predictions = predict_task_priority(df)
                df['Predicted Priority'] = predictions

                # Convert deadline to string to be stored in session
                df['deadline'] = df['deadline'].dt.strftime('%Y-%m-%d')

                # Convert data to list of dictionaries and store in session
                request.session['df'] = df.to_dict('records')

                return redirect('task_result')  # Redirect to task result page
            except Exception as e:
                logger.error(f"Error processing CSV upload: {e}")
                messages.error(request, f"Error processing CSV upload: {e}")
                return redirect('task_prioritization')

    else:
        task_form = TaskInputForm()
        upload_form = UploadTaskForm()

    return render(request, 'task_prioritization.html', {'task_form': task_form, 'upload_form': upload_form})

def generate_insight(task_data):
    # Business Impact Insight
    business_impact = task_data['business_impact']
    if business_impact == 'High':
        business_impact_insight = "High business impact tasks tend to receive higher priority."
    elif business_impact == 'Medium':
        business_impact_insight = "Medium business impact tasks often have a balanced priority."
    else:
        business_impact_insight = "Low business impact tasks may have lower priority unless other factors increase importance."

    # Task Status Insight
    current_status = task_data['current_status']
    if current_status == 'In Progress':
        status_insight = "Tasks in progress may receive higher attention for completion."
    elif current_status == 'To Do':
        status_insight = "Tasks in 'To Do' are scheduled but may be prioritized based on urgency."
    else:
        status_insight = "Completed tasks typically have the lowest priority."

    # Estimated Effort Insight
    effort = task_data['estimated_effort']
    if effort > 20:
        effort_insight = "Tasks with higher estimated effort may need splitting into smaller sub-tasks."
    elif effort > 10:
        effort_insight = "Medium-sized tasks are manageable and may receive higher priority."
    else:
        effort_insight = "Small tasks can be quick wins and might be prioritized for faster completion."

    return {
        'business_impact': business_impact_insight,
        'status': status_insight,
        'effort': effort_insight
}
def generate_visualizations(df):
    """Generate and return base64 encoded images for visualizations."""
    # Visualization 1: Distribution of Task Priorities
    img1 = BytesIO()
    plt.figure(figsize=(8, 6))
    df['priority_level'].value_counts().sort_index().plot(kind='bar', color='lightblue')
    plt.title('Distribution of Task Priorities')
    plt.xlabel('Priority Level')
    plt.ylabel('Number of Tasks')
    plt.tight_layout()
    plt.savefig(img1, format='png')
    plt.close()
    img1.seek(0)
    image1_base64 = base64.b64encode(img1.getvalue()).decode('utf-8')

    # Visualization 2: Business Impact vs Task Completion Status
    img2 = BytesIO()
    plt.figure(figsize=(8, 6))
    pd.crosstab(df['business_impact'], df['current_status']).plot(kind='bar', stacked=True, colormap='coolwarm')
    plt.title('Business Impact vs Task Completion Status')
    plt.xlabel('Business Impact')
    plt.ylabel('Number of Tasks')
    plt.tight_layout()
    plt.savefig(img2, format='png')
    plt.close()
    img2.seek(0)
    image2_base64 = base64.b64encode(img2.getvalue()).decode('utf-8')

    # Visualization 3: Distribution of Tasks by Task Type
    img3 = BytesIO()
    plt.figure(figsize=(8, 6))
    df['task_type'].value_counts().plot(kind='bar', color='lightgreen')
    plt.title('Distribution of Tasks by Task Type')
    plt.xlabel('Task Type')
    plt.ylabel('Number of Tasks')
    plt.tight_layout()
    plt.savefig(img3, format='png')
    plt.close()
    img3.seek(0)
    image3_base64 = base64.b64encode(img3.getvalue()).decode('utf-8')

    # Visualization 4: Estimated Effort by Task Completion Status
    img4 = BytesIO()
    plt.figure(figsize=(8, 6))
    df.groupby('current_status')['estimated_effort'].mean().plot(kind='bar', color='lightcoral')
    plt.title('Estimated Effort by Task Completion Status')
    plt.xlabel('Task Status')
    plt.ylabel('Average Estimated Effort')
    plt.tight_layout()
    plt.savefig(img4, format='png')
    plt.close()
    img4.seek(0)
    image4_base64 = base64.b64encode(img4.getvalue()).decode('utf-8')

    return image1_base64, image2_base64, image3_base64, image4_base64

def visualization(request):
    try:
        # Fetch all tasks from the ManualTask table
        manual_tasks = ManualTask.objects.all().values()
        
        if manual_tasks.exists():
            manual_tasks_df = pd.DataFrame(manual_tasks)
            
            # Generate visualizations based on task data
            image1_base64, image2_base64, image3_base64 = generate_visualizations(manual_tasks_df)
            
            # Pass the images to the template
            context = {
                'image1': image1_base64,
                'image2': image2_base64,
                'image3': image3_base64,
            }
        else:
            context = {
                'error': 'No tasks found to generate visualizations.'
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

# Add filtering and sorting in the task_result view
def task_result(request):
    context = {}

    # For manual task input (display insights)
    if 'priority' in request.session:
        priority = request.session.get('priority')
        task_title = request.session.get('task_title')
        task_type = request.session.get('task_type')
        insight = request.session.get('insight', "")
        status_insight = request.session.get('status_insight', "")
        effort_insight = request.session.get('effort_insight', "")

        # Clear session data after retrieving to avoid showing it again later
        del request.session['priority']
        del request.session['task_title']
        del request.session['task_type']
        del request.session['insight']
        del request.session['status_insight']
        del request.session['effort_insight']

        # Set the context to display the manual task result and insights
        context.update({
            'priority': priority,
            'task_title': task_title,
            'task_type': task_type,
            'insight': insight,
            'status_insight': status_insight,
            'effort_insight': effort_insight,
            'is_manual_task': True  # A flag to distinguish manual task insights from CSV data
        })

    # For uploaded file and visualization (display filters, table, and visualizations)
    elif 'df' in request.session:
        df = pd.DataFrame(request.session.get('df'))
        print("Data in Session (Before Filtering):", df.head())
        print("Available Columns in df:", df.columns)

        # Get filter values from the request
        task_type = request.GET.get('task_type', '')
        priority = request.GET.get('priority', '')
        status = request.GET.get('status', '')
        sort_by = request.GET.get('sort_by', 'deadline')

        print("Filters Received: Task Type:", task_type, "Priority:", priority, "Status:", status)

        # Apply filtering logic
        task_type_mapping = {
            'coa': 'Certificate of Acceptance (CoA)',
            'portal': 'Portal',
            'auth_control': 'User Authentication and Access Control',
            'finalization_integration': 'Finalization and Integration',
            'crm': 'Customer Relationship Management (CRM)',
            'data_storage': 'Data Management and Storage',
            'comm_collab': 'Communication and Collaboration Tools',
            'workflow_management': 'Workflow Management',
            'contract_documentation': 'Contract Documentation',
            'tok': 'Transfer of Knowledge (TOK)',
            'training_management': 'Training Management',
            'hrm': 'Human Resource Management (HRM)',
            'inventory_management': 'Inventory Management',
            'project_management': 'Project Management',
            'reporting_analytics': 'Reporting and Analytics',
            'tot': 'Transfer of Technology (TOT)',
            'ea': 'Enterprise Architecture (EA)',
            'payment_gateway': 'Payment Gateway Integration',
            'change_management': 'Change Management',
            'notification_system': 'Notification System',
            'audit_compliance': 'Audit and Compliance',
            'devops': 'DevOps',
            'management_plans': 'Management Plans',
            'monitoring_reports': 'Monitoring & Control Reports',
            'ui_ux': 'User Interface and User Experience (UI/UX)',
            'data_lake_warehouse': 'Data Lake & Data Warehouse Operation',
            'data_services': 'Data Services & Management',
            'closure_report': 'Project Closure Report',
            'bcp': 'Business Continuity Plan (BCP)',
            'office_renovation': 'Office Renovation',
            'portal_lms': 'Portal Learning Management System (LMS)',
            'pmis': 'Project Management Information System (PMIS)'
        }

        # Apply task type filter
        if task_type in task_type_mapping:
            df = df[df['task_type'] == task_type_mapping[task_type]]
            print(f"Filtered by Task Type: {task_type_mapping[task_type]}")

        # Mapping for priority filter: 0 = Low, 1 = Medium, 2 = High
        priority_mapping = {
            '0': 'Low',
            '1': 'Medium',
            '2': 'High'
        }

        # Apply priority filter if it exists
        if priority in priority_mapping:
            df = df[df['priority_level'] == priority_mapping[priority]]
            print(f"Filtered by Priority Level: {priority_mapping[priority]}")

        # Apply status filter if it exists
        status_mapping = {
            'todo': 'To Do',
            'progress': 'In Progress',
            'completed': 'Completed'
        }
        if status in status_mapping:
            df = df[df['current_status'] == status_mapping[status]]
            print(f"Filtered by Status: {status_mapping[status]}")

        # Apply sorting
        df = df.sort_values(by=[sort_by])

        # Check if the DataFrame is empty after filtering
        if df.empty:
            context['no_data_message'] = "No task data available for the selected filters."
        else:
            # Generate visualizations if data exists
            image1_base64, image2_base64, image3_base64, image4_base64 = generate_visualizations(df)

            context.update({
                'df': df.to_html(classes='table table-striped'),
                'image1': image1_base64,
                'image2': image2_base64,
                'image3': image3_base64,
                'image4': image4_base64,
                'is_manual_task': False  # A flag to distinguish this from manual task insights
            })

    else:
        context['no_data_message'] = "No data uploaded. Please upload a CSV file."

    return render(request, 'task_result.html', context)

def export_csv(request):
    df = pd.DataFrame(request.session.get('df'))
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="filtered_tasks.csv"'

    writer = csv.writer(response)
    writer.writerow(df.columns)
    for _, row in df.iterrows():
        writer.writerow(row)

    return response

def export_pdf(request):
    # Fetch the filtered data from session
    df = pd.DataFrame(request.session.get('df'))

    buffer = BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=landscape(A4))

    # Get a sample stylesheet for paragraphs
    styles = getSampleStyleSheet()

    # Modify the data to wrap text in task_title and task_type columns
    data = [df.columns.tolist()]  # Add header row
    for row in df.values.tolist():
        row[0] = Paragraph(row[0], styles['BodyText'])  # Wrap task_title
        row[1] = Paragraph(row[1], styles['BodyText'])  # Wrap task_type
        data.append(row)

    # Set column widths
    col_widths = [
        2.0 * inch,  # task_title column
        2.0 * inch,  # task_type column
        1.0 * inch,  # current_status
        1.0 * inch,  # business_impact
        0.8 * inch,  # estimated_effort
        0.8 * inch,  # priority_level
        1.0 * inch,  # deadline
        0.8 * inch,  # Predicted Priority
    ]

    # Create the table with styles
    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),  # Ensure top-aligned text
        ('ALIGN', (0, 0), (1, -1), 'LEFT'),  # Left-align task_title and task_type for readability
    ]))

    # Build the PDF
    elements = []
    elements.append(Paragraph("Filtered Task Results", styles['Title']))
    elements.append(table)
    pdf.build(elements)

    # Get the PDF bytes and return response
    buffer.seek(0)
    response = HttpResponse(buffer, content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="filtered_tasks.pdf"'

    return response

# Helper functions for insights
def get_insight(business_impact):
    if business_impact == 'High':
        return "High business impact tasks tend to receive higher priority."
    elif business_impact == 'Medium':
        return "Medium business impact tasks often have a balanced priority."
    else:
        return "Low business impact tasks may have lower priority unless other factors increase importance."


def get_status_insight(current_status):
    if current_status == 'In Progress':
        return "Tasks in progress may receive higher attention for completion."
    elif current_status == 'To Do':
        return "Tasks in 'To Do' are scheduled but may be prioritized based on urgency."
    else:
        return "Completed tasks typically have the lowest priority."


def get_effort_insight(effort):
    if effort > 20:
        return "Tasks with higher estimated effort may need splitting into smaller sub-tasks."
    elif effort > 10:
        return "Medium-sized tasks are manageable and may receive higher priority."
    else:
        return "Small tasks can be quick wins and might be prioritized for faster completion."