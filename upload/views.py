import csv
from datetime import datetime
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse
from tasks.models import Task
from django.db.utils import IntegrityError
from django.contrib import messages
import pandas as pd
from .forms import UploadFileForm

# Helper function to convert date format
def convert_date(date_str):
    try:
        return datetime.strptime(date_str, '%d/%m/%Y').strftime('%Y-%m-%d')
    except ValueError:
        return None

def validate_csv_data(df):
    errors = []
    
    # Checking for missing columns
    required_columns = ['Task ID', 'Task Title', 'Task Type', 'Creation Date', 'Deadline', 'Current Status', 'Priority Level', 'Resource_ID', 'Assignee Name', 'Estimated Effort (Hours)', 'Business Impact']
    for col in required_columns:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
    
    # Validate date fields
    df['Creation Date'] = pd.to_datetime(df['Creation Date'], format='%d/%m/%Y', errors='coerce')
    df['Deadline'] = pd.to_datetime(df['Deadline'], format='%d/%m/%Y', errors='coerce')
    
    if df['Creation Date'].isnull().any():
        errors.append("Some rows have invalid 'Creation Date'. Expected format: DD/MM/YYYY")
    if df['Deadline'].isnull().any():
        errors.append("Some rows have invalid 'Deadline'. Expected format: DD/MM/YYYY")
    
    # Validate numeric fields
    if df['Estimated Effort (Hours)'].apply(pd.to_numeric, errors='coerce').isnull().any():
        errors.append("Some rows have invalid 'Estimated Effort'. Expected a numeric value.")
    
    return errors

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            try:
                df = pd.read_csv(file)

                # Validate CSV data
                errors = validate_csv_data(df)
                if errors:
                    for error in errors:
                        messages.error(request, error)
                    return render(request, 'upload.html', {'form': form})

                # Counters for feedback
                created_count = 0
                updated_count = 0

                # Insert tasks into the database
                for _, row in df.iterrows():
                    try:
                        task, created = Task.objects.update_or_create(
                            task_id=row['Task ID'],
                            defaults={
                                'task_title': row['Task Title'],
                                'task_type': row['Task Type'],
                                'creation_date': row['Creation Date'],
                                'deadline': row['Deadline'],
                                'current_status': row['Current Status'],
                                'priority_level': row['Priority Level'],
                                'resource_id': row['Resource_ID'],
                                'assignee_name': row['Assignee Name'],
                                'estimated_effort': row['Estimated Effort (Hours)'],
                                'business_impact': row['Business Impact'],
                            }
                        )
                        if created:
                            created_count += 1
                        else:
                            updated_count += 1
                    except IntegrityError as e:
                        messages.error(request, f"Error inserting row with Task ID {row['Task ID']}: {e}")
                        continue

                messages.success(request, f"File uploaded successfully. {created_count} tasks created, {updated_count} tasks updated.")
                return HttpResponseRedirect(reverse('upload_success'))

            except Exception as e:
                messages.error(request, f"Error reading the file: {e}")
                return render(request, 'upload.html', {'form': form})
        else:
            messages.error(request, 'Form is not valid. Please upload a valid CSV file.')
    else:
        form = UploadFileForm()

    return render(request, 'upload.html', {'form': form})


# Add the missing upload_success view
def upload_success(request):
    return render(request, 'upload_success.html')
