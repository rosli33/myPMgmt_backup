import csv
from datetime import datetime
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse
from .forms import UploadFileForm
from tasks.models import Task
from django.db.utils import IntegrityError 

# Helper function to convert date format
def convert_date(date_str):
    try:
        return datetime.strptime(date_str, '%d/%m/%Y').strftime('%Y-%m-%d')
    except ValueError:
        return None

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            decoded_file = file.read().decode('utf-8').splitlines()
            reader = csv.DictReader(decoded_file)

            for row in reader:
                # Convert the date fields to the correct format
                creation_date = convert_date(row['Creation Date'])
                deadline = convert_date(row['Deadline'])

                try:
                    Task.objects.update_or_create(
                        task_id=row['Task ID'],
                        defaults={
                            'task_title': row['Task Title'],
                            'task_type': row['Task Type'],
                            'creation_date': creation_date,  # Use converted date
                            'deadline': deadline,              # Use converted date
                            'current_status': row['Current Status'],
                            'priority_level': row['Priority Level'],
                            'resource_id': row['Resource_ID'],
                            'assignee_name': row['Assignee Name'],
                            'estimated_effort': row['Estimated Effort (Hours)'],
                            'business_impact': row['Business Impact'],
                        }
                    )
                except IntegrityError as e:
                    print(f"Error inserting row: {row['Task ID']} - {e}")

            return HttpResponseRedirect(reverse('upload_success'))
    else:
        form = UploadFileForm()
    
    return render(request, 'upload.html', {'form': form})

# Add the missing upload_success view
def upload_success(request):
    return render(request, 'upload_success.html')