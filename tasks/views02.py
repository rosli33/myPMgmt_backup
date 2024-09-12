from django.shortcuts import render, redirect
from .models import Task
from .utils import predict_priority
from datetime import date

def home(request):
    return render(request, 'index.html')  # Since it's directly under 'templates/', you don't need the app prefix

def task_list(request):
    # Fetch all tasks from the database
    tasks = Task.objects.all()
    
    # Pass the tasks to the template
    return render(request, 'tasks/task_list.html', {'tasks': tasks})

def create_task(request):
    if request.method == 'POST':
        # Extract form data from the request
        task_type = request.POST['task_type']
        current_status = request.POST['current_status']
        business_impact = request.POST['business_impact']
        estimated_effort = request.POST['estimated_effort']
        deadline = request.POST.get('deadline')  # Get deadline, but it might be None

        # Convert deadline to a date object if it's provided
        if deadline:
            deadline = date.fromisoformat(deadline)
            days_until_deadline = (deadline - date.today()).days
        else:
            days_until_deadline = None  # Handle case when deadline is not provided

        # Create the task object
        task = Task(
            task_type=task_type,
            current_status=current_status,
            business_impact=business_impact,
            estimated_effort=estimated_effort,
            deadline=deadline
        )

        # Save the task and predict priority
        task.priority = predict_priority(task)
        task.save()

        return redirect('task_list')
    return render(request, 'tasks/create_task.html')
