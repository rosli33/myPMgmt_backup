from django.shortcuts import render
from .models import Task
from .utils import predict_priority

def task_list(request):
    tasks = Task.objects.all()
    
    # Predict the priority for each task
    for task in tasks:
        task.priority = predict_priority(task)
    
    return render(request, 'tasks/task_list.html', {'tasks': tasks})
