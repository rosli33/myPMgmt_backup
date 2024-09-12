from django.urls import path
from . import views02

urlpatterns = [
    path('', views02.home, name='home'),  # This serves the main page at the root URL
    path('', views02.task_list, name='task_list'),  # The task list view
    path('create/', views02.create_task, name='create_task'),  # Assuming you also have a task creation view
]
