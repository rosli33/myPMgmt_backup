from django.urls import path
from . import views

urlpatterns = [
    path('task_prioritization/', views.task_prioritization, name='task_prioritization'),
    path('result/', views.generate_visualizations, name='task_visualizaton'),
]

