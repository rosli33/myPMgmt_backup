from django.urls import path
from . import views

urlpatterns = [
    path('task_prioritization/', views.task_prioritization, name='task_prioritization'),
    path('result/', views.task_result, name='task_result'),
    path('tasks/export_csv/', views.export_csv, name='export_csv'),  # Export CSV functionality
    path('tasks/export_pdf/', views.export_pdf, name='export_pdf'),  # Export PDF functionality
    path('export_pdf/', views.export_pdf, name='export_pdf'),
]

