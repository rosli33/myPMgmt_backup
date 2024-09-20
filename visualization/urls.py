from django.urls import path
from . import views

urlpatterns = [
    path('analytics/', views.profile_analytics, name='visualization'),  # Adjust the path to be clear
]
