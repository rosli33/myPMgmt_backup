from django.urls import path
from . import views

urlpatterns = [
    path('allocation/', views.resource_allocation, name='resource_allocation'),  # resource allocation path
]
