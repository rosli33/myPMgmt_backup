from django.urls import path
from . import views

urlpatterns = [
    path('file/', views.upload_file, name='upload'),  # Changed 'upload/' to 'file/' for clarity
    path('success/', views.upload_success, name='upload_success'),
]
