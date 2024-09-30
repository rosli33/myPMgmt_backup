from django.urls import path
from . import views  # assuming your views are in the current folder

urlpatterns = [
    # Other URLs
    path('upload/', views.upload_file, name='upload'),  # This ensures that 'upload' is a valid URL name
    path('upload/success/', views.upload_success, name='upload_success'),  # Ensure this is also included
]
