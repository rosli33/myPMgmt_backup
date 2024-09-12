from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_file, name='upload'),  # Upload page
    path('success/', views.upload_success, name='upload_success'),  # Upload success page
]

