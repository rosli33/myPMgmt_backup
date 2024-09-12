"""
URL configuration for myPMgmt project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# myPMgmt/urls.py
from django.contrib import admin
from django.urls import path, include
from .views import home, visualization
# from .views import visualization  # Import your views
from . import views  # This will import views from the current app


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name='home'),  # Home page
    path('upload/', include('upload.urls')),  # Include the upload app
    path('visualization/', visualization, name='visualization'),  # New visualization page
    path('task_prioritization/', views.task_prioritization, name='task_prioritization'),
]






