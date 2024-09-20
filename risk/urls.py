from django.urls import path
from . import views

urlpatterns = [
    path('risk_management/', views.risk_management, name='risk_management'),
]
