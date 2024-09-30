import pandas as pd
import numpy as np
from datetime import datetime
from django.shortcuts import render
from tasks.models import Task
from django.conf import settings
  
# Handle unseen labels by returning a default or fallback index
def handle_unseen_labels(le, value):
    # Check if value is in classes_, if not, return a default class index (e.g., 0)
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        # You can decide on the default class, such as the first one (e.g., 'coa')
        return le.transform([le.classes_[0]])[0]

def home(request):
    try:
        context = {}

        return render(request, 'home.html', context)
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return render(request, 'home.html', {'error': str(e)})

