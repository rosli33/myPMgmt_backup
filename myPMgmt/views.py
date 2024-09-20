import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web rendering
from tasks.forms import TASK_TYPE_CHOICES, STATUS_CHOICES, PRIORITY_CHOICES, BUSINESS_IMPACT_CHOICES
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
import base64
import logging
from django.shortcuts import render
from tasks.models import Task
from django.conf import settings
# from tensorflow.keras.models import load_model
# Function to load the model and make predictions
# from tensorflow.keras.optimizers import Adam
  
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
        # Generate visualizations
        image4_base64, image5_base64, image6_base64 = generate_visualization_images()
        
        # Pass base64 images to the template
        context = {
            'image4': image4_base64,
            'image5': image5_base64,
            'image6': image6_base64,
        }
        return render(request, 'home.html', context)
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return render(request, 'home.html', {'error': str(e)})