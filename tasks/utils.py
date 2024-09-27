import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pandas as pd

def generate_visualizations(df):
    """Generate and return base64 encoded images for visualizations."""
    # Visualization 1: Distribution of Task Priorities
    img1 = BytesIO()
    plt.figure(figsize=(8, 6))
    df['priority_level'].value_counts().sort_index().plot(kind='bar', color='lightblue')
    plt.title('Distribution of Task Priorities')
    plt.xlabel('Priority Level')
    plt.ylabel('Number of Tasks')
    plt.tight_layout()
    plt.savefig(img1, format='png')
    plt.close()
    img1.seek(0)
    image1_base64 = base64.b64encode(img1.getvalue()).decode('utf-8')

    # Visualization 2: Business Impact vs Task Completion Status
    img2 = BytesIO()
    plt.figure(figsize=(8, 6))
    pd.crosstab(df['business_impact'], df['current_status']).plot(kind='bar', stacked=True, colormap='coolwarm')
    plt.title('Business Impact vs Task Completion Status')
    plt.xlabel('Business Impact')
    plt.ylabel('Number of Tasks')
    plt.tight_layout()
    plt.savefig(img2, format='png')
    plt.close()
    img2.seek(0)
    image2_base64 = base64.b64encode(img2.getvalue()).decode('utf-8')

    # Visualization 3: Distribution of Tasks by Task Type
    img3 = BytesIO()
    plt.figure(figsize=(8, 6))
    df['task_type'].value_counts().plot(kind='bar', color='lightgreen')
    plt.title('Distribution of Tasks by Task Type')
    plt.xlabel('Task Type')
    plt.ylabel('Number of Tasks')
    plt.tight_layout()
    plt.savefig(img3, format='png')
    plt.close()
    img3.seek(0)
    image3_base64 = base64.b64encode(img3.getvalue()).decode('utf-8')

    # Visualization 4: Estimated Effort by Task Completion Status
    img4 = BytesIO()
    plt.figure(figsize=(8, 6))
    df.groupby('current_status')['estimated_effort'].mean().plot(kind='bar', color='lightcoral')
    plt.title('Estimated Effort by Task Completion Status')
    plt.xlabel('Task Status')
    plt.ylabel('Average Estimated Effort')
    plt.tight_layout()
    plt.savefig(img4, format='png')
    plt.close()
    img4.seek(0)
    image4_base64 = base64.b64encode(img4.getvalue()).decode('utf-8')

    return image1_base64, image2_base64, image3_base64, image4_base64
