import pygwalker as pyg
from django.shortcuts import render
from tasks.models import Task
import pandas as pd
import numpy as np
from django_pandas.io import read_frame

 # Profile Analytics for Task Data
def profile_analytics(request):
    # Query the task data from the database
    tasks_qs = Task.objects.all()  # Fetch all tasks from the database

    # Convert the queryset to a Pandas DataFrame
    tasks_df = read_frame(tasks_qs)

    # Basic custom spec for default chart
    spec = {
        "mark": "bar",  # Default chart type
        "encoding": {
            "x": {"field": "task_type", "type": "nominal"},  # X-axis: task_type
            "y": {"field": "estimated_effort", "type": "quantitative"},  # Y-axis: estimated_effort
            "color": {"field": "priority_level", "type": "nominal"},  # Color by priority_level
            "size": {"field": "business_impact", "type": "quantitative"},  # Size by business impact
        }
    }

    # Walk the data with the provided custom spec
    walker = pyg.walk(tasks_df, spec=spec)

    # Convert the Pygwalker object to HTML to embed in the template
    walker_html = walker.to_html()

    # Render the 'visualization.html' template and pass the generated HTML
    return render(request, 'visualization.html', {'pygwalker': walker_html})
 
 # def generate_manualtask_visualizations():
#     # Fetch data from the ManualTask table
#     manual_tasks = ManualTask.objects.all().values()
#     manual_tasks_df = pd.DataFrame(manual_tasks)

#     # Visualization 1: Distribution of Priority Levels
#     fig1, ax1 = plt.subplots(figsize=(10, 6))
#     manual_tasks_df['priority_level'].value_counts().plot(kind='bar', ax=ax1, color='skyblue')
#     ax1.set_title('Distribution of Task Priorities (Manual Tasks)')
#     ax1.set_xlabel('Priority')
#     ax1.set_ylabel('Number of Tasks')

#     # Save the figure to a BytesIO object
#     buffer1 = BytesIO()
#     plt.savefig(buffer1, format='png')
#     buffer1.seek(0)
#     image7_base64 = base64.b64encode(buffer1.read()).decode('utf-8')
#     plt.close(fig1)

#     # Visualization 2: Business Impact vs Estimated Effort
#     fig2, ax2 = plt.subplots(figsize=(10, 6))
#     manual_tasks_df.groupby('business_impact')['estimated_effort'].mean().plot(kind='bar', ax=ax2, color='green')
#     ax2.set_title('Average Estimated Effort by Business Impact (Manual Tasks)')
#     ax2.set_xlabel('Business Impact')
#     ax2.set_ylabel('Average Estimated Effort')

#     # Save the second figure to a BytesIO object
#     buffer2 = BytesIO()
#     plt.savefig(buffer2, format='png')
#     buffer2.seek(0)
#     image8_base64 = base64.b64encode(buffer2.read()).decode('utf-8')
#     plt.close(fig2)

#     # Visualization 3: Estimated Effort by Priority Level
#     fig3, ax3 = plt.subplots(figsize=(10, 6))
#     manual_tasks_df.groupby('priority_level')['estimated_effort'].mean().plot(kind='bar', ax=ax3, color='blue')
#     ax3.set_title('Estimated Effort by Priority Level (Manual Tasks)')
#     ax3.set_xlabel('Priority Level')
#     ax3.set_ylabel('Average Estimated Effort')

#     # Save the third figure to a BytesIO object
#     buffer3 = BytesIO()
#     plt.savefig(buffer3, format='png')
#     buffer3.seek(0)
#     image9_base64 = base64.b64encode(buffer3.read()).decode('utf-8')
#     plt.close(fig3)

#     return image7_base64, image8_base64, image9_base64

# def manual_task_visualization(request):
#     try:
#         # Generate visualizations from ManualTask data
#         image7_base64, image8_base64, image9_base64 = generate_manualtask_visualizations()

#         # Pass base64 images to the template
#         context = {
#             'image7': image7_base64,
#             'image8': image8_base64,
#             'image9': image9_base64,
#         }
#         return render(request, 'manual_task_visualization.html', context)
#     except Exception as e:
#         print(f"Error generating visualizations: {e}")
#         return render(request, 'manual_task_visualization.html', {'error': str(e)})

# def generate_visualization_images():  # For visualization.html
#     # Retrieve the task data from the database
#     tasks = Task.objects.all().values()
#     new_tasks_df = pd.DataFrame(tasks)

#     # Visualization 1: Distribution of Priority Levels
#     fig1, ax1 = plt.subplots(figsize=(10, 6))
#     new_tasks_df['priority_level'].value_counts().plot(kind='bar', color='skyblue', ax=ax1)
#     ax1.set_title('Distribution of Priority Levels')
#     ax1.set_xlabel('Priority Level')
#     ax1.set_ylabel('Number of Tasks')
#     ax1.set_xticks(ax1.get_xticks())
#     plt.xticks(rotation=45)

#     # Save the first plot to a buffer
#     buffer1 = BytesIO()
#     fig1.savefig(buffer1, format='png')
#     buffer1.seek(0)
#     image1_base64 = base64.b64encode(buffer1.read()).decode('utf-8')
#     plt.close(fig1)

#     # Visualization 2: Task Status vs. Business Impact
#     fig2, ax2 = plt.subplots(figsize=(10, 6))
#     impact_status_counts = new_tasks_df.groupby(['current_status', 'business_impact']).size().unstack()
#     impact_status_counts.plot(kind='bar', stacked=True, colormap='viridis', ax=ax2)
#     ax2.set_title('Task Status vs. Business Impact')
#     ax2.set_xlabel('Task Status')
#     ax2.set_ylabel('Number of Tasks')
#     plt.xticks(rotation=45)

#     # Save the second plot to a buffer
#     buffer2 = BytesIO()
#     fig2.savefig(buffer2, format='png')
#     buffer2.seek(0)
#     image2_base64 = base64.b64encode(buffer2.read()).decode('utf-8')
#     plt.close(fig2)

#     # Visualization 3: Estimated Effort by Priority Level
#     fig3, ax3 = plt.subplots(figsize=(10, 6))
#     new_tasks_df.boxplot(column='estimated_effort', by='priority_level', grid=False, patch_artist=True, ax=ax3)
#     ax3.set_title('Estimated Effort by Priority Level')
#     ax3.set_xlabel('Priority Level')
#     ax3.set_ylabel('Estimated Effort (Hours)')
#     plt.suptitle('')

#     # Save the third plot to a buffer
#     buffer3 = BytesIO()
#     fig3.savefig(buffer3, format='png')
#     buffer3.seek(0)
#     image3_base64 = base64.b64encode(buffer3.read()).decode('utf-8')
#     plt.close(fig3)

#     return image1_base64, image2_base64, image3_base64