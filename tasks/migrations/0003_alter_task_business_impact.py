# Generated by Django 5.1.1 on 2024-09-06 04:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tasks', '0002_remove_task_priority_task_assignee_name_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='task',
            name='business_impact',
            field=models.CharField(choices=[('Low', 'Low'), ('Medium', 'Medium'), ('High', 'High')], default='Medium', max_length=6),
        ),
    ]
