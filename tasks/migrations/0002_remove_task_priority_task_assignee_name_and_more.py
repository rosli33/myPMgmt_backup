# Generated by Django 5.1.1 on 2024-09-06 04:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tasks', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='task',
            name='priority',
        ),
        migrations.AddField(
            model_name='task',
            name='assignee_name',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AddField(
            model_name='task',
            name='priority_level',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='task',
            name='resource_id',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='task',
            name='task_id',
            field=models.CharField(default='default_id', max_length=20, unique=True),
        ),
        migrations.AddField(
            model_name='task',
            name='task_title',
            field=models.CharField(default='Untitled Task', max_length=255),
        ),
        migrations.AlterField(
            model_name='task',
            name='current_status',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
    ]
