from django import forms
from django.db import models

STATUS_CHOICES = [
    ('todo', 'To Do'),
    ('progress', 'In Progress'),
    ('completed', 'Completed'),
]

PRIORITY_CHOICES = [
    ('low', 'Low'),
    ('medium', 'Medium'),
    ('high', 'High'),
]

TASK_TYPE_CHOICES = [
    ('coa', 'Certificate of Acceptance (CoA)'),
    ('portal', 'Portal'),
    ('auth_control', 'User Authentication and Access Control'),
    ('finalization_integration', 'Finalization and Integration'),
    ('crm', 'Customer Relationship Management (CRM)'),
    ('data_storage', 'Data Management and Storage'),
    ('comm_collab', 'Communication and Collaboration Tools'),
    ('workflow_management', 'Workflow Management'),
    ('contract_documentation', 'Contract Documentation'),
    ('tok', 'Transfer of Knowledge (TOK)'),
    ('training_management', 'Training Management'),
    ('hrm', 'Human Resource Management (HRM)'),
    ('inventory_management', 'Inventory Management'),
    ('project_management', 'Project Management'),
    ('reporting_analytics', 'Reporting and Analytics'),
    ('tot', 'Transfer of Technology (TOT)'),
    ('ea', 'Enterprise Architecture (EA)'),
    ('payment_gateway', 'Payment Gateway Integration'),
    ('change_management', 'Change Management'),
    ('notification_system', 'Notification System'),
    ('audit_compliance', 'Audit and Compliance'),
    ('devops', 'DevOps'),
    ('management_plans', 'Management Plans'),
    ('monitoring_reports', 'Monitoring & Control Reports'),
    ('ui_ux', 'User Interface and User Experience (UI/UX)'),
    ('data_lake_warehouse', 'Data Lake & Data Warehouse Operation'),
    ('data_services', 'Data Services & Management'),
    ('closure_report', 'Project Closure Report'),
    ('bcp', 'Business Continuity Plan (BCP)'),
    ('office_renovation', 'Office Renovation'),
    ('portal_lms', 'Portal Learning Management System (LMS)'),
    ('pmis', 'Project Management Information System (PMIS)'),
    ]

BUSINESS_IMPACT_CHOICES = [
        ('Low', 'Low'),
        ('Medium', 'Medium'),
        ('High', 'High'),
    ]

class TaskInputForm(forms.Form):
    task_title = forms.CharField(max_length=200, widget=forms.TextInput(attrs={'class': 'form-control'}))
    task_type = forms.ChoiceField(choices=TASK_TYPE_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
    current_status = forms.ChoiceField(choices=STATUS_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
    business_impact = forms.ChoiceField(choices=BUSINESS_IMPACT_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
    estimated_effort = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    priority = forms.ChoiceField(choices=PRIORITY_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
    deadline = forms.DateField(widget=forms.SelectDateWidget(attrs={'class': 'form-control'}))

class UploadTaskForm(forms.Form):
    csv_file = forms.FileField(widget=forms.FileInput(attrs={'class': 'form-control'}))
