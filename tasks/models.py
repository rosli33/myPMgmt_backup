from django.db import models

class Task(models.Model):
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

    task_id = models.CharField(max_length=20, unique=True, default='default_id')
    task_title = models.CharField(max_length=255, default='Untitled Task')
    task_type = models.CharField(max_length=100, choices=TASK_TYPE_CHOICES)
    current_status = models.CharField(max_length=100, null=True, blank=True)
    priority_level = models.CharField(max_length=50, null=True, blank=True)
    resource_id = models.CharField(max_length=50, null=True, blank=True)
    assignee_name = models.CharField(max_length=255, null=True, blank=True)
    estimated_effort = models.FloatField()
    business_impact = models.CharField(max_length=10, choices=BUSINESS_IMPACT_CHOICES)
    deadline = models.DateField(null=True, blank=True)
    creation_date = models.DateField(auto_now_add=True)

    def __str__(self):
        return self.task_title

class ManualTask(models.Model):
    task_title = models.CharField(max_length=255)
    task_type = models.CharField(max_length=100)
    current_status = models.CharField(max_length=100)
    business_impact = models.CharField(max_length=100)
    estimated_effort = models.FloatField()
    priority_level = models.CharField(max_length=100)
    deadline = models.DateField(null=True, blank=True)

    def __str__(self):
        return self.task_title
