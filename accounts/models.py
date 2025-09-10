from django.db import models
from django.utils import timezone
import json

# Create your models here.

class PDFUpload(models.Model):
    """Model to store uploaded PDF files and their extraction results"""
    file = models.FileField(upload_to='pdfs/')
    uploaded_at = models.DateTimeField(default=timezone.now)
    processed = models.BooleanField(default=False)
    processing_error = models.TextField(blank=True, null=True)
    
    # Extracted data fields
    account_info = models.JSONField(default=dict, blank=True)
    total_transactions = models.IntegerField(default=0)
    pages_processed = models.IntegerField(default=0)
    monthly_analysis = models.JSONField(default=dict, blank=True)
    
    def __str__(self):
        return f"PDF Upload {self.id} - {self.file.name}"
    
    class Meta:
        ordering = ['-uploaded_at']

class Transaction(models.Model):
    """Model to store individual transactions"""
    pdf_upload = models.ForeignKey(PDFUpload, on_delete=models.CASCADE, related_name='transactions')
    date = models.CharField(max_length=20)
    description = models.TextField()
    debit = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    credit = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    balance = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    
    def __str__(self):
        return f"{self.date} - {self.description[:50]}"
    
    class Meta:
        ordering = ['date']
