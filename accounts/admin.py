from django.contrib import admin
from .models import PDFUpload, Transaction

@admin.register(PDFUpload)
class PDFUploadAdmin(admin.ModelAdmin):
    list_display = ('id', 'file', 'uploaded_at', 'processed', 'total_transactions', 'pages_processed')
    list_filter = ('processed', 'uploaded_at')
    search_fields = ('file',)
    readonly_fields = ('uploaded_at', 'account_info', 'monthly_analysis')
    
    def get_queryset(self, request):
        return super().get_queryset(request).prefetch_related('transactions')

@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = ('date', 'description', 'debit', 'credit', 'balance', 'pdf_upload')
    list_filter = ('date', 'pdf_upload')
    search_fields = ('description', 'date')
    readonly_fields = ('pdf_upload',)
