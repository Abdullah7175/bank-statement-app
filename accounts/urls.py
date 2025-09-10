from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_pdf, name='upload_pdf'),
    path('list/', views.list_pdf_uploads, name='list_pdf_uploads'),
    path('results/<int:pdf_id>/', views.get_pdf_results, name='get_pdf_results'),
    path('delete/<int:pdf_id>/', views.delete_pdf_upload, name='delete_pdf_upload'),
] 