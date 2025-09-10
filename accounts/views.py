from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
import json
from .models import PDFUpload, Transaction
from .pdf_extractor import BankStatementExtractor

# Helper to extract only the required fields for the frontend
FRONTEND_FIELDS = [
    'customer_name',
    'account_number',
    'iban_number',
    'financial_period',
    'opening_balance',
    'closing_balance',
    'pages_processed',
    'total_transactions',
]

def get_frontend_result(pdf_upload):
    account_info = pdf_upload.account_info or {}
    return {
        'customer_name': account_info.get('customer_name', ''),
        'account_number': account_info.get('account_number', ''),
        'iban_number': account_info.get('iban_number', ''),
        'financial_period': account_info.get('financial_period', ''),
        'opening_balance': account_info.get('opening_balance', 0),
        'closing_balance': account_info.get('closing_balance', 0),
        'pages_processed': pdf_upload.pages_processed,
        'total_transactions': pdf_upload.total_transactions,
    }

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_pdf(request):
    try:
        if 'file' not in request.FILES:
            return Response(
                {'error': 'No file provided'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        pdf_file = request.FILES['file']
        if not pdf_file.name.lower().endswith('.pdf'):
            return Response(
                {'error': 'Only PDF files are allowed'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        pdf_upload = PDFUpload.objects.create(
            file=pdf_file,
            processed=False
        )
        try:
            file_path = pdf_upload.file.path
            print(f"Processing PDF: {file_path}")
            extractor = BankStatementExtractor()
            results = extractor.process_bank_statement(file_path)
            print(f"PDF processing completed successfully")
            if 'error' in results:
                pdf_upload.processing_error = results['error']
                pdf_upload.save()
                return Response(
                    {'error': results['error']}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            pdf_upload.processed = True
            pdf_upload.account_info = results.get('account_info', {})
            pdf_upload.total_transactions = results.get('total_transactions', 0)
            pdf_upload.pages_processed = results.get('pages_processed', 0)
            pdf_upload.monthly_analysis = results.get('monthly_analysis', {})
            pdf_upload.save()
            transactions = results.get('transactions', [])
            for transaction_data in transactions:
                Transaction.objects.create(
                    pdf_upload=pdf_upload,
                    date=transaction_data.get('date', ''),
                    description=transaction_data.get('description', ''),
                    debit=transaction_data.get('debit', 0),
                    credit=transaction_data.get('credit', 0),
                    balance=transaction_data.get('balance', 0)
                )
            # Build a full response for the frontend
            account_info = pdf_upload.account_info or {}
            # Add pages_processed and total_transactions to account_info for frontend
            account_info['pages_processed'] = pdf_upload.pages_processed
            account_info['total_transactions'] = pdf_upload.total_transactions
            response_data = {
                'account_info': account_info,
                'monthly_analysis': pdf_upload.monthly_analysis or {},
                'analytics': results.get('analytics', {
                    'average_fluctuation': 0,
                    'net_cash_flow_stability': 0,
                    'total_foreign_transactions': 0,
                    'total_foreign_amount': 0,
                    'overdraft_frequency': 0,
                    'overdraft_total_days': 0
                })
            }
            return Response(response_data, status=status.HTTP_201_CREATED)
        except Exception as e:
            error_msg = f'Error processing PDF: {str(e)}'
            print(f"PDF processing error: {error_msg}")
            pdf_upload.processing_error = error_msg
            pdf_upload.save()
            return Response(
                {'error': error_msg}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    except Exception as e:
        return Response(
            {'error': f'Error uploading file: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def get_pdf_results(request, pdf_id):
    try:
        pdf_upload = PDFUpload.objects.get(id=pdf_id)
        response_data = get_frontend_result(pdf_upload)
        return Response(response_data, status=status.HTTP_200_OK)
    except PDFUpload.DoesNotExist:
        return Response(
            {'error': 'PDF upload not found'}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return Response(
            {'error': f'Error retrieving results: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def list_pdf_uploads(request):
    try:
        pdf_uploads = PDFUpload.objects.all()
        response_data = [get_frontend_result(pdf) for pdf in pdf_uploads]
        return Response(response_data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response(
            {'error': f'Error listing PDF uploads: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['DELETE'])
def delete_pdf_upload(request, pdf_id):
    try:
        pdf_upload = PDFUpload.objects.get(id=pdf_id)
        if pdf_upload.file:
            pdf_upload.file.delete(save=False)
        pdf_upload.delete()
        return Response(
            {'message': 'PDF upload deleted successfully'}, 
            status=status.HTTP_204_NO_CONTENT
        )
    except PDFUpload.DoesNotExist:
        return Response(
            {'error': 'PDF upload not found'}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return Response(
            {'error': f'Error deleting PDF upload: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
