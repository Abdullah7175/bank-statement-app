"""
PDF Scraper Package
==================

This package contains various PDF scrapers for extracting bank statement data.

Main Components:
- enhanced_balance_scraper: Main scraper that extracts only opening and closing balances
- Individual bank scrapers: alrajhi, alinma, SNB, etc.
- OCR functionality for image-based PDFs

Usage:
    from scrapper.enhanced_balance_scraper import extract_balances_from_pdf
    
    opening, closing = extract_balances_from_pdf("path/to/pdf")
"""

__version__ = "1.0.0"
__author__ = "PDF Scraper Team"

# Import main functions for easy access
from .enhanced_balance_scraper import extract_balances_from_pdf, main

__all__ = [
    'extract_balances_from_pdf',
    'main'
]
