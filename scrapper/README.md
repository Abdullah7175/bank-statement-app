# PDF Scraper Package

This package contains all the PDF scraping functionality for extracting bank statement data.

## Package Structure

```
scrapper/
├── __init__.py                    # Package initialization
├── enhanced_balance_scraper.py    # Main scraper (extracts only opening/closing balances)
├── alrajhi.py                     # Alrajhi bank scraper
├── alrajhi_large_statement.py     # Alrajhi large statement scraper
├── alinma_scrapper.py             # Alinma bank scraper
├── alina_en.py                    # Alina English scraper
├── alinma_advnce.py               # Advanced Alinma scraper
├── SNB.py                         # Saudi National Bank scraper
├── sbn_large.py                   # SNB large statement scraper
├── toyland_ocr_scrapper.py        # OCR scraper for image-based PDFs
├── combine_scrapper.py            # Original combine scraper
├── enhanced_combine_scrapper.py   # Enhanced combine scraper
└── README.md                      # This file
```

## Main Components

### Enhanced Balance Scraper
The main scraper that extracts only opening and closing balances from PDFs.

**Features:**
- Extracts only opening and closing balances
- Supports multiple extraction methods (PDFplumber, existing scrapers, OCR)
- Handles different bank formats (Alrajhi, Alinma, SNB, etc.)
- OCR support for image-based PDFs
- Corrected balance extraction for specific PDFs

### Individual Bank Scrapers
- **Alrajhi**: Handles Alrajhi bank statements
- **Alinma**: Handles Alinma bank statements  
- **SNB**: Handles Saudi National Bank statements
- **OCR**: Handles image-based PDFs using advanced OCR techniques

## Usage

### From Root Directory
```bash
python3 run_balance_scraper.py
```

### As Python Package
```python
from scrapper import extract_balances_from_pdf

# Extract balances from a single PDF
opening, closing = extract_balances_from_pdf("path/to/pdf")

# Run the full scraper
from scrapper.enhanced_balance_scraper import main
main()
```

### Direct Import
```python
from scrapper.enhanced_balance_scraper import extract_balances_from_pdf
from scrapper.alrajhi import parse_account_summary
from scrapper.alinma_scrapper import parse_header
```

## Dependencies

- pdfplumber
- PyMuPDF (fitz)
- pytesseract
- PIL (Pillow)
- opencv-python
- numpy

## Installation

```bash
pip install pdfplumber PyMuPDF pytesseract pillow opencv-python numpy
```

## Notes

- All scrapers are now organized in this package for easy portability
- The main entry point is `enhanced_balance_scraper.py`
- OCR functionality is available for image-based PDFs
- All balance extractions have been corrected for specific PDF formats
