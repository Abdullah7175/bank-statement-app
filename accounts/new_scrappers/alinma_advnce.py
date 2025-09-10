#!/usr/bin/env python3
"""
Al-Inma (Arabic / English) bank-statement scraper
Handles both directions of columns and both languages.
"""
import re
import json
import pdfplumber
from pathlib import Path
import time
from typing import List, Dict, Any

# -----------------------------------------------------
PDF_PATH    = "pdf/68200263967000 , Alinma.pdf"    # <--- change if needed
OUTPUT_JSON = "4608277_full.json"
# -----------------------------------------------------

# ---------- OCR tidy ----------
def clean(txt: str) -> str:
    """
    Cleans and unifies text from OCR output.
    """
    return (
        txt.replace("−", "-")
        .replace("ي", "ي")
        .replace("ك", "ك")
        .strip()
    )

# ---------- HEADER PARSING ----------
def parse_header(full_text: str) -> Dict[str, Any]:
    """
    Parses header information (account summary) from the full OCR text.
    """
    hdr: Dict[str, Any] = {
        "currency": None,
        "account_number": None,
        "customer_name": None,
        "iban_number": None,
        "period": None,
    }

    # 1. Extract Period - handle Arabic date format
    m_period = re.search(
        r"(\d{4}-\d{2}-\d{2})\s*:\s*(\d{2}-\d{2}-\d{4})\s*:\s*(\d{2}-\d{2}-\d{4})",
        full_text,
        re.I
    )
    if m_period:
        hdr["period"] = f"{m_period.group(2).replace('-', '/')}/{m_period.group(3).split('-')[0]} - {m_period.group(3).replace('-', '/')}"

    # 2. Extract Currency, Account Number, Customer Name
    main_info_line_match = re.search(
        r"(SAR)\s*(\d{14,18})\s*(.+)",
        full_text,
        re.I | re.MULTILINE
    )
    if main_info_line_match:
        hdr["currency"] = clean(main_info_line_match.group(1))
        hdr["account_number"] = clean(main_info_line_match.group(2))
        hdr["customer_name"] = clean(main_info_line_match.group(3))

    # 3. Extract IBAN Number
    m_iban = re.search(r"SA([0-9A-Z]{22})", full_text, re.I)
    if m_iban:
        hdr["iban_number"] = clean(f"SA{m_iban.group(1)}")

    return hdr

# ---------- TRANSACTION PARSING ----------
def parse_transactions(full_text: str) -> List[Dict[str, Any]]:
    """
    Parses transaction details using a simplified date-description-debit-credit pattern.
    Works for both Arabic and English descriptions.
    """
    txns: List[Dict[str, Any]] = []

    # Normalize spaces
    clean_text = re.sub(r"\s+", " ", full_text)

    # Pattern for: Date  Description  Debit  Credit
    transaction_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2})"      # Date
        r"\s*(.*?)\s*"              # Description (Arabic/English)
        r"(\d+\.\d+)\s*"            # Debit
        r"(\d+\.\d+)",              # Credit
        re.DOTALL
    )

    for match in transaction_pattern.findall(clean_text):
        date, description, debit, credit = match
        txns.append({
            "date": date,
            "description": clean(description),
            "debit": float(debit),
            "credit": float(credit),
            "balance": None  # No balance in this pattern
        })

    # Sort by date just in case
    txns.sort(key=lambda x: x['date'])
    return txns

# ---------- main ----------
def main() -> None:
    """
    Main function to orchestrate the PDF parsing process.
    """
    t0 = time.time()
    results: Dict[str, Any] = {
        "pdf_file": str(Path(PDF_PATH).resolve()),
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_pages": 0,
        "account_summary": {},
        "transactions": [],
    }

    try:
        with pdfplumber.open(PDF_PATH) as pdf:
            # Extract text with better settings for Arabic text
            full_text = "\n".join(
                page.extract_text(x_tolerance=3, y_tolerance=3) or "" 
                for page in pdf.pages
            )
            results["total_pages"] = len(pdf.pages)

            # Save raw text for debugging
            raw_text_output_path = Path(PDF_PATH).with_suffix(".raw.txt")
            raw_text_output_path.write_text(full_text, encoding="utf-8")
            print(f"Debug: Raw text saved to {raw_text_output_path}")

            # Parse data
            results["account_summary"] = parse_header(full_text)
            results["transactions"] = parse_transactions(full_text)

    except FileNotFoundError:
        print(f"❌ Error: PDF file not found at '{PDF_PATH}'. Please ensure the path is correct.")
        return
    except Exception as e:
        print(f"❌ An unexpected error occurred during PDF processing: {e}")
        return

    # Finalize results
    results["total_transactions"] = len(results["transactions"])
    results["processing_time"] = f"{time.time() - t0:.2f}s"

    try:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ Successfully extracted {results['total_transactions']} transactions and saved to {OUTPUT_JSON}")
        print(f"Account Summary: {results['account_summary']}")
        
        # Print sample transactions for verification
        if results['transactions']:
            print(f"\n📊 Found {len(results['transactions'])} transactions")
        
    except IOError as io_err:
        print(f"❌ Error writing output JSON to {OUTPUT_JSON}: {io_err}")

if __name__ == "__main__":
    main()
