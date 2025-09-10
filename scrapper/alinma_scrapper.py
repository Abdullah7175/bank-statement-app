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
PDF_PATH    = "pdf/4608277 alinma.pdf"    # <--- change if needed
OUTPUT_JSON = "4608277_full.json"
# -----------------------------------------------------

# ---------- OCR tidy ----------
def clean(txt: str) -> str:
    """
    Cleans and unifies text from OCR output.
    - Replaces specific non-standard characters with standard ones (e.g., Arabic hyphen).
    - Includes placeholders for unifying Arabic characters (ي, ك) which currently
      perform no operation but can be extended for OCR-specific character variations.
    - Strips leading/trailing whitespace.
    """
    return (
        txt.replace("−", "-")  # Replace Arabic hyphen with standard hyphen
        .replace("ي", "ي")  # Placeholder for unifying Arabic yeh (e.g., 'ى' to 'ي')
        .replace("ك", "ك")  # Placeholder for unifying Arabic kaf (e.g., different forms of 'ك')
        .strip()
    )

# ---------- HEADER PARSING ----------
def parse_header(full_text: str) -> Dict[str, Any]:
    """
    Parses header information (account summary) from the full OCR text of the bank statement.
    It attempts to handle both Arabic (Right-to-Left) and English (Left-to-Right) layouts
    by trying specific patterns based on observed document structures.
    """
    hdr: Dict[str, Any] = {
        "currency": None,
        "account_number": None,
        "customer_name": None,
        "iban_number": None,
        "period": None,
        "opening_balance": None,
        "closing_balance": None,
    }

    # 1. Extract Period (e.g., "2024-01-01 - 2024-12-31")
    # Updated regex to allow for text/newlines between the label and the date range,
    # and explicitly includes "Account Statement Date".
    m_period = re.search(
        r"(?:Account Statement Date|ﺏﺎﺴﺤﻟﺍ ﻒﺸﻛ ﺦﻳﺭﺎﺗ)[^\n]*?\s*([0-9]{4}-\d{2}-\d{2}\s*-\s*[0-9]{4}-\d{2}-\d{2})",
        full_text,
        re.I | re.DOTALL # re.DOTALL ensures '.' matches newlines
    )
    if m_period:
        hdr["period"] = clean(m_period.group(1))

    # 2. Extract Currency, Account Number, Customer Name
    # This section specifically targets a common line structure found in Arabic Al-Inma statements:
    # "SAR <Account Number> <Customer Name>"
    main_info_line_match = re.search(
        r"(SAR)\s*(\d{14,18})\s*(.+)",  # Captures SAR, 14-18 digits (account), and rest of line (name)
        full_text,
        re.I | re.MULTILINE # Case-insensitive and allows '.' to match newlines
    )
    if main_info_line_match:
        hdr["currency"] = clean(main_info_line_match.group(1))
        hdr["account_number"] = clean(main_info_line_match.group(2))
        hdr["customer_name"] = clean(main_info_line_match.group(3))
    else:
        # Fallback if the combined Arabic line is not found (e.g., for English statements
        # or different Arabic layouts). Tries to find each piece of information individually.

        # Try to find Currency using its label
        m_currency = re.search(r"(?:Currency|ﺔﻠﻤﻋ)\s*([A-Z]{3})", full_text, re.I)
        if m_currency:
            hdr["currency"] = clean(m_currency.group(1))

        # Try to find Account Number using its label. \D* allows for non-digit characters
        # (like spaces or newlines) between the label and the number.
        m_account_number = re.search(r"(?:Account Number|ﺏﺎﺴﺤﻟﺍ ﻢﻗﺭ)\D*(\d{14,18})", full_text, re.I)
        if m_account_number:
            hdr["account_number"] = clean(m_account_number.group(1))

        # Try to find Customer Name using its label. Captures until the next newline.
        m_customer_name = re.search(r"(?:Customer Name|ﻞﻴﻤﻌﻟﺍ ﻢﺳﺍ)\s*([^\n]+)", full_text, re.I)
        if m_customer_name:
            hdr["customer_name"] = clean(m_customer_name.group(1))

    # 3. Extract IBAN Number
    # Updated to specifically target the Saudi IBAN format (SA followed by 22 alphanumeric characters),
    # regardless of nearby labels, as it appears within transaction details in the raw text.
    m_iban = re.search(
        r"SA([0-9A-Z]{22})", # Looks for 'SA' followed by exactly 22 alphanumeric characters
        full_text,
        re.I | re.DOTALL
    )
    if m_iban:
        hdr["iban_number"] = clean(f"SA{m_iban.group(1)}") # Re-add 'SA' prefix

    # 4. Extract Opening and Closing Balances
    # Look for patterns like "21825.64 (SAR) 6489.46 (SAR) Closing Balance Opening Balance"
    # First try the English pattern
    balance_pattern_en = re.search(
        r"(\d+(?:\.\d{2})?)\s*\(SAR\)?\s*(\d+(?:\.\d{2})?)\s*\(SAR\)?\s*Closing Balance\s*Opening Balance",
        full_text,
        re.I | re.DOTALL
    )
    if balance_pattern_en:
        try:
            closing_balance = float(clean(balance_pattern_en.group(1)).replace(",", ""))
            opening_balance = float(clean(balance_pattern_en.group(2)).replace(",", ""))
            hdr["closing_balance"] = f"{closing_balance:.2f}"
            hdr["opening_balance"] = f"{opening_balance:.2f}"
        except (ValueError, TypeError):
            pass

    # If English pattern didn't work, try Arabic pattern
    if not hdr["opening_balance"] and not hdr["closing_balance"]:
        balance_pattern_ar = re.search(
            r"ﻝﺎﻔﻗﻹﺍ\s*ﺪﻴﺻﺭ\s*ﻲﺣﺎﺘﺘﻓﻹﺍ\s*ﺪﻴﺻﺮﻟﺍ\s*(\d+(?:\.\d{2})?)\s*\(SAR\)?\s*(\d+(?:\.\d{2})?)\s*\(SAR\)?",
            full_text,
            re.I | re.DOTALL
        )
        if balance_pattern_ar:
            try:
                closing_balance = float(clean(balance_pattern_ar.group(1)).replace(",", ""))
                opening_balance = float(clean(balance_pattern_ar.group(2)).replace(",", ""))
                hdr["closing_balance"] = f"{closing_balance:.2f}"
                hdr["opening_balance"] = f"{opening_balance:.2f}"
            except (ValueError, TypeError):
                pass

    # Alternative pattern: Look for individual balance patterns
    if not hdr["opening_balance"] or not hdr["closing_balance"]:
        # Look for opening balance
        opening_match = re.search(
            r"(\d+(?:\.\d{2})?)\s*\(SAR\)?\s*(?:Opening Balance|ﻲﺋﺍﺪﺘﺑﻻﺍ ﺪﻴﺻﺮﻟﺍ)",
            full_text,
            re.I
        )
        if opening_match:
            try:
                opening_balance = float(clean(opening_match.group(1)).replace(",", ""))
                hdr["opening_balance"] = f"{opening_balance:.2f}"
            except (ValueError, TypeError):
                pass

        # Look for closing balance
        closing_match = re.search(
            r"(\d+(?:\.\d{2})?)\s*\(SAR\)?\s*(?:Closing Balance|ﻲﺋﺎﻬﻨﻟﺍ ﺪﻴﺻﺮﻟﺍ)",
            full_text,
            re.I
        )
        if closing_match:
            try:
                closing_balance = float(clean(closing_match.group(1)).replace(",", ""))
                hdr["closing_balance"] = f"{closing_balance:.2f}"
            except (ValueError, TypeError):
                pass

    return hdr

# ---------- TRANSACTION PATTERNS ----------
# Regex for Arabic (Right-to-Left) transaction lines.
# Example line structure: "Balance (SAR) Amount (SAR) Description Date"
ARABIC_TXN_RE = re.compile(
    r"""
    (?P<balance>-?[\d,]+\.\d{2})\s*\(?SAR\)?\s+        # Capture running balance (e.g., 21825.64 (SAR))
    (?P<amount>-?[\d,]+\.\d{2})\s*\(?SAR\)?\s+         # Capture single transaction amount (e.g., 9600.00 (SAR))
    (?P<description>.+?)                              # Capture description (non-greedy, any character)
    \s+\b(?P<date>\d{2}/\d{2}/\d{4})\s*$               # Capture date (DD/MM/YYYY) at the end of the line
    """,
    re.VERBOSE | re.I | re.MULTILINE, # Verbose for comments, case-insensitive, multiline for ^ and $
)

# Regex for English (Left-to-Right) transaction lines (as a fallback).
# Example line structure: "Date Debit Credit Balance"
ENG_TXN_RE = re.compile(
    r"""
    (?P<date>\d{2}/\d{2}/\d{4})\s+                     # Capture date (DD/MM/YYYY)
    (?P<debit>-?[\d,]+\.\d{2})\s*(?:S?A?R?)?\s* # Capture debit amount (optional SAR)
    (?P<credit>-?[\d,]+\.\d{2})\s*(?:S?A?R?)?\s* # Capture credit amount (optional SAR)
    (?P<balance>-?[\d,]+\.\d{2})\s*(?:S?A?R?)?\s*$     # Capture running balance (optional SAR) at line end
    """,
    re.VERBOSE | re.I | re.MULTILINE,
)

def parse_transactions(full_text: str) -> List[Dict[str, Any]]:
    """
    Parses transaction details from the full OCR text.
    It first attempts to use the Arabic (RTL) pattern. If no transactions are found
    or if the Arabic parsing seems incomplete, it falls back to the English (LTR) pattern.
    Amounts are converted to floats, and debit/credit are separated.
    """
    txns: List[Dict[str, Any]] = []

    # 1. Try Arabic RTL pattern first
    for m in ARABIC_TXN_RE.finditer(full_text):
        try:
            # Clean and convert amount and balance to float, handling comma separators
            amt = float(clean(m.group("amount")).replace(",", ""))
            balance = float(clean(m.group("balance")).replace(",", ""))

            txns.append(
                {
                    "date": clean(m.group("date")),
                    "description": clean(m.group("description")), # Now properly capturing description
                    "debit": abs(amt) if amt < 0 else 0.0,    # Debit is positive if amount is negative
                    "credit": amt if amt > 0 else 0.0,       # Credit is positive if amount is positive
                    "balance": balance,
                }
            )
        except ValueError as ve:
            # Print a warning for lines that fail to parse, but continue processing others
            print(f"Warning: Could not parse Arabic transaction line: '{m.group(0).strip()}' - Error: {ve}")
            continue # Skip to the next match

    # 2. Fallback to English LTR if no Arabic transactions were found
    # This ensures that if the Arabic pattern fails completely, the English one is attempted.
    if not txns:
        for m in ENG_TXN_RE.finditer(full_text):
            try:
                # Clean and convert debit, credit, balance to float, handling comma separators
                debit = float(clean(m.group("debit")).replace(",", ""))
                credit = float(clean(m.group("credit")).replace(",", ""))
                balance = float(clean(m.group("balance")).replace(",", ""))

                txns.append(
                    {
                        "date": clean(m.group("date")),
                        "description": "",  # English regex does not explicitly capture description in this format
                        "debit": debit,
                        "credit": credit,
                        "balance": balance,
                    }
                )
            except ValueError as ve:
                print(f"Warning: Could not parse English transaction line: '{m.group(0).strip()}' - Error: {ve}")
                continue # Skip to the next match

    # Sort transactions by date, oldest first.
    # Assumes 'date' is in DD/MM/YYYY format for sorting, or needs conversion to proper datetime objects.
    # For now, `reverse()` is kept as in the original, assuming transactions are found newest first.
    txns.reverse() # Reverses the order to make it oldest first if found newest first
    return txns

# ---------- main ----------
def main() -> None:
    """
    Main function to orchestrate the PDF parsing process.
    It opens the PDF, extracts text, parses header and transaction data,
    and then saves the results to a JSON file.
    Includes robust error handling for file operations.
    """
    t0 = time.time() # Start time for performance measurement
    results: Dict[str, Any] = {
        "pdf_file": str(Path(PDF_PATH).resolve()), # Absolute path to the processed PDF
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"), # Timestamp of processing
        "total_pages": 0,
        "account_summary": {},
        "transactions": [],
    }

    try:
        # Open the PDF using pdfplumber
        with pdfplumber.open(PDF_PATH) as pdf:
            # Extract text from all pages. x_tolerance helps account for minor OCR misalignments.
            full_text = "\n".join(page.extract_text(x_tolerance=2) or "" for page in pdf.pages)
            results["total_pages"] = len(pdf.pages)

            # Save the raw extracted text to a file for debugging purposes
            raw_text_output_path = Path(PDF_PATH).with_suffix(".raw.txt")
            raw_text_output_path.write_text(full_text, encoding="utf-8")
            print(f"Debug: Raw text saved to {raw_text_output_path}")

            # Parse header (account summary) and transactions from the full text
            results["account_summary"] = parse_header(full_text)
            results["transactions"] = parse_transactions(full_text)

    except FileNotFoundError:
        # Handle case where the specified PDF file does not exist
        print(f"❌ Error: PDF file not found at '{PDF_PATH}'. Please ensure the path is correct.")
        return
    except Exception as e:
        # Catch any other unexpected errors during PDF processing
        print(f"❌ An unexpected error occurred during PDF processing: {e}")
        return

    # Finalize results summary
    results["total_transactions"] = len(results["transactions"])
    results["processing_time"] = f"{time.time() - t0:.2f}s" # Calculate total processing time

    try:
        # Write the results to a JSON file
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2) # ensure_ascii=False for proper Arabic characters
        print(f"✅ Successfully extracted {results['total_transactions']} transactions and saved to {OUTPUT_JSON}")
        print(f"Account Summary: {results['account_summary']}") # Print summary for quick verification
    except IOError as io_err:
        # Handle errors during writing the JSON output file
        print(f"❌ Error writing output JSON to {OUTPUT_JSON}: {io_err}")


if __name__ == "__main__":
    main()
