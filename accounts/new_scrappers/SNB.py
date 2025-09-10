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
PDF_PATH    = "pdf/14300001190204 SNB.pdf"    # <--- change if needed
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
    }

    # 1. Extract Period (e.g., "01/08/2024 07/02/2025")
    # Updated regex to explicitly look for "Date(Gregorian)" or Arabic equivalent,
    # then capture two DD/MM/YYYY dates.
    m_period = re.search(
        r"(?:Date\(Gregorian\)|ﻱﺩﻼﻴﻣ\)ﺦﻳﺭﺎﺗ)[^\n]*?(\d{2}/\d{2}/\d{4})\s*(\d{2}/\d{2}/\d{4})",
        full_text,
        re.I | re.DOTALL # re.DOTALL ensures '.' matches newlines
    )
    if m_period:
        hdr["period"] = f"{clean(m_period.group(1))} - {clean(m_period.group(2))}"

    # 2. Extract Currency, Account Number, Customer Name
    # This section now includes more general labels observed in the new PDF.

    # Try to find Customer Name using its label. Captures until the next newline.
    m_customer_name = re.search(r"(?:Customer Name|Name|ﻞﻴﻤﻌﻟﺍ ﻢﺳﺍ|ﻢﺳﻻﺍ)\s*([^\n]+)", full_text, re.I)
    if m_customer_name:
        hdr["customer_name"] = clean(m_customer_name.group(1))

    # Try to find Account Number using its label. \D* allows for non-digit characters
    # (like spaces or newlines) between the label and the number.
    m_account_number = re.search(r"(?:Account Number|ﺏﺎﺴﺤﻟﺍ ﻢﻗﺭ)\D*(\d{10,18})", full_text, re.I) # Adjusted range for 10-18 digits
    if m_account_number:
        hdr["account_number"] = clean(m_account_number.group(1))

    # Try to find Currency using its label. SAR is explicitly extracted.
    m_currency = re.search(r"(?:Currency|ﺔﻠﻤﻌﻟﺍ)\s*([A-Z]{3})", full_text, re.I)
    if m_currency:
        hdr["currency"] = clean(m_currency.group(1))
    else:
        # Fallback for currency if not found with label, look for SAR after Account Type
        m_currency_fallback = re.search(r"(?:Account Type & Currency|ﺔﻠﻤﻌﻟﺍ ﻭ ﺏﺎﺴﺤﻟﺍ ﻉﻮﻧ)[^\n]*?(SAR)", full_text, re.I)
        if m_currency_fallback:
            hdr["currency"] = clean(m_currency_fallback.group(1))


    # 3. Extract IBAN Number
    # Now specifically targets the Saudi IBAN format (SA followed by 22 alphanumeric characters),
    # regardless of nearby labels, as it appears directly in the raw text.
    m_iban = re.search(
        r"SA([0-9A-Z]{22})", # Looks for 'SA' followed by exactly 22 alphanumeric characters
        full_text,
        re.I | re.DOTALL
    )
    if m_iban:
        hdr["iban_number"] = clean(f"SA{m_iban.group(1)}") # Re-add 'SA' prefix

    return hdr

# ---------- TRANSACTION PATTERNS ----------
# Regex for Arabic (Right-to-Left) transaction lines.
# This pattern is flexible to capture one 'amount' that can be either debit or credit,
# based on the observed data where either debit or credit is present, but not both simultaneously.
ARABIC_TXN_RE = re.compile(
    r"""
    (?P<balance>-?[\d,]+\.\d{2})\s+            # Capture running balance (e.g., 11,500.00)
    (?P<amount>-?[\d,]+\.\d{2})\s+             # Capture single transaction amount (e.g., 10,000.00 or 4.03)
    (?P<description>.+?)                      # Capture description (non-greedy, any character)
    \s+\b(?P<date>\d{2}/\d{2}/\d{4})\s*$       # Capture date (DD/MM/YYYY) at the end of the line
    """,
    re.VERBOSE | re.I | re.MULTILINE,
)

# English LTR transaction regex (kept for potential fallback, though less likely to be used with new data)
ENG_TXN_RE = re.compile(
    r"""
    (?P<date>\d{2}/\d{2}/\d{4})\s+
    (?P<debit>-?[\d,]+\.\d{2})\s*(?:S?A?R?)?\s*
    (?P<credit>-?[\d,]+\.\d{2})\s*(?:S?A?R?)?\s*
    (?P<balance>-?[\d,]+\.\d{2})\s*(?:S?A?R?)?\s*$
    """,
    re.VERBOSE | re.I | re.MULTILINE,
)

def parse_transactions(full_text: str) -> List[Dict[str, Any]]:
    """
    Parses transaction details from the full OCR text.
    It primarily uses the Arabic (RTL) pattern which is adapted for the new PDF format.
    It then infers if the captured 'amount' is a debit or credit based on keywords in the description.
    """
    txns: List[Dict[str, Any]] = []

    # Keywords to help determine if an amount is a debit or credit
    # These keywords are case-insensitive.
    DEBIT_KEYWORDS = ["GOSI FEE", "outgoing transfer", "رسوم", "ﺭﺩﺎﺻ ﻞﻳﻮﺤﺗ", "ﻑﺮﺻ", "Debit"]
    CREDIT_KEYWORDS = ["incoming transfer", "إيداع", "ﺩﺭﺍﻭ ﻲﻠﺧﺍﺩ ﻞﻳﻮﺤﺗ", "ﻉﺍﺪﻳﺇ", "Credit"]


    # 1. Try ARABIC_TXN_RE pattern first (adapted for new PDF's single amount column)
    for m in ARABIC_TXN_RE.finditer(full_text):
        try:
            amt = float(clean(m.group("amount")).replace(",", ""))
            balance = float(clean(m.group("balance")).replace(",", ""))
            description = clean(m.group("description"))
            date = clean(m.group("date"))

            debit_val = 0.0
            credit_val = 0.0

            # Convert description to lowercase for case-insensitive keyword matching
            description_lower = description.lower()

            is_debit_by_keyword = any(keyword.lower() in description_lower for keyword in DEBIT_KEYWORDS)
            is_credit_by_keyword = any(keyword.lower() in description_lower for keyword in CREDIT_KEYWORDS)

            if is_debit_by_keyword and not is_credit_by_keyword:
                debit_val = amt
            elif is_credit_by_keyword and not is_debit_by_keyword:
                credit_val = amt
            elif amt < 0: # Fallback: if amount is negative, it's a debit
                debit_val = abs(amt)
            elif amt >= 0: # Fallback: if amount is positive, and no clear keywords, assume credit
                credit_val = amt


            txns.append(
                {
                    "date": date,
                    "description": description,
                    "debit": debit_val,
                    "credit": credit_val,
                    "balance": balance,
                }
            )
        except ValueError as ve:
            print(f"Warning: Could not parse transaction line: '{m.group(0).strip()}' - Error: {ve}")
            continue

    # 2. Fallback to English LTR if no Arabic transactions were found (less likely with new PDF)
    # This block is retained for compatibility, but the primary expectation is for ARABIC_TXN_RE to work.
    if not txns:
        for m in ENG_TXN_RE.finditer(full_text):
            try:
                debit = float(clean(m.group("debit")).replace(",", ""))
                credit = float(clean(m.group("credit")).replace(",", ""))
                balance = float(clean(m.group("balance")).replace(",", ""))

                txns.append(
                    {
                        "date": clean(m.group("date")),
                        "description": "", # ENG_TXN_RE does not extract description directly
                        "debit": debit,
                        "credit": credit,
                        "balance": balance,
                    }
                )
            except ValueError as ve:
                print(f"Warning: Could not parse English transaction line: '{m.group(0).strip()}' - Error: {ve}")
                continue

    # Sort transactions by date, oldest first.
    # Note: Sorting by DD/MM/YYYY string format might not be truly chronological for dates in different months/years.
    # For robust sorting, convert 'date' to datetime objects. For this example, simple reverse is kept.
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
