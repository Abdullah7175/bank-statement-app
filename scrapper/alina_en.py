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
# --- IMPORTANT: Change this path to your PDF file ---
PDF_PATH    = "pdf/Alinma EN (1) (1).pdf"
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
        "ref_no": None,
        "customer_name": None,
        "period": None,
        "account_number": None,
        "currency": None,
        "opening_balance": None,
        "closing_balance": None,
        "number_of_deposits": None,
        "totals_deposits": None,
        "number_of_withdraws": None,
        "total_withdraws": None,
        "iban_number": None, # IBAN not clearly labeled in this new format, kept as None unless inferred
    }



    # 1. Extract Alinma ID
    # Pattern: "Alinma ID 00000542995 ﺀﺎﻤﻧﻹﺍ ﻢﻗﺭ"
    m_alinma_id = re.search(r"(?:Alinma ID|ﺀﺎﻤﻧﻹﺍ ﻢﻗﺭ)\s*([A-Z0-9]+)", full_text, re.I)
    if m_alinma_id:
        hdr["alinma_id"] = clean(m_alinma_id.group(1))

    # 2. Extract Reference Number
    # Pattern: "Ref. No. 0901000003bc013d ﻲﻌﺟﺮﻤﻟﺍ ﻢﻗﺮﻟﺍ"
    m_ref_no = re.search(r"(?:Ref\. No\.|ﻲﻌﺟﺮﻤﻟﺍ ﻢﻗﺮﻟﺍ)\s*([A-Z0-9]+)", full_text, re.I)
    if m_ref_no:
        hdr["ref_no"] = clean(m_ref_no.group(1))
    
    # 3. Extract Customer Name
    # Pattern: "Customer Name AL ZEYADI,ABDULAZIZA ﻞﻴﻤﻌﻟﺍ ﻢﺳﺍ"
    m_customer_name = re.search(
        r"(?:Customer Name|ﻞﻴﻤﻌﻟﺍ ﻢﺳﺍ)\s*([^\n]+)",
        full_text,
        re.I
    )
    if m_customer_name:
        hdr["customer_name"] = clean(m_customer_name.group(1))

    # 4. Extract Period (Date From[DD/MM/YYYY] To[DD/MM/YYYY])
    # Pattern: "Date From[18/05/2024] To[18/06/2025] ﺦﻳﺭﺎﺘﻟﺍ"
    m_period = re.search(
        r"(?:Date From|ﺦﻳﺭﺎﺘﻟﺍ)\[(\d{2}/\d{2}/\d{4})\]\s*To\[(\d{2}/\d{2}/\d{4})\]",
        full_text,
        re.I
    )
    if m_period:
        hdr["period"] = f"{clean(m_period.group(1))} - {clean(m_period.group(2))}"

    # 5. Extract Account Number
    # Pattern: "Account Number 68201354629000 ﺏﺎﺴﺤﻟﺍ ﻢﻗﺭ"
    m_account_number = re.search(
        r"(?:Account Number|ﺏﺎﺴﺤﻟﺍ ﻢﻗﺭ)\s*(\d+)",
        full_text,
        re.I
    )
    if m_account_number:
        hdr["account_number"] = clean(m_account_number.group(1))

    # 6. Extract Account Currency
    # Pattern: "Account Currency SAR ﺏﺎﺴﺤﻟﺍ ﺔﻠﻤﻋ"
    m_currency = re.search(
        r"(?:Account Currency|ﺏﺎﺴﺤﻟﺍ ﺔﻠﻤﻋ)\s*([A-Z]{3})",
        full_text,
        re.I
    )
    if m_currency:
        hdr["currency"] = clean(m_currency.group(1))

    # 7. Extract Opening Balance
    # Pattern: "Opening Balance 5,908.54 ﻲﺣﺎﺘﺘﻓﻻﺍ ﺏﺎﺴﺤﻟﺍ ﺪﻴﺻﺭ"
    m_opening_balance = re.search(
        r"(?:Opening Balance|ﻲﺣﺎﺘﺘﻓﻻﺍ ﺏﺎﺴﺤﻟﺍ ﺪﻴﺻﺭ)\s*([-\d,\.]+)", # Capture just the number
        full_text,
        re.I
    )
    if m_opening_balance:
        hdr["opening_balance"] = clean(m_opening_balance.group(1))

    # 8. Extract Closing Balance
    # Pattern: "Closing Balance 0 ﻝﺎﻔﻗﻹﺍ ﺪﻴﺻﺭ"
    m_closing_balance = re.search(
        r"(?:Closing Balance|ﻝﺎﻔﻗﻹﺍ ﺪﻴﺻﺭ)\s*([-\d,\.]+)", # Capture just the number
        full_text,
        re.I
    )
    if m_closing_balance:
        hdr["closing_balance"] = clean(m_closing_balance.group(1))

    # 9. Extract Number Of Deposits and Totals Deposits
    # Pattern: "Number Of Deposits 22 ﺕﺎﻋﺍﺪﻳﻹﺍ ﺩﺪﻋ\nTotals Deposits 27,200 ﺕﺎﻋﺍﺪﻳﻹﺍ ﻲﻟﺎﻤﺟﺇ"
    m_deposits = re.search(
        r"(?:Number Of Deposits|ﺕﺎﻋﺍﺪﻳﻹﺍ ﺩﺪﻋ)\s*(\d+)\s*\n*(?:Totals Deposits|ﺕﺎﻋﺍﺪﻳﻹﺍ ﻲﻟﺎﻤﺟﺇ)\s*([-\d,\.]+)",
        full_text,
        re.I | re.DOTALL # Use DOTALL to match across newlines
    )
    if m_deposits:
        hdr["number_of_deposits"] = int(clean(m_deposits.group(1)))
        hdr["totals_deposits"] = float(clean(m_deposits.group(2)).replace(",", ""))

    # 10. Extract Number of Withdraws and Total Withdraws
    # Pattern: "Number of Withdraws 69 ﺕﺎﺑﻮﺤﺴﻟﺍ ﺩﺪﻋ\nTotal Withdraws -33,108.54 ﺕﺎﺑﻮﺤﺴﻟﺍ ﻲﻟﺎﻤﺟﺇ"
    m_withdraws = re.search(
        r"(?:Number of Withdraws|ﺕﺎﺑﻮﺤﺴﻟﺍ ﺩﺪﻋ)\s*(\d+)\s*\n*(?:Total Withdraws|ﺕﺎﺑﻮﺤﺴﻟﺍ ﻲﻟﺎﻤﺟﺇ)\s*([-\d,\.]+)",
        full_text,
        re.I | re.DOTALL # Use DOTALL to match across newlines
    )
    if m_withdraws:
        hdr["number_of_withdraws"] = int(clean(m_withdraws.group(1)))
        hdr["total_withdraws"] = float(clean(m_withdraws.group(2)).replace(",", ""))
        
    # IBAN is not explicitly labeled with "IBAN" in this format.
    # The "Ref. No." (0901000003bc013d) does not match a standard SA IBAN format.
    # Therefore, IBAN will remain None for this document.

    return hdr

# ---------- TRANSACTION PATTERNS ----------
# Updated Regex for transaction lines based on "Balance Credit Debit Description Transaction Date" structure
# This regex aims for a more direct capture of fields based on observed OCR output.
# It captures Balance, a single Amount (which can be Credit or Debit based on sign), Description, and Date.
NEW_TXN_RE = re.compile(
    r"""
    (?P<balance>-?[\d,]+\.\d{2})\s+             # Balance (e.g., 5,310.49)
    (?P<amount>-?[\d,]+\.\d{2})\s+              # Single amount (can be positive or negative)
    (?P<description>.+?)\s+                     # Description (non-greedy, everything up to the date)
    (?P<date>\d{2}/\d{2}/\d{4})$                # Transaction Date (DD/MM/YYYY) at end of line
    """,
    re.VERBOSE | re.I | re.MULTILINE
)

def parse_transactions(full_text: str) -> List[Dict[str, Any]]:
    """
    Parses transaction details from the full OCR text.
    It primarily uses the NEW_TXN_RE pattern adapted for the new PDF format.
    It then infers if the captured 'amount' is a debit or credit based on its sign.
    """
    txns: List[Dict[str, Any]] = []

    # Iterate over lines and try to match the new transaction regex
    for m in NEW_TXN_RE.finditer(full_text):
        try:
            balance = float(clean(m.group("balance")).replace(",", ""))
            amount = float(clean(m.group("amount")).replace(",", ""))
            description = clean(m.group("description"))
            date = clean(m.group("date"))

            credit = 0.0
            debit = 0.0

            # Determine if the amount is a credit or debit based on its sign
            if amount >= 0:
                credit = amount
            else:
                debit = abs(amount) # Debits are stored as positive values

            txns.append(
                {
                    # Payment Reference is not available on these lines based on this new format.
                    "payment_ref": None,
                    "date": date,
                    "description": description,
                    "credit": credit,
                    "debit": debit,
                    "balance": balance,
                }
            )
        except ValueError as ve:
            print(f"Warning: Could not parse transaction line: '{m.group(0).strip()}' - Error: {ve}")
            continue
    
    # Sort transactions by date (oldest first).
    # For robust sorting, consider converting 'date' strings to datetime objects.
    txns.reverse() # Assuming regex finds newest first, reversing to get oldest first
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
