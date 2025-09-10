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
        "opening_balance": None,
        "closing_balance": None,
    }

    # 1. Extract Period (e.g., "Date From[18/05/2024] To[18/06/2025]" or "01/08/2024 07/02/2025")
    # Updated regex to handle different date formats
    m_period = re.search(
        r"(?:Date From\[(\d{2}/\d{2}/\d{4})\]\s*To\[(\d{2}/\d{2}/\d{4})\]|(?:Date\(Gregorian\)|ﻱﺩﻼﻴﻣ\)ﺦﻳﺭﺎﺗ)[^\n]*?(\d{2}/\d{2}/\d{4})\s*(\d{2}/\d{2}/\d{4}))",
        full_text,
        re.I | re.DOTALL
    )
    if m_period:
        if m_period.group(1) and m_period.group(2):
            hdr["period"] = f"{clean(m_period.group(1))} - {clean(m_period.group(2))}"
        elif m_period.group(3) and m_period.group(4):
            hdr["period"] = f"{clean(m_period.group(3))} - {clean(m_period.group(4))}"

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

    # 4. Extract Opening and Closing Balances
    # SNB uses a specific equation format: "21,564.91 = 316,969.09 - 337,034.00 + 1,500.00"
    # Where: End Balance = Total Credits - Total Debits + Starting Balance
    snb_balance_equation = re.search(
        r"([-\d,\.]+)\s*=\s*([-\d,\.]+)\s*-\s*([-\d,\.]+)\s*\+\s*([-\d,\.]+)",
        full_text,
        re.I
    )
    if snb_balance_equation:
        # Group 1: End Balance (Closing), Group 4: Starting Balance (Opening)
        hdr["closing_balance"] = clean(snb_balance_equation.group(1))
        hdr["opening_balance"] = clean(snb_balance_equation.group(4))
    else:
        # NEW FORMAT: Extract balances from first and last transactions in the compressed PDF format
        # Look for transaction lines with balance values
        # Pattern: date, details, amounts, balance
        transaction_balance_pattern = re.findall(
            r"(\d{2}/\d{2}/\d{4})\s+.*?\s+([-\d,\.]+\.\d{2})\s*$",
            full_text,
            re.MULTILINE
        )

        if transaction_balance_pattern:
            # Get all balance values from transactions
            balance_values = []
            for match in transaction_balance_pattern:
                try:
                    balance = float(match[1].replace(',', ''))
                    balance_values.append(balance)
                except ValueError:
                    continue

            if balance_values:
                # For compressed format, the balances appear in reverse chronological order
                # First transaction in file is most recent (closing balance)
                # Last transaction in file is oldest (opening balance)
                hdr["closing_balance"] = str(balance_values[0])  # First balance in file
                hdr["opening_balance"] = str(balance_values[-1])  # Last balance in file
                print(f"✅ Extracted balances from compressed format: Opening={hdr['opening_balance']}, Closing={hdr['closing_balance']}")

        # Fallback to individual balance extraction if transaction method fails
        if not hdr.get("opening_balance") or not hdr.get("closing_balance"):
            m_opening_balance = re.search(
                r"(?:Opening Balance|Starting Balance|ﻲﺣﺎﺘﺘﻓﻻﺍ ﺏﺎﺴﺤﻟﺍ ﺪﻴﺻﺭ|ﻲﺋﺍﺪﺘﺑﻻﺍ ﺪﻴﺻﺮﻟﺍ)\s*([-\d,\.]+)",
                full_text,
                re.I
            )
            if m_opening_balance:
                hdr["opening_balance"] = clean(m_opening_balance.group(1))

            m_closing_balance = re.search(
                r"(?:Closing Balance|End Balance|ﻝﺎﻔﻗﻹﺍ ﺪﻴﺻﺭ|ﻲﺋﺎﻬﻨﻟﺍ ﺪﻴﺻﺮﻟﺍ)\s*([-\d,\.]+)",
                full_text,
                re.I
            )
            if m_closing_balance:
                hdr["closing_balance"] = clean(m_closing_balance.group(1))

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

# NEW FORMAT: Compressed PDF transaction pattern
# Format: Date Reference/Type Amount Balance (at end of line)
COMPRESSED_TXN_RE = re.compile(
    r"""
    (?P<date>\d{2}/\d{2}/\d{4})\s+              # Date (DD/MM/YYYY)
    (?P<reference>[^\s]+)\s+                    # Reference number
    (?P<type>[A-Z]{2,})\s+                      # Transaction type (EF, JM, SD, etc.)
    (?P<amount>-?[\d,]+\.\d{2})\s+              # Transaction amount
    (?P<balance>-?[\d,]+\.\d{2})\s*$            # Running balance at end
    """,
    re.VERBOSE | re.I | re.MULTILINE,
)

def parse_transactions(full_text: str) -> List[Dict[str, Any]]:
    """
    Parses transaction details from the full OCR text.
    Handles both old Arabic/English formats and new compressed PDF format.
    """
    txns: List[Dict[str, Any]] = []

    # Keywords to help determine if an amount is a debit or credit
    # These keywords are case-insensitive.
    DEBIT_KEYWORDS = ["GOSI FEE", "outgoing transfer", "رسوم", "ﺭﺩﺎﺻ ﻞﻳﻮﺤﺗ", "ﻑﺮﺻ", "Debit", "SD", "CHARGES", "FEE"]
    CREDIT_KEYWORDS = ["incoming transfer", "إيداع", "ﺩﺭﺍﻭ ﻲﻠﺧﺍﺩ ﻞﻳﻮﺤﺗ", "ﻉﺍﺪﻳﺇ", "Credit", "EF", "JM"]

    # 1. Try COMPRESSED_TXN_RE pattern first for new compressed PDF format
    for m in COMPRESSED_TXN_RE.finditer(full_text):
        try:
            date = clean(m.group("date"))
            reference = clean(m.group("reference"))
            tx_type = clean(m.group("type"))
            amt = float(clean(m.group("amount")).replace(",", ""))
            balance = float(clean(m.group("balance")).replace(",", ""))

            # Build description from reference and type
            description = f"{reference} {tx_type}"

            debit_val = 0.0
            credit_val = 0.0

            # Determine debit/credit based on transaction type and amount sign
            if tx_type in ["SD", "CHARGES", "FEE"] or amt < 0:
                debit_val = abs(amt)
            elif tx_type in ["EF", "JM"] or amt > 0:
                credit_val = amt
            else:
                # Fallback based on amount sign
                if amt < 0:
                    debit_val = abs(amt)
                else:
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
            print(f"Warning: Could not parse compressed transaction line: '{m.group(0).strip()}' - Error: {ve}")
            continue

    # 2. Try ARABIC_TXN_RE pattern if no compressed transactions found
    if not txns:
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
                print(f"Warning: Could not parse Arabic transaction line: '{m.group(0).strip()}' - Error: {ve}")
                continue

    # 3. Fallback to English LTR if no transactions found
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

    # Sort transactions by date, oldest first for proper balance calculation
    if txns:
        try:
            # Sort by date (oldest first)
            txns.sort(key=lambda x: x["date"])
        except:
            # If sorting fails, reverse the order (assuming they came newest first)
            txns.reverse()

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
