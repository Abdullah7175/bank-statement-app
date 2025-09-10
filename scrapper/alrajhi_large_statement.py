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
PDF_PATH    = "pdf/300000010006080891288 , ALRajhi.pdf"    # <--- change if needed
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

    # 1. Extract Customer Name: It's on the line *after* "Customer Name Duration"
    # The regex looks for "Customer Name" or "Name", followed by "Duration",
    # then a newline, and captures the content on the subsequent line.
    m_customer_name = re.search(
        r"(?:Customer Name|Name)\s*Duration\s*[\n\r]+\s*([^\n]+)",
        full_text,
        re.I
    )
    if m_customer_name:
        hdr["customer_name"] = clean(m_customer_name.group(1))

    # 2. Extract Period (Duration): Gregorian dates from the line after "Duration"
    # The regex targets "Duration" or its Arabic equivalent, followed by a newline
    # and then captures the Gregorian date range (DD/MM/YYYY - DD/MM/YYYY).
    m_period = re.search(
        r"(?:Duration|ﺔﻛﺮﺣ)\s*[\n\r]+\s*(\d{2}/\d{2}/\d{4}\s*-\s*\d{2}/\d{2}/\d{4})",
        full_text,
        re.I
    )
    if m_period:
        hdr["period"] = clean(m_period.group(1))

    # 3. Extract Account Number and Currency
    # This regex looks for "Account Number" or its Arabic equivalent, optionally followed by "Currency" or its Arabic equivalent,
    # then a newline, then captures the digits for the account number, followed by spaces and the currency name.
    m_acc_curr = re.search(
        r"(?:Account Number|ﺏﺎﺴﺤﻟﺍ ﻢﻗﺭ)\s*(?:Currency|ﺔﻠﻤﻌﻟﺍ)?\s*[\n\r]+\s*(\d+)\s*([A-Za-z\s]+)",
        full_text,
        re.I
    )
    if m_acc_curr:
        hdr["account_number"] = clean(m_acc_curr.group(1))
        # Standardize "Saudi Riyal" to "SAR" for the currency field.
        currency_name = clean(m_acc_curr.group(2)).lower()
        if "riyad" in currency_name:
            hdr["currency"] = "SAR"
        else:
            hdr["currency"] = clean(m_acc_curr.group(2))
    else: # Fallback if currency not found directly with account number label
        m_currency_fallback = re.search(r"(?:Currency|ﺔﻠﻤﻌﻟﺍ)\s*([A-Za-z\s]+)", full_text, re.I)
        if m_currency_fallback:
            currency_name = clean(m_currency_fallback.group(1)).lower()
            if "riyad" in currency_name:
                hdr["currency"] = "SAR"
            else:
                hdr["currency"] = clean(m_currency_fallback.group(1))

    # 4. Extract IBAN Number
    # This regex specifically targets "IBAN" or its Arabic equivalent, then captures the Saudi IBAN format (SA followed by 22 alphanumeric characters).
    m_iban = re.search(
        r"(?:IBAN|ﻥﺎﺒﻳﺃ)\s*(SA[A-Z0-9]{22})",
        full_text,
        re.I
    )
    if m_iban:
        hdr["iban_number"] = clean(m_iban.group(1))

    # 5. Extract Opening and Closing Balances
    # This regex looks for "Opening Balance" and "Closing Balance" labels (or their Arabic equivalents) on one line,
    # followed by a newline, and then captures the two corresponding balance amounts (e.g., "0.00 SAR" or "0 SAR").
    m_balances = re.search(
        r"(?:Opening Balance|ﻲﺋﺍﺪﺘﺑﻻﺍ ﺪﻴﺻﺮﻟﺍ)\s*(?:Closing Balance|ﻲﺋﺎﻬﻨﻟﺍ ﺪﻴﺻﺮﻟﺍ)\s*[\n\r]+\s*([-\d,\.]+\s*SAR)\s*([-\d,\.]+\s*SAR)",
        full_text,
        re.I
    )
    if m_balances:
        # Clean and convert balances to float, handling cases where balance might be "0" without decimals
        opening_str = clean(m_balances.group(1)).replace("SAR", "").strip()
        closing_str = clean(m_balances.group(2)).replace("SAR", "").strip()

        try:
            # Handle both "0.00" and "0" formats
            hdr["opening_balance"] = float(opening_str.replace(",", ""))
        except ValueError:
            hdr["opening_balance"] = 0.0

        try:
            # Handle both "0.00" and "0" formats
            hdr["closing_balance"] = float(closing_str.replace(",", ""))
        except ValueError:
            hdr["closing_balance"] = 0.0

    return hdr

# ---------- TRANSACTION PATTERNS ----------
# Regex to capture the main parts of a transaction line,
# focusing on the consistent Payment Reference and Date, and then the rest of the line.
TRANSACTION_LINE_START_RE = re.compile(
    r"^(?P<payment_ref>\d{10,})\s+"                  # Payment Reference (at least 10 digits, starts line)
    r"(?P<date>\d{2}/\d{2}/\d{2})\s+"                # Date (DD/MM/YY)
    r"(?P<raw_data_after_date>.+)$",                 # Capture everything else on the line
    re.MULTILINE
)

def parse_transactions(full_text: str) -> List[Dict[str, Any]]:
    """
    Parses transaction details from the full OCR text.
    It identifies transaction lines by their starting Payment Reference and Date.
    It then parses the remaining data on the line to extract description, credit, debit, and balance.
    Due to inconsistencies in the provided sample for credit/debit/balance columns,
    this function infers credit/debit based on the sign of the *first* detected amount
    after the description, and considers the *last* detected amount as the balance.
    """
    txns: List[Dict[str, Any]] = []

    # Keywords for additional inference, if needed (less critical with explicit amounts)
    DEBIT_PHRASES = ["commission", "debit", "refund", "رسوم"]
    CREDIT_PHRASES = ["credit", "deposit", "income", "إيداع", "ﺩﺭﺍﻭ"]

    # Split the text into lines to process each line individually
    lines = full_text.split('\n')
    
    for line in lines:
        m = TRANSACTION_LINE_START_RE.match(line)
        if m:
            payment_ref = clean(m.group("payment_ref"))
            date = clean(m.group("date"))
            raw_data_after_date = clean(m.group("raw_data_after_date"))

            # Find all currency amounts (e.g., "123.45 SAR" or "-12.34 SAR")
            # This captures the numeric value and the "SAR" part
            money_amount_matches = list(re.finditer(r"(-?[\d,]+\.\d{2})\s*SAR", raw_data_after_date))

            balance = 0.0
            credit = 0.0
            debit = 0.0
            description = raw_data_after_date # Start with everything as description

            if money_amount_matches:
                # The last amount is consistently the balance
                last_amount_match = money_amount_matches[-1]
                balance = float(last_amount_match.group(1).replace(",", ""))
                
                # Remove the balance part from the description
                description = raw_data_after_date[:last_amount_match.start()].strip()

                # Process remaining amounts (potential credit/debit)
                if len(money_amount_matches) > 1:
                    # The second-to-last amount is likely the primary transaction value (credit/debit)
                    primary_transaction_amount_match = money_amount_matches[-2]
                    primary_amount = float(primary_transaction_amount_match.group(1).replace(",", ""))

                    # Remove this amount from description
                    description = description[:primary_transaction_amount_match.start()].strip()

                    # Infer if it's credit or debit based on its sign
                    if primary_amount >= 0:
                        credit = primary_amount
                    else:
                        debit = abs(primary_amount) # Debits are typically positive in output

                    # Handle cases like "6249.49 SAR 0.00 SAR" -> Credit=6249.49, Debit=0.00
                    # This implies there might be a third (first) amount which is a credit/debit.
                    # This specific pattern is tricky without fixed columns.
                    # Let's assume the first of the two (before balance) is Credit and second is Debit if present.
                    if len(money_amount_matches) > 2:
                        # This means there are three SAR amounts: [Credit] [Debit] [Balance]
                        credit_match = money_amount_matches[0]
                        debit_match = money_amount_matches[1]

                        credit = float(credit_match.group(1).replace(",", ""))
                        debit = float(debit_match.group(1).replace(",", ""))
                        
                        # Adjust description to remove these two amounts as well
                        description = raw_data_after_date[:credit_match.start()].strip() + \
                                      raw_data_after_date[credit_match.end():debit_match.start()].strip() + \
                                      raw_data_after_date[debit_match.end():last_amount_match.start()].strip()
                
                # Further refine description by removing any remaining "SAR" strings that were not part of an amount extraction
                description = re.sub(r'\s*SAR\s*', ' ', description).strip()
            
            # Additional heuristic for description, if it contains explicit credit/debit values as text
            description_lower = description.lower()
            if any(phrase in description_lower for phrase in DEBIT_PHRASES) and credit == 0 and debit == 0:
                 # If debit phrase found and amounts aren't set, try to assign based on a single amount.
                all_amounts = re.findall(r"(-?[\d,]+\.\d{2})", raw_data_after_date)
                if len(all_amounts) >= 2: # At least a transaction amount and a balance
                    potential_debit = float(clean(all_amounts[-2]).replace(",", "")) # Second last is likely the transaction amount
                    if potential_debit < 0:
                        debit = abs(potential_debit)
            elif any(phrase in description_lower for phrase in CREDIT_PHRASES) and credit == 0 and debit == 0:
                all_amounts = re.findall(r"(-?[\d,]+\.\d{2})", raw_data_after_date)
                if len(all_amounts) >= 2:
                    potential_credit = float(clean(all_amounts[-2]).replace(",", ""))
                    if potential_credit >= 0:
                        credit = potential_credit


            txns.append(
                {
                    "payment_ref": payment_ref,
                    "date": date,
                    "description": description,
                    "credit": credit,
                    "debit": debit,
                    "balance": balance,
                }
            )
    
    # Sort transactions by date (oldest first)
    # Convert 'DD/MM/YY' to sortable format or use datetime objects for robust sorting
    # For simplicity, keeping reverse() if they are extracted newest first,
    # but actual robust sorting should involve converting dates.
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
