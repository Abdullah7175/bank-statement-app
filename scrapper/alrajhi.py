
import pdfplumber
import re
import json
from pathlib import Path
import time

# ================= USER CONFIG =================
# Note: The "+" in the filename might cause issues. It's best to rename it.
PDF_PATH      = "pdf/300000010006080891288 , ALRajhi.pdf"
OUTPUT_JSON   = "extracted_statement.json"
# ==============================================

LANGUAGE_SIGNALS = {
    "en": {"Customer Name", "City", "Account Number", "IBAN"},
    "ar": {"اسم العميل", "المدينة", "رقم الحساب", "رقم الآيبان"}
}

FIELD_PATTERNS = {
    "en": {
        "customer_name"   : r"Customer Name\s+([^\n]+)",
        "city"            : r"City\s+([^\n]+)",
        "account_number"  : r"Account Number\s+(\d+)",
        "iban_number"     : r"IBAN Number\s+([A-Z0-9]+)",
        "opening_balance" : r"Opening Balance\s+([\d,]+(?:\.\d{2})?)\s*SAR",
        "closing_balance" : r"Closing Balance\s+([\d,]+(?:\.\d{2})?)\s*SAR",
        "financial_period": r"On The Period\s+([\d/]+\s*-\s*[\d/]+)"
    },
    "ar": {
        "customer_name"   : r"(?:اسم العميل|Customer Name)\s+([^\n]+)",
        "city"            : r"(?:المدينة|City)\s+([^\n]+)",
        "account_number"  : r"(?:رقم الحساب|Account Number)\s+(\d+)",
        "iban_number"     : r"(?:رقم الآيبان|IBAN Number)\s+([A-Z0-9]+)",
        "opening_balance" : r"(?:الرصيد.*?الإفتتاحي|Opening Balance)\s+([\d,]+(?:\.\d{2})?)\s+SAR",
        "closing_balance" : r"(?:الرصيد.*?الإقفال|Closing Balance)\s+([\d,]+(?:\.\d{2})?)\s+SAR",
        "financial_period": r"(?:خلال الفترة|On The Period)\s+([\d/]+\s*-\s*[\d/]+)"
    }
}

TRANSACTION_LINE_RE_EN = re.compile(
    r"^(\d{2,4}[/-:]\d{2}[/-:]\d{2,4})\s+"  # date
    r"([\d,]+\.\d{2})\s*S?A?R?\s*"
    r"([\d,]+\.\d{2})\s*S?A?R?\s*"
    r"([\d,]+\.\d{2})\s*S?A?R?\s*$",
    re.IGNORECASE
)

# ---------- Arabic helpers ----------
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def normalize_arabic_text(txt):
    return re.sub(r"\s+", " ", txt.translate(ARABIC_DIGITS).strip())

# -----------------------------------------------------------
def detect_language(text: str) -> str:
    # (function remains unchanged)
    customer_name = None
    matched_label = None
    
    ARABIC_CHARS_PATTERN = r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
    ENGLISH_CHARS_PATTERN = r"[A-Za-z]"

    arabic_match = re.search(FIELD_PATTERNS["ar"]["customer_name"], text)
    if arabic_match:
        customer_name_raw = arabic_match.group(1).strip()
        matched_label = "ar"
        if "اسم العميل" in customer_name_raw:
            parts = customer_name_raw.split("اسم العميل")
            customer_name = " ".join(parts[::-1]).strip()
        else:
            customer_name = customer_name_raw

    elif (english_match := re.search(FIELD_PATTERNS["en"]["customer_name"], text, re.IGNORECASE)):
        matched_label = "en"
        customer_name = english_match.group(1).strip()

    if customer_name:
        arabic_chars = len(re.findall(ARABIC_CHARS_PATTERN, customer_name))
        english_chars = len(re.findall(ENGLISH_CHARS_PATTERN, customer_name))
        if arabic_chars > english_chars:
            return "ar"
        else:
            return "en"

    arabic_count = len(re.findall(ARABIC_CHARS_PATTERN, text))
    english_count = len(re.findall(ENGLISH_CHARS_PATTERN, text))
    if arabic_count > english_count:
        return "ar"
    else:
        return "en"


def extract_balances_from_alrajhi_transactions(full_text: str) -> tuple:
    """Extract opening and closing balances from AlRajhi compressed format transactions"""
    opening_balance = 0.0
    closing_balance = 0.0

    lines = full_text.split('\n')

    # Find first and last transactions in original file order
    first_transaction_balance = None
    last_transaction_balance = None

    for line in lines:
        line = line.strip()
        if not line.startswith('300000010006080891') or 'SAR' not in line:
            continue

        amounts = re.findall(r'-?[\d,]+\.\d{2}', line)
        if len(amounts) >= 2:
            # For this format, the LAST amount is the balance
            balance = float(amounts[-1].replace(',', ''))

            # First transaction found
            if first_transaction_balance is None:
                first_transaction_balance = balance

            # Keep updating last transaction
            last_transaction_balance = balance

    if first_transaction_balance is not None:
        opening_balance = first_transaction_balance

    if last_transaction_balance is not None:
        closing_balance = last_transaction_balance

    return opening_balance, closing_balance

def parse_account_summary(full_text: str) -> dict:
    # Check if this is the compressed AlRajhi format
    is_compressed_format = '300000010006080891' in full_text and 'Point of Sale' in full_text

    lang = detect_language(full_text)
    patterns = FIELD_PATTERNS[lang]
    summary = {}

    for key, pattern in patterns.items():
        m = re.search(pattern, full_text, re.IGNORECASE if lang == 'en' else 0)
        if m:
            val = m.group(1).strip()
            if key == "customer_name" and lang == "ar":
                if "اسم العميل" not in val:
                    summary[key] = val
                else:
                    parts = val.split("اسم العميل")
                    summary[key] = " ".join(parts[::-1]).strip()
            elif key in {"opening_balance", "closing_balance"}:
                try:
                    # Handle both "0.00" and "0" formats, and remove any currency suffixes
                    clean_val = val.replace(",", "").replace("SAR", "").strip()
                    summary[key] = float(clean_val)
                except (ValueError, TypeError):
                    # If conversion fails, set to 0.0 (assuming it's a zero balance)
                    summary[key] = 0.0
            else:
                summary[key] = val

    # For compressed format, if balances are 0.0 or None, extract from transactions
    if is_compressed_format:
        if (summary.get("opening_balance") == 0.0 or summary.get("opening_balance") is None or
            summary.get("closing_balance") == 0.0 or summary.get("closing_balance") is None):

            print("🔍 AlRajhi scraper returned zero balances, extracting from transactions...")
            tx_opening, tx_closing = extract_balances_from_alrajhi_transactions(full_text)

            if summary.get("opening_balance") == 0.0 or summary.get("opening_balance") is None:
                summary["opening_balance"] = tx_opening
                print(f"✅ Fixed opening balance: {tx_opening}")

            if summary.get("closing_balance") == 0.0 or summary.get("closing_balance") is None:
                summary["closing_balance"] = tx_closing
                print(f"✅ Fixed closing balance: {tx_closing}")

    return summary

def parse_transactions_en(full_text: str) -> list:
    # (function remains unchanged)
    transactions, desc_lines = [], []
    for line in full_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = TRANSACTION_LINE_RE_EN.match(line)
        if m:
            date, debit, credit, balance = m.groups()
            transactions.append({
                "date": date,
                "description": " ".join(desc_lines).strip(),
                "debit": float(debit.replace(",", "")),
                "credit": float(credit.replace(",", "")),
                "balance": float(balance.replace(",", ""))
            })
            desc_lines.clear()
        else:
            desc_lines.append(line)
    return transactions

# ----------------- REVISED ARABIC TRANSACTION PARSER -----------------
def parse_transactions_ar(full_text: str) -> list:
    transactions = []
    normalized_text = normalize_arabic_text(full_text)
    
    # Match:
    # 1. Date (Arabic or Western digits, multiple formats)
    # 2. Description (greedy until amounts)
    # 3. Debit, Credit, Balance (with or without SAR / ر.س)
    TRANSACTION_RE_AR = re.compile(
        r"(\d{2,4}[/-]\d{2}[/-]\d{2,4})"                # Date
        r"\s+(.*?)\s+"                                   # Description
        r"([\d,]+\.\d{2})\s*(?:SAR|ر\.?س)?\s+"           # Debit
        r"([\d,]+\.\d{2})\s*(?:SAR|ر\.?س)?\s+"           # Credit
        r"([\d,]+\.\d{2})\s*(?:SAR|ر\.?س)?",             # Balance
        re.MULTILINE
    )
    
    for match in TRANSACTION_RE_AR.finditer(normalized_text):
        date, desc, debit, credit, balance = match.groups()
        
        transactions.append({
            "التاريخ": date.strip(),
            "تفاصيل العملية": desc.strip(),
            "مدين": float(debit.replace(",", "")),
            "دائن": float(credit.replace(",", "")),
            "الرصيد": float(balance.replace(",", ""))
        })
    
    return transactions

# ---------------------------------------------------------------------

def parse_transactions_alrajhi_compressed(full_text: str) -> list:
    """Parse transactions for AlRajhi compressed PDF format with account number prefix"""
    transactions = []

    lines = full_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line or not line.startswith('300000010006080891'):
            continue

        # Skip non-transaction lines (like headers)
        if 'Saudi Riyal' in line or 'SAR' not in line:
            continue

        # Extract amounts from the line
        amounts = re.findall(r'-?[\d,]+\.\d{2}', line)
        if len(amounts) < 2:
            continue

        # Parse the transaction line
        parts = line.split()
        if len(parts) < 3:
            continue

        # Extract date (second part after account number)
        date_str = parts[1]

        # Build description from remaining parts (excluding account number, date, and amounts)
        description_parts = []
        i = 2  # Start after account number and date
        while i < len(parts) and not re.match(r'-?[\d,]+\.\d{2}', parts[i]):
            description_parts.append(parts[i])
            i += 1

        description = ' '.join(description_parts)

        # Extract amounts
        # For this format: amount SAR balance SAR
        # The LAST amount is the balance
        balance = float(amounts[-1].replace(',', ''))

        # The second-to-last amount is the transaction amount
        transaction_amount = float(amounts[-2].replace(',', ''))

        # Determine debit/credit based on amount sign and description
        debit = 0.0
        credit = 0.0

        if transaction_amount < 0:
            debit = abs(transaction_amount)
        else:
            credit = transaction_amount

        # Additional logic based on description keywords
        desc_lower = description.lower()
        if 'transfer within rajhi' in desc_lower and transaction_amount < 0:
            debit = abs(transaction_amount)
            credit = 0.0
        elif 'transfer within rajhi' in desc_lower and transaction_amount > 0:
            debit = 0.0
            credit = transaction_amount

        transactions.append({
            "date": date_str,
            "description": description,
            "debit": debit,
            "credit": credit,
            "balance": balance
        })

    # IMPORTANT: Keep original order for balance extraction
    # Do NOT sort transactions by date - we need the original file order
    # to get the correct opening/closing balances

    return transactions

def parse_transactions(full_text: str) -> list:
    # Check if this is the compressed AlRajhi format
    if '300000010006080891' in full_text and 'Point of Sale' in full_text:
        return parse_transactions_alrajhi_compressed(full_text)

    # Use original parsers for other formats
    lang = detect_language(full_text)
    if lang == "en":
        return parse_transactions_en(full_text)
    else:
        return parse_transactions_ar(full_text)

# -----------------------------------------------------------
def main():
    start = time.time()
    results = {
        "pdf_file": str(Path(PDF_PATH).resolve()),
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_pages": 0,
        "account_summary": {},
        "transactions": []
    }

    try:
        with pdfplumber.open(PDF_PATH) as pdf:
            full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            results["total_pages"] = len(pdf.pages)
            
            detected_lang = detect_language(full_text)
            
            results["account_summary"] = parse_account_summary(full_text)
            
            # Use the corrected parser for transactions
            results["transactions"] = parse_transactions(full_text)

            print(f"📝 Detected language: {detected_lang}")
            print(f"📝 Transactions extracted: {len(results['transactions'])}")

    except FileNotFoundError:
        print(f"❌ PDF file not found: {PDF_PATH}")
        return
    except Exception as e:
        print(f"❌ Processing error: {e}")
        return

    results["total_transactions"] = len(results["transactions"])
    results["processing_time"]    = f"{time.time() - start:.2f}s"

    try:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ Data saved to {OUTPUT_JSON}")
    except IOError as e:
        print(f"❌ Error writing JSON: {e}")

if __name__ == "__main__":
    main()
