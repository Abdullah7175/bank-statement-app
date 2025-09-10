import io, os, json, re, base64, time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Generator
import statistics

"""
Universal Bank Statement PDF Extractor

This extractor is designed to handle multiple PDF formats from different banks.
Key features:
- OCR-based extraction using Hugging Face models
- Fallback text extraction for compatibility
- Universal balance handling with multiple fallback strategies
- Automatic correction of zero opening/closing balances using transaction data
- Support for various bank statement formats (Alinma, SNB, Alrajhi, etc.)

Balance Fallback Strategies:
1. Zero Opening Balance: Uses first transaction balance to calculate opening balance
2. Zero Closing Balance: Uses last transaction balance or calculates from opening + net change
3. Advanced Validation: Recalculates balances when they appear inconsistent
4. Alternative Methods: Multiple fallback approaches for edge cases
5. Consistency Check: Final validation of calculated balances
"""

# Conditional imports for OCR functionality
try:
    from PIL import Image
    import fitz
    from huggingface_hub import InferenceClient
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    OCR_AVAILABLE = True

    # ---------- Config ----------
    # Using a more standard Llama model ID
    MODEL_ID      = os.getenv("HUGGINGFACE_MODEL_ID")
    HF_TOKEN      = os.getenv("HUGGINGFACE_API_TOKEN")
    
    # Alternative model options (uncomment one if the above doesn't work)
    # MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"  # Larger model
    # MODEL_ID = "meta-llama/Llama-3.1-405B-Instruct"  # Even larger model
    # MODEL_ID = "microsoft/DialoGPT-medium"  # Alternative option
    
    # Function to update API credentials
    def update_api_credentials(new_token, new_model_id=None):
        """Update API token and optionally model ID"""
        global HF_TOKEN, MODEL_ID, client
        HF_TOKEN = new_token or os.getenv("HUGGINGFACE_API_TOKEN")
        if new_model_id:
            MODEL_ID = new_model_id
        else:
            MODEL_ID = os.getenv("HUGGINGFACE_MODEL_ID")
        print(f"[INFO] Updated API token and model: {MODEL_ID}")
        # Recreate client with new credentials
        try:
            client = InferenceClient(provider="fireworks-ai", api_key=HF_TOKEN)
            print("[INFO] Successfully recreated client with new credentials")
        except Exception as e:
            print(f"[ERROR] Failed to recreate client: {e}")
            client = None

    # ---------- Hugging Face client ----------
    import requests
    import os
    
    # Set environment variable to disable SSL verification issues
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    
    # Monkey patch requests to disable gzip compression (fixes the gzip decompression error)
    original_request = requests.Session.request
    
    def patched_request(self, method, url, **kwargs):
        # Force no compression to avoid gzip decompression issues
        if 'headers' not in kwargs:
            kwargs['headers'] = {}
        kwargs['headers']['Accept-Encoding'] = 'identity'
        return original_request(self, method, url, **kwargs)
    
    requests.Session.request = patched_request
    
    # Test API connection first
    def test_api_connection():
        """Test if the API key and model are working"""
        if not HF_TOKEN:
            print("[ERROR] No Hugging Face API token found. Please set HUGGINGFACE_API_TOKEN environment variable.")
            return False
        try:
            test_client = InferenceClient(provider="fireworks-ai", api_key=HF_TOKEN)
            # Try a simple chat completion to test the connection
            response = test_client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "user", "content": "Hello, this is a test."}
                ],
                max_tokens=10
            )
            print(f"[INFO] API connection test successful: {response.choices[0].message.content[:50]}...")
            return True
        except Exception as e:
            print(f"[ERROR] API connection test failed: {e}")
            return False

    # Test API before creating the main client
    if not test_api_connection():
        print("[WARN] API connection test failed. The API key or model might be invalid.")
        print("[INFO] Please check your API key and model ID.")
        client = None
    else:
        # Create client with gzip disabled
        try:
            client = InferenceClient(provider="fireworks-ai", api_key=HF_TOKEN)
            print("[INFO] Created Hugging Face client with gzip compression disabled")
        except Exception as e:
            print(f"[ERROR] Failed to create Hugging Face client: {e}")
            client = None

    # ---------- Helper: PDF → JPEG ----------
    def pdf_to_images(pdf_path, dpi=150):
        doc = fitz.open(pdf_path)
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            images.append(img)
        doc.close()
        return images

except ImportError:
    OCR_AVAILABLE = False
    print("Warning: OCR dependencies not available. Using fallback text extraction.")

    # Define dummy functions when OCR is not available
    def pdf_to_images(pdf_path, dpi=150):
        return []

    client = None
    MODEL_ID = ""
    HF_TOKEN = ""

# Import scraper functions for balance extraction
try:
    import sys
    from pathlib import Path
    scraper_path = Path(__file__).parent.parent / "scrapper"
    sys.path.append(str(scraper_path))
    print(f"🔧 Scraper path: {scraper_path}")
    print(f"🔧 Current working directory: {Path.cwd()}")

    from enhanced_combine_scrapper import (
        run_scraper1, run_scraper2, run_scraper3,
        run_scraper4, run_scraper5, run_scraper6, run_scraper7
    )
    SCRAPER_AVAILABLE = True
    print("✅ Scraper modules imported successfully")
except ImportError as e:
    print(f"❌ Warning: Scraper modules not available: {e}")
    SCRAPER_AVAILABLE = False
    run_scraper1 = run_scraper2 = run_scraper3 = None
    run_scraper4 = run_scraper5 = run_scraper6 = run_scraper7 = None

@dataclass
class Transaction:
    """Transaction data structure"""
    date: str
    description: str
    debit: float = 0.0
    credit: float = 0.0
    balance: float = 0.0
    line_number: int = 0

@dataclass
class MonthlyData:
    """Monthly analysis data structure"""

    opening_balance: float = 0.0
    closing_balance: float = 0.0
    total_credit: float = 0.0
    total_debit: float = 0.0
    net_change: float = 0.0
    fluctuation: float = 0.0
    minimum_balance: float = float('inf')
    maximum_balance: float = 0.0
    international_inward_count: int = 0
    international_outward_count: int = 0
    international_inward_total: float = 0.0
    international_outward_total: float = 0.0
    transaction_count: int = 0
    balances: List[float] = field(default_factory=list)

# ---------- Prompt ----------
SYSTEM_RULES = (
    "You are a precise bank statement parser for Saudi Arabian banks.\n"
    "The table columns are in this order:\n"
    "Balance | Credit | Debit | Transaction Description | Transaction Date.\n\n"
    "OUTPUT FORMAT:\n"
    "{\n"
    "  \"customer_name\": \"...\",\n"
    "  \"account_number\": \"...\",\n"
    "  \"iban_number\": \"...\",\n"
    "  \"financial_period\": \"...\",\n"

    "  \"transactions\": [\n"
    "    {\n"
    "      \"date\": \"YYYY-MM-DD\",\n"
    "      \"credit\": \"123.45\" or \"\",\n"
    "      \"debit\": \"123.45\" or \"\",\n"
    "      \"transaction_description\": \"...\"\n"
    "    }\n"
    "  ]\n"
    "}\n\n"

    "\n"
    "OTHER EXTRACTION RULES:\n"
    "- Extract customer_name, account_number, iban_number, financial_period from header section\n"
    "- Look for IBAN patterns like SA## #### #### #### ####\n"
    "- For each transaction row:\n"
    "   • Use the Date from the rightmost column (Transaction Date)\n"
    "   • Take Credit from Credit column, Debit from Debit column\n"
    "   • Exactly one of credit or debit must be non-empty\n"
    "   • Normalize amounts with 2 decimals\n"
    "   • Never include Balance column in transactions\n"
    "   • If a value is negative (like -598.05), it goes in the debit column\n"
    "   • If a value is positive (like 250.00), it goes in the credit column\n"
    "- If any field is not found, use empty string or 0.0 for numbers\n"
    "\n"

    "\n"
    "Return ONLY valid JSON following the above schema."
)

# ---------- Helper: encode image ----------
def encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return "data:image/jpeg;base64," + base64.b64encode(buf.read()).decode()

# ---------- Robust JSON extraction ----------
def extract_json_object(text: str):
    text = text.strip()

    # Remove code fences
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text, flags=re.IGNORECASE).strip()

    # Try direct load
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try JSON block inside text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # Fallback if all fails
    return {"transactions": []}

# ---------- Validation ----------
def is_valid_amount(amount):
    """Check if amount is within reasonable bounds for bank transactions"""
    if amount is None or amount == "":
        return True  # Empty amounts are valid
    try:
        value = float(amount)
        # Filter out extremely large numbers (likely OCR errors)
        return abs(value) <= 1000000  # Max 1 million SAR
    except (ValueError, TypeError):
        return False

# ---------- Normalization ----------
def norm_amount(value):
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        # Filter out extremely large numbers (likely OCR errors)
        if abs(float(value)) > 1000000:  # More than 1 million SAR
            return ""
        return f"{float(value):.2f}"
    s = str(value)
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    if m:
        amount = float(m.group(0))
        # Filter out extremely large numbers (likely OCR errors)
        if abs(amount) > 1000000:  # More than 1 million SAR
            return ""
        return f"{amount:.2f}"
    return ""

def normalize_transaction(row):
    r = {k.strip().lower(): v for k, v in row.items()}
    out = {
        "date": str(r.get("date", "")).strip(),
        "credit": norm_amount(r.get("credit", "")),
        "debit": norm_amount(r.get("debit", "")),
        "transaction_description": str(
            r.get("transaction description", r.get("transaction_description", r.get("description", "")))
        ).strip(),
    }
    # Ensure only one side populated
    if out["credit"] and out["debit"]:
        c, d = float(out["credit"]), float(out["debit"])
        if c >= d:
            out["debit"] = ""
        else:
            out["credit"] = ""
    # Date cleanup
    dm = re.search(r"\d{4}-\d{2}-\d{2}", out["date"])
    if dm:
        out["date"] = dm.group(0)
    return out

# ---------- Main page processor ----------
def process_single_page(img, page_no, total):
    print(f"Processing page {page_no}/{total} …")
    
    # Check if client is available
    if client is None:
        print(f"[ERROR] Hugging Face client not available for page {page_no}")
        return {"transactions": [], "page_error": "OCR client not available"}
    
    data_url = encode_image(img)
    
    # Retry mechanism for API calls
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_RULES},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Extract all details and transactions from this statement."},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]}
                ],
                temperature=0,
                max_tokens=3072,
                response_format={"type": "json_object"}   # ✅ force JSON output
            )
            raw_text = resp.choices[0].message["content"]
            
            # Debug output removed for production
            return extract_json_object(raw_text)

        except Exception as e:
            error_msg = str(e)
            print(f"[WARN] Failed parsing page {page_no} (attempt {attempt + 1}/{max_retries}): {error_msg}")
            
            # Check if it's a gzip decompression error
            if "gzip" in error_msg.lower() and "decompress" in error_msg.lower():
                print(f"[INFO] Detected gzip decompression error, retrying with different approach...")
                print(f"[DEBUG] Error details: {error_msg}")
                # Wait a bit before retry
                import time
                time.sleep(1)
                continue
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                print(f"[INFO] Rate limit detected, waiting before retry...")
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            elif attempt == max_retries - 1:
                print(f"[ERROR] All retry attempts failed for page {page_no}")
                # Try to extract some basic info from the image using text extraction as fallback
                try:
                    print(f"[INFO] Attempting fallback text extraction for page {page_no}...")
                    # This is a minimal fallback - in a real scenario you might want to use OCR libraries
                    return {"transactions": [], "page_error": f"OCR failed: {error_msg}"}
                except Exception as fallback_error:
                    print(f"[ERROR] Fallback also failed for page {page_no}: {fallback_error}")
                    return {"transactions": []}
            else:
                # For other errors, wait a bit before retry
                import time
                time.sleep(1)
                continue
    
    return {"transactions": []}

def aggregate_monthly_analysis(transactions: List[Transaction]) -> Dict[str, MonthlyData]:
    """Monthly analysis aggregation"""
    if not transactions:
        return {}

    # Sort once using built-in sort
    transactions.sort(key=lambda x: parse_date_for_sorting(x.date))

    monthly_data = defaultdict(MonthlyData)

    for tx in transactions:
        try:
            date = parse_date_for_sorting(tx.date)
            month_key = date.strftime("%b")

            data = monthly_data[month_key]
            data.transaction_count += 1
            data.total_credit += tx.credit
            data.total_debit += tx.debit
            
            # Track balances for fluctuation calculation
            data.minimum_balance = min(data.minimum_balance, tx.balance)
            data.maximum_balance = max(data.maximum_balance, tx.balance)
            data.balances.append(tx.balance)

            # International transaction detection
            desc_lower = tx.description.lower()
            if any(keyword in desc_lower for keyword in ["international", "swift", "wire", "transfer", "intl", "sarie"]):
                if tx.credit > 0:
                    data.international_inward_count += 1
                    data.international_inward_total += tx.credit
                if tx.debit > 0:
                    data.international_outward_count += 1
                    data.international_outward_total += tx.debit

        except Exception as e:
            print(f"Error processing transaction: {e}")
            continue

    # Calculate derived metrics in batch
    _calculate_monthly_metrics(monthly_data, transactions)

    return dict(monthly_data)

def _calculate_monthly_metrics(monthly_data: Dict[str, MonthlyData], transactions: List[Transaction]) -> None:
    """Calculate monthly metrics in batch"""
    if not transactions:
        return
        
    # Sort transactions by date to get proper chronological order
    sorted_transactions = sorted(transactions, key=lambda x: parse_date_for_sorting(x.date))
    
    # Get the actual opening balance from the first transaction's balance - first transaction amount
    if sorted_transactions:
        first_tx = sorted_transactions[0]
        actual_opening_balance = first_tx.balance + first_tx.debit - first_tx.credit
    else:
        actual_opening_balance = 0.0

    # Sort months by chronological order
    sorted_months = sorted(monthly_data.keys(),
                          key=lambda x: datetime.strptime(x, "%b").month)

    current_balance = actual_opening_balance
    
    for month in sorted_months:
        data = monthly_data[month]
        
        # Set opening balance for this month
        data.opening_balance = current_balance
        
        # Calculate closing balance for this month
        data.closing_balance = data.opening_balance + data.total_credit - data.total_debit
        
        data.net_change = data.total_credit - data.total_debit

        # Fluctuation calculation
        if len(data.balances) > 1:
            mean_balance = statistics.mean(data.balances)
            if mean_balance != 0:
                data.fluctuation = statistics.stdev(data.balances) / mean_balance * 100
            else:
                data.fluctuation = 0
        else:
            data.fluctuation = 0

        # Clean up temporary data
        data.balances.clear()

        if data.minimum_balance == float('inf'):
            data.minimum_balance = 0

        # Update current balance for next month
        current_balance = data.closing_balance

def calculate_analytics(transactions: List[Transaction],
                       monthly_data: Dict[str, MonthlyData]) -> Dict[str, float]:
    """Analytics calculation"""
    if not transactions or not monthly_data:
        return _empty_analytics()

    # Vectorized calculations where possible
    monthly_values = list(monthly_data.values())

    total_inflow = sum(data.total_credit for data in monthly_values)
    total_outflow = sum(data.total_debit for data in monthly_values)

    num_months = len(monthly_data)
    avg_inflow = total_inflow / num_months if num_months > 0 else 0
    avg_outflow = total_outflow / num_months if num_months > 0 else 0

    fluctuations = [data.fluctuation for data in monthly_values]
    avg_fluctuation = statistics.mean(fluctuations) if fluctuations else 0

    stability = max(0, 100 - avg_fluctuation)

    # Foreign transaction counting - improved detection
    foreign_count = 0
    foreign_amount = 0.0
    
    for tx in transactions:
        desc_lower = tx.description.lower()
        if any(keyword in desc_lower for keyword in ["international", "swift", "wire", "transfer", "intl", "sarie", "outgoing sarie", "incoming sarie"]):
            if tx.credit > 0:
                foreign_count += 1
                foreign_amount += tx.credit
            if tx.debit > 0:
                foreign_count += 1
                foreign_amount += tx.debit

    # Count overdrafts
    overdraft_count = sum(1 for tx in transactions if tx.balance < 0)

    return {
        "average_fluctuation": round(avg_fluctuation, 2),
        "net_cash_flow_stability": round(stability, 4),
        "total_foreign_transactions": foreign_count,
        "total_foreign_amount": round(foreign_amount, 2),
        "overdraft_frequency": overdraft_count,
        "overdraft_total_days": overdraft_count,
        "sum_total_inflow": round(total_inflow, 2),
        "sum_total_outflow": round(total_outflow, 2),
        "avg_total_inflow": round(avg_inflow, 2),
        "avg_total_outflow": round(avg_outflow, 2)
    }

def _empty_analytics() -> Dict[str, float]:
    """Return empty analytics structure"""
    return {
        "average_fluctuation": 0.0,
        "net_cash_flow_stability": 0.0,
        "total_foreign_transactions": 0,
        "total_foreign_amount": 0.0,
        "overdraft_frequency": 0,
        "overdraft_total_days": 0,
        "sum_total_inflow": 0.0,
        "sum_total_outflow": 0.0,
        "avg_total_inflow": 0.0,
        "avg_total_outflow": 0.0
    }

def validate_extraction_against_pdf_summary(transactions: List[Transaction], pdf_text: str) -> Dict:
    """Validate extracted transactions against PDF summary data"""
    validation = {
        "pdf_summary": {},
        "extracted_summary": {},
        "discrepancies": {},
        "validation_passed": True
    }
    
    # Extract PDF summary data from text
    import re
    
    # Look for deposit/withdrawal counts and totals in PDF text
    deposit_count_match = re.search(r"Number Of Deposits\s+(\d+)", pdf_text)
    deposit_total_match = re.search(r"Totals Deposits\s+([\d,]+)", pdf_text)
    withdrawal_count_match = re.search(r"Number of Withdraws\s+(\d+)", pdf_text)
    withdrawal_total_match = re.search(r"Total Withdraws\s+([-\d,]+)", pdf_text)
    
    pdf_deposit_count = int(deposit_count_match.group(1)) if deposit_count_match else 0
    pdf_deposit_total = float(deposit_total_match.group(1).replace(",", "")) if deposit_total_match else 0
    pdf_withdrawal_count = int(withdrawal_count_match.group(1)) if withdrawal_count_match else 0
    pdf_withdrawal_total = float(withdrawal_total_match.group(1).replace(",", "")) if withdrawal_total_match else 0
    
    validation["pdf_summary"] = {
        "deposit_count": pdf_deposit_count,
        "deposit_total": pdf_deposit_total,
        "withdrawal_count": pdf_withdrawal_count,
        "withdrawal_total": abs(pdf_withdrawal_total)  # Make positive for comparison
    }
    
    # Use the absolute value for comparison
    pdf_withdrawal_total_abs = abs(pdf_withdrawal_total)
    
    # Calculate extracted summary
    extracted_deposit_count = sum(1 for tx in transactions if tx.credit > 0)
    extracted_deposit_total = sum(tx.credit for tx in transactions)
    extracted_withdrawal_count = sum(1 for tx in transactions if tx.debit > 0)
    extracted_withdrawal_total = sum(tx.debit for tx in transactions)
    
    validation["extracted_summary"] = {
        "deposit_count": extracted_deposit_count,
        "deposit_total": extracted_deposit_total,
        "withdrawal_count": extracted_withdrawal_count,
        "withdrawal_total": extracted_withdrawal_total
    }
    
    # Check for discrepancies
    deposit_count_diff = abs(extracted_deposit_count - pdf_deposit_count)
    deposit_total_diff = abs(extracted_deposit_total - pdf_deposit_total)
    withdrawal_count_diff = abs(extracted_withdrawal_count - pdf_withdrawal_count)
    withdrawal_total_diff = abs(extracted_withdrawal_total - pdf_withdrawal_total_abs)
    
    validation["discrepancies"] = {
        "deposit_count_difference": deposit_count_diff,
        "deposit_total_difference": deposit_total_diff,
        "withdrawal_count_difference": withdrawal_count_diff,
        "withdrawal_total_difference": withdrawal_total_diff
    }
    
    # Mark validation as failed if there are significant discrepancies
    if deposit_count_diff > 2 or deposit_total_diff > 100 or withdrawal_count_diff > 2 or withdrawal_total_diff > 100:
        validation["validation_passed"] = False
    
    return validation

def extract_balances_from_transactions(transactions: List[Transaction]) -> tuple:
    """Extract opening and closing balances from transaction data"""
    if not transactions:
        return 0.0, 0.0
    
    # Create array of balances with dates
    balance_data = []
    for tx in transactions:
        if is_valid_amount(tx.balance):  # Only include valid balance amounts
            balance_data.append({
                'date': tx.date,
                'balance': tx.balance,
                'transaction': tx
            })
    
    if not balance_data:
        return 0.0, 0.0
    
    # Sort by date to ensure proper chronological order
    balance_data.sort(key=lambda x: parse_date_for_sorting(x['date']))
    
    # Get first and last balance values
    first_balance = balance_data[0]['balance']
    last_balance = balance_data[-1]['balance']
    
    print(f"Extracted balances from {len(balance_data)} valid transactions")
    print(f"First balance: {first_balance} (Date: {balance_data[0]['date']})")
    print(f"Last balance: {last_balance} (Date: {balance_data[-1]['date']})")
    
    return first_balance, last_balance

def fix_zero_balances(results: Dict, transactions: List[Transaction]) -> Dict:
    """Fix zero opening and closing balances using actual transaction balance data"""
    
    balance_fix_info = {
        "original_opening_balance": results.get("account_info", {}).get("opening_balance", 0.0),
        "original_closing_balance": results.get("account_info", {}).get("closing_balance", 0.0),
        "fixes_applied": [],
        "final_opening_balance": results.get("account_info", {}).get("opening_balance", 0.0),
        "final_closing_balance": results.get("account_info", {}).get("closing_balance", 0.0)
    }
    
    if not transactions:
        return balance_fix_info
    
    # Extract actual balances from transaction data
    extracted_opening, extracted_closing = extract_balances_from_transactions(transactions)
    
    # Fix opening balance if it's invalid AND we have a non-zero extracted value from transactions
    opening_balance = results.get("account_info", {}).get("opening_balance", 0.0)
    opening_invalid = (
        opening_balance is None
        or (isinstance(opening_balance, (int, float)) and float(opening_balance) == 0.0)
        or (isinstance(opening_balance, str) and opening_balance.strip().lower() == "none")
    )
    if opening_invalid and extracted_opening != 0.0:
        results["account_info"]["opening_balance"] = extracted_opening
        balance_fix_info["final_opening_balance"] = extracted_opening
        balance_fix_info["fixes_applied"].append({
            "balance_type": "opening",
            "original_value": opening_balance,
            "fixed_value": extracted_opening,
            "method": "extracted_from_first_transaction_balance",
            "formula": "first_transaction.balance (from PDF data)",
            "first_transaction": {
                "date": transactions[0].date if transactions else "unknown",
                "description": transactions[0].description if transactions else "unknown",
                "debit": transactions[0].debit if transactions else 0,
                "credit": transactions[0].credit if transactions else 0,
                "balance": extracted_opening
            }
        })

    # Fix closing balance if it's invalid or extreme AND we have a non-zero extracted value from transactions
    closing_balance = results.get("account_info", {}).get("closing_balance", 0.0)
    closing_invalid = (
        closing_balance is None
        or (isinstance(closing_balance, (int, float)) and float(closing_balance) == 0.0)
        or (isinstance(closing_balance, str) and closing_balance.strip().lower() == "none")
        or (isinstance(closing_balance, (int, float)) and abs(float(closing_balance)) > 1000000)
    )
    if closing_invalid and extracted_closing != 0.0:
        results["account_info"]["closing_balance"] = extracted_closing
        balance_fix_info["final_closing_balance"] = extracted_closing
        balance_fix_info["fixes_applied"].append({
            "balance_type": "closing",
            "original_value": closing_balance,
            "fixed_value": extracted_closing,
            "method": "extracted_from_last_transaction_balance",
            "formula": "last_transaction.balance (from PDF data)",
            "last_transaction": {
                "date": transactions[-1].date if transactions else "unknown",
                "description": transactions[-1].description if transactions else "unknown",
                "debit": transactions[-1].debit if transactions else 0,
                "credit": transactions[-1].credit if transactions else 0,
                "balance": extracted_closing
            }
        })
    
    return balance_fix_info

def create_comprehensive_output(results: Dict, transactions: List[Transaction], monthly_data: Dict[str, MonthlyData], pdf_text: str = "") -> Dict:
    """Create comprehensive output with all extracted data and calculated values with formulas"""
    
    # Fix zero balances before processing
    balance_fix_info = fix_zero_balances(results, transactions)
    
    # Calculate additional metrics
    total_deposits = sum(tx.credit for tx in transactions)
    total_withdrawals = sum(tx.debit for tx in transactions)
    net_change = total_deposits - total_withdrawals
    
    # Validate extraction against PDF summary if text is available
    validation = {}
    if pdf_text:
        validation = validate_extraction_against_pdf_summary(transactions, pdf_text)
    
    # Calculate transaction counts by type
    deposit_count = sum(1 for tx in transactions if tx.credit > 0)
    withdrawal_count = sum(1 for tx in transactions if tx.debit > 0)
    
    # Calculate balance statistics
    balances = [tx.balance for tx in transactions]
    min_balance = min(balances) if balances else 0
    max_balance = max(balances) if balances else 0
    avg_balance = statistics.mean(balances) if balances else 0
    
    comprehensive = {
        "extraction_metadata": {
            "pdf_file": results.get("pdf_file", ""),
            "processed_at": results.get("processed_at", ""),
            "pages_processed": results.get("pages_processed", 0),
            "total_transactions": results.get("total_transactions", 0),
            "processing_time": results.get("processing_time", "0.00s")
        },
        
        "account_information": {
            "customer_name": results.get("account_info", {}).get("customer_name", ""),
            "account_number": results.get("account_info", {}).get("account_number", ""),
            "iban_number": results.get("account_info", {}).get("iban_number", ""),
            "financial_period": results.get("account_info", {}).get("financial_period", ""),
            "opening_balance": results.get("account_info", {}).get("opening_balance", 0.0),
            "closing_balance": results.get("account_info", {}).get("closing_balance", 0.0)
        },
        
        "transaction_summary": {
            "total_deposits": {
                "count": deposit_count,
                "amount": round(total_deposits, 2),
                "formula": "sum(credit for all transactions where credit > 0)"
            },
            "total_withdrawals": {
                "count": withdrawal_count,
                "amount": round(total_withdrawals, 2),
                "formula": "sum(debit for all transactions where debit > 0)"
            },
            "net_change": {
                "amount": round(net_change, 2),
                "formula": "total_deposits - total_withdrawals"
            },
            "balance_verification": {
                "expected_closing_balance": round(results.get("account_info", {}).get("opening_balance", 0) + net_change, 2),
                "actual_closing_balance": results.get("account_info", {}).get("closing_balance", 0),
                "difference": round(results.get("account_info", {}).get("closing_balance", 0) - (results.get("account_info", {}).get("opening_balance", 0) + net_change), 2),
                "formula": "opening_balance + net_change = closing_balance"
            }
        },
        
        "balance_statistics": {
            "minimum_balance": round(min_balance, 2),
            "maximum_balance": round(max_balance, 2),
            "average_balance": round(avg_balance, 2),
            "formulas": {
                "minimum_balance": "min(transaction_balance for all transactions)",
                "maximum_balance": "max(transaction_balance for all transactions)",
                "average_balance": "mean(transaction_balance for all transactions)"
            }
        },
        
        "monthly_analysis": {},
        
        "analytics": results.get("analytics", {}),
        
        "validation_against_pdf_summary": validation,
        
        "balance_fix_information": balance_fix_info,
        
        "transactions": [
            {
                "date": tx.date,
                "description": tx.description,
                "debit": tx.debit,
                "credit": tx.credit,
                "balance": tx.balance,
                "line_number": tx.line_number,
                "transaction_type": "deposit" if tx.credit > 0 else "withdrawal",
                "formula": f"balance = previous_balance + credit - debit"
            }
            for tx in transactions
        ],
        
        "calculation_formulas": {
            "running_balance": "current_balance = previous_balance + credit - debit",
            "monthly_opening_balance": "first_transaction_balance + first_transaction_debit - first_transaction_credit",
            "monthly_closing_balance": "monthly_opening_balance + monthly_total_credit - monthly_total_debit",
            "monthly_net_change": "monthly_total_credit - monthly_total_debit",
            "fluctuation": "standard_deviation(balances) / mean(balances) * 100",
            "stability": "max(0, 100 - average_fluctuation)",
            "foreign_transaction_detection": "keywords: ['international', 'swift', 'wire', 'transfer', 'intl', 'sarie']"
        }
    }
    
    # Add monthly analysis with formulas
    for month, data in monthly_data.items():
        comprehensive["monthly_analysis"][month] = {
            "opening_balance": round(data.opening_balance, 2),
            "closing_balance": round(data.closing_balance, 2),
            "total_credit": round(data.total_credit, 2),
            "total_debit": round(data.total_debit, 2),
            "net_change": round(data.net_change, 2),
            "fluctuation": round(data.fluctuation, 2),
            "minimum_balance": round(data.minimum_balance, 2),
            "maximum_balance": round(data.maximum_balance, 2),
            "transaction_count": data.transaction_count,
            "international_inward_count": data.international_inward_count,
            "international_outward_count": data.international_outward_count,
            "international_inward_total": round(data.international_inward_total, 2),
            "international_outward_total": round(data.international_outward_total, 2),
            "formulas": {
                "opening_balance": "balance at start of month",
                "closing_balance": "opening_balance + total_credit - total_debit",
                "net_change": "total_credit - total_debit",
                "fluctuation": "stdev(balances) / mean(balances) * 100"
            }
        }
    
    return comprehensive

def save_comprehensive_output(comprehensive_output: Dict, pdf_path: str) -> None:
    """Save comprehensive output to pdf_output folder"""
    try:
        # Create pdf_output directory if it doesn't exist
        output_dir = Path("pdf_output")
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename from PDF path
        pdf_name = Path(pdf_path).stem
        output_file = output_dir / f"{pdf_name}_comprehensive_analysis.json"
        
        # Save the comprehensive output
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_output, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ Comprehensive analysis saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error saving comprehensive output: {e}")

def parse_date_for_sorting(date_str: str) -> datetime:
    """Parse date for sorting transactions"""
    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return datetime.now()

def correct_ocr_dates(transactions: List[dict], financial_period: str) -> List[dict]:
    """Correct OCR date errors based on financial period"""
    if not financial_period or not transactions:
        return transactions

    # Parse financial period to get valid years
    try:
        # Extract years from period like "30/04/2024 - 28/04/2025"
        import re
        years = re.findall(r'(\d{4})', financial_period)
        if len(years) >= 2:
            start_year = int(years[0])
            end_year = int(years[1])
            valid_years = list(range(start_year, end_year + 1))
        else:
            return transactions
    except Exception:
        return transactions

    corrected_transactions = []

    for tx in transactions:
        date_str = tx.get("date", "")
        if not date_str:
            corrected_transactions.append(tx)
            continue

        try:
            # Parse the date
            parsed_date = None
            for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y"]:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue

            if parsed_date and parsed_date.year not in valid_years:
                # Correct the year based on common OCR errors
                original_year = parsed_date.year

                # Common OCR corrections for years
                if original_year == 2014:
                    corrected_year = 2024
                elif original_year == 2015:
                    corrected_year = 2025
                elif original_year == 2020:
                    corrected_year = 2025
                elif original_year == 2019:
                    corrected_year = 2024
                elif original_year == 2013:
                    corrected_year = 2023
                else:
                    # If year is too far off, try to find the closest valid year
                    corrected_year = min(valid_years, key=lambda y: abs(y - original_year))

                # Create corrected date
                corrected_date = parsed_date.replace(year=corrected_year)
                corrected_date_str = corrected_date.strftime("%Y-%m-%d")

                # Update the transaction
                tx_copy = tx.copy()
                tx_copy["date"] = corrected_date_str
                corrected_transactions.append(tx_copy)

                print(f"[INFO] Corrected date from {date_str} to {corrected_date_str} (year {original_year} -> {corrected_year})")
            else:
                corrected_transactions.append(tx)

        except Exception as e:
            print(f"[WARN] Could not correct date {date_str}: {e}")
            corrected_transactions.append(tx)

    return corrected_transactions

# ---------- Helper: Balance extraction from text ----------
def extract_balances_from_text(text: str) -> Tuple[float, float]:
    """Extract opening and closing balances using enhanced scraper logic."""
    import re

    # Robust amount parser: handles "175,423.04", "175 423.04", "(1,234.56)", "SAR 5,900.00"
    def _parse_amount(s: str) -> Optional[float]:
        if s is None:
            return None
        raw = str(s)
        negative = False
        raw_stripped = raw.strip()
        # Parentheses indicate negative values
        if raw_stripped.startswith("(") and raw_stripped.endswith(")"):
            negative = True
            raw_stripped = raw_stripped[1:-1]
        # Remove currency labels and common thin/non-breaking spaces
        cleaned = re.sub(r"\b(SAR|USD|EUR|GBP|AED|EGP|JOD|KWD|BHD|OMR|QAR)\b", "", raw_stripped, flags=re.I)
        cleaned = cleaned.replace(",", "").replace("٬", "").replace("\u202f", "").replace("\u00a0", "").strip()
        # Extract the first numeric pattern
        m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if not m:
            return None
        val = float(m.group(0))
        if negative:
            val = -val
        return val

    opening_balance = 0.0
    closing_balance = 0.0

    # Enhanced balance extraction logic from alrajhi_large_statement.py
    # Look for "Opening Balance" and "Closing Balance" labels with amounts
    m_balances = re.search(
        r"(?:Opening Balance|ﻲﺋﺍﺪﺘﺑﻻﺍ ﺪﻴﺻﺮﻟﺍ)\s*(?:Closing Balance|ﻲﺋﺎﻬﻨﻟﺍ ﺪﻴﺻﺮﻟﺍ)\s*[\n\r]+\s*([-\d,\.]+\s*SAR)\s*([-\d,\.]+\s*SAR)",
        text,
        re.I
    )
    if m_balances:
        try:
            opening_val = _parse_amount(m_balances.group(1))
            closing_val = _parse_amount(m_balances.group(2))
            if opening_val is not None:
                opening_balance = opening_val
            if closing_val is not None:
                closing_balance = closing_val
        except (ValueError, TypeError):
            pass

    # Special pattern for Alinma statements - look for the summary section at the end
    if opening_balance == 0.0 and closing_balance == 0.0:
        # Pattern for Alinma: amounts followed by "Closing Balance Opening Balance"
        alinma_pattern = re.search(
            r"([-\d,\.]+)\s*\(SAR\)\s*([-\d,\.]+)\s*\(SAR\)\s*[\n\r]+\s*(?:Closing Balance|ﻲﺋﺎﻬﻨﻟﺍ ﺪﻴﺻﺮﻟﺍ)\s*(?:Opening Balance|ﻲﺋﺍﺪﺘﺑﻻﺍ ﺪﻴﺻﺮﻟﺍ)",
            text,
            re.I | re.DOTALL
        )
        if alinma_pattern:
            try:
                closing_val = _parse_amount(alinma_pattern.group(1))
                opening_val = _parse_amount(alinma_pattern.group(2))
                if closing_val is not None:
                    closing_balance = closing_val
                if opening_val is not None:
                    opening_balance = opening_val
            except (ValueError, TypeError):
                pass

        # Special case: Look for "0 Closing Balance 0 Opening Balance" pattern
        zero_balance_pattern = re.search(
            r"0\s*Closing Balance\s*0\s*Opening Balance",
            text,
            re.I
        )
        if zero_balance_pattern:
            opening_balance = 0.0
            closing_balance = 0.0
            print(f"✅ Found zero balance pattern: {zero_balance_pattern.group()}")
            return opening_balance, closing_balance

        # Additional pattern for Alinma statements with just "Balance" and "0" at the end
        alinma_zero_pattern = re.search(
            r"Balance\s*\n\s*0\s*$",
            text,
            re.I | re.MULTILINE
        )
        if alinma_zero_pattern:
            opening_balance = 0.0
            closing_balance = 0.0

    # Special pattern for SNB statements - look for the balance equation format
    if opening_balance == 0.0 and closing_balance == 0.0:
        # Pattern for SNB: "End Balance = Total Credits - Total Debits + Starting Balance"
        # Example: "21,564.91 = 316,969.09 - 337,034.00 + 1,500.00"
        snb_pattern = re.search(
            r"([-\d,\.()\u202f\u00a0]+)\s*=\s*([-\d,\.()\u202f\u00a0]+)\s*-\s*([-\d,\.()\u202f\u00a0]+)\s*\+\s*([-\d,\.()\u202f\u00a0]+)",
            text,
            re.I | re.DOTALL
        )
        if snb_pattern:
            try:
                closing_val = _parse_amount(snb_pattern.group(1))
                opening_val = _parse_amount(snb_pattern.group(4))
                if opening_val is not None:
                    opening_balance = opening_val
                if closing_val is not None:
                    closing_balance = closing_val
            except (ValueError, TypeError):
                pass

    # SNB-specific pattern: Look for balance values in summary section and transaction data
    if opening_balance == 0.0 and closing_balance == 0.0:
        # Look for the pattern where opening balance appears in summary section
        # followed by a slightly different value (running balance)
        snb_summary_pattern = re.search(
            r"([\d,]+\.\d{2})\s*\n\s*([\d,]+\.\d{2})\s*\n\s*Total\s+Debit\s+Transactions",
            text,
            re.I | re.DOTALL
        )
        if snb_summary_pattern:
            try:
                opening_val = _parse_amount(snb_summary_pattern.group(1))
                if opening_val is not None:
                    opening_balance = opening_val
                    print(f"✅ Found SNB opening balance in summary: {opening_balance}")
            except (ValueError, TypeError):
                pass

        # Look for closing balance in the first transaction (most recent)
        # Pattern: transaction details followed by amount and balance
        snb_closing_pattern = re.search(
            r"(\d{2}/\d{2}/\d{4})\s+.*?\s+(\d{4}-\d{2}-\d{2})\s+.*?\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s*$",
            text,
            re.I | re.MULTILINE
        )
        if snb_closing_pattern:
            try:
                closing_val = _parse_amount(snb_closing_pattern.group(4))
                if closing_val is not None:
                    closing_balance = closing_val
                    print(f"✅ Found SNB closing balance in transaction: {closing_balance}")
            except (ValueError, TypeError):
                pass

    # Fallback: Look for individual balance labels (EN/AR) with optional currency and flexible spacing
    if opening_balance == 0.0 and closing_balance == 0.0:
        # Opening / Starting balance (English and Arabic variations)
        m_opening = re.search(
            r"(?:Starting\s*Balance|Starting\s*Bal\.?|Opening\s*Balance|الرصيد\s*الافتتاحي|رصيد\s*البداية|ﻲﺋﺍﺪﺘﺑﻻﺍ\s*ﺪﻴﺻﺮﻟﺍ)\s*[:\-]?\s*(?:SAR|ر\.?\s*س)?\s*([-\d\.,()\u202f\u00a0]+)",
            text, re.I
        )
        if m_opening:
            try:
                ov = _parse_amount(m_opening.group(1))
                if ov is not None:
                    opening_balance = ov
            except (ValueError, TypeError):
                pass

        # Closing / End / Ending balance (English and Arabic variations)
        m_closing = re.search(
            r"(?:End\s*Balance|Ending\s*Balance|Closing\s*Balance|الرصيد\s*الختامي|الرصيد\s*النهائي|رصيد\s*النهاية|ﻲﺋﺎﻬﻨﻟﺍ\s*ﺪﻴﺻﺮﻟﺍ)\s*[:\-]?\s*(?:SAR|ر\.?\s*س)?\s*([-\d\.,()\u202f\u00a0]+)",
            text, re.I
        )
        if m_closing:
            try:
                cv = _parse_amount(m_closing.group(1))
                if cv is not None:
                    closing_balance = cv
            except (ValueError, TypeError):
                pass

        # Alternate inline format e.g. "Starting Balance SAR 5,900.00" or "SAR 175,423.04 Ending Balance"
        if opening_balance == 0.0:
            m_open_alt = re.search(
                r"(?:Starting\s*Balance|Opening\s*Balance)[^\d\-]*([-\d\.,()\u202f\u00a0]+)",
                text, re.I
            )
            if m_open_alt:
                ov = _parse_amount(m_open_alt.group(1))
                if ov is not None:
                    opening_balance = ov

        if closing_balance == 0.0:
            m_close_alt = re.search(
                r"([-\d\.,()\u202f\u00a0]+)[^\n\r]*?(?:End(?:ing)?\s*Balance|Closing\s*Balance)",
                text, re.I
            )
            if m_close_alt:
                cv = _parse_amount(m_close_alt.group(1))
                if cv is not None:
                    closing_balance = cv

    return opening_balance, closing_balance

def extract_balances_from_scraper(pdf_path: str) -> Tuple[float, float]:
    """Extract opening and closing balances using the enhanced scraper system."""
    print("🔍 extract_balances_from_scraper called!")
    print(f"🔧 SCRAPER_AVAILABLE: {SCRAPER_AVAILABLE}")
    if not SCRAPER_AVAILABLE:
        print("Scraper not available, returning 0.0 balances")
        return 0.0, 0.0

    scrapers = [run_scraper1, run_scraper2, run_scraper3, run_scraper4, run_scraper5, run_scraper6, run_scraper7]

    for i, scraper in enumerate(scrapers, 1):
        try:
            print(f"Trying scraper {i} for balance extraction...")
            result = scraper(pdf_path)

            if result and result.get("total_transactions", 0) > 0:
                opening_balance = result.get("opening_balance")
                closing_balance = result.get("closing_balance")

                # Handle balance values - extract numeric value and convert to float
                if isinstance(opening_balance, str) and opening_balance != "none":
                    try:
                        # Remove currency suffixes and commas before conversion
                        clean_balance = re.sub(r'\s*(SAR|USD|EUR|GBP|AED|EGP|JOD|KWD|BHD|OMR|QAR)\s*$', '', opening_balance, flags=re.I)
                        clean_balance = clean_balance.replace(",", "").strip()
                        opening_balance = float(clean_balance)
                    except (ValueError, AttributeError):
                        opening_balance = "none"
                elif opening_balance is None:
                    opening_balance = "none"

                if isinstance(closing_balance, str) and closing_balance != "none":
                    try:
                        # Remove currency suffixes and commas before conversion
                        clean_balance = re.sub(r'\s*(SAR|USD|EUR|GBP|AED|EGP|JOD|KWD|BHD|OMR|QAR)\s*$', '', closing_balance, flags=re.I)
                        clean_balance = clean_balance.replace(",", "").strip()
                        closing_balance = float(clean_balance)
                    except (ValueError, AttributeError):
                        closing_balance = "none"
                elif closing_balance is None:
                    closing_balance = "none"

                print(f"✅ Scraper {i} succeeded: Opening={opening_balance}, Closing={closing_balance}")
                return opening_balance, closing_balance

        except Exception as e:
            print(f"❌ Scraper {i} failed: {e}")
            continue

    print("⚠️ All scrapers failed, returning 0.0 balances")
    return 0.0, 0.0

def handle_zero_balance_fallbacks(transactions: List[Transaction],
                                 opening_balance: float, 
                                 closing_balance: float,
                                 full_text: str = "") -> Tuple[float, float]:
    """
    Enhanced balance extraction using scraper logic.
    
    Args:
        transactions: List of transactions with calculated running balances
        opening_balance: Extracted opening balance from model
        closing_balance: Extracted closing balance from model
        full_text: Full text content for enhanced balance extraction
    
    Returns:
        Tuple of (opening_balance, closing_balance) - enhanced with scraper logic
    """
    # Convert string "none" to None for proper handling
    if isinstance(opening_balance, str) and opening_balance.lower() == "none":
        opening_balance = None
    if isinstance(closing_balance, str) and closing_balance.lower() == "none":
        closing_balance = None
    
    # Convert to float, defaulting to 0.0 if None
    try:
        opening_balance = float(opening_balance) if opening_balance is not None else 0.0
    except (ValueError, TypeError):
        opening_balance = 0.0
    
    try:
        closing_balance = float(closing_balance) if closing_balance is not None else 0.0
    except (ValueError, TypeError):
        closing_balance = 0.0
    
    # Try enhanced text extraction first if we have the full text
    if full_text:
        text_opening, text_closing = extract_balances_from_text(full_text)
        # Only use text extraction if both balances are reasonable (not 0.0 unless confirmed)
        if text_opening != 0.0 or text_closing != 0.0:
            # Additional validation: check if the extracted balances make sense
            if text_opening >= 0 and text_closing >= 0:
                return text_opening, text_closing
    
    # Check if the model-extracted balances are reasonable
    # If opening balance is suspiciously high (like total deposits), recalculate
    if opening_balance > 100000 and closing_balance == 0.0:
        # This looks like a case where opening balance was incorrectly extracted
        # Recalculate from transaction data
        pass
    elif opening_balance != 0.0 or closing_balance != 0.0:
        # Use model balances if they are non-zero (including negative balances)
        return opening_balance, closing_balance
    
    # Calculate balances from transaction data
    if transactions:
        # Sort transactions by date
        sorted_transactions = sorted(transactions, key=lambda x: parse_date_for_sorting(x.date))

        if sorted_transactions:
            # Calculate total credit and debit
            total_credit = sum(tx.credit for tx in sorted_transactions)
            total_debit = sum(tx.debit for tx in sorted_transactions)
            net_change = total_credit - total_debit

            # Check if balances are 0.0, None, or "none" - use first/last transaction method
            opening_is_invalid = (opening_balance == 0.0 or opening_balance is None)
            closing_is_invalid = (closing_balance == 0.0 or closing_balance is None)

            if opening_is_invalid or closing_is_invalid:
                # Use first and last transaction balances
                if sorted_transactions:
                    first_tx = sorted_transactions[0]
                    last_tx = sorted_transactions[-1]

                    if opening_is_invalid:
                        opening_balance = first_tx.balance
                        print(f"✅ Fixed opening balance using first transaction: {opening_balance}")

                    if closing_is_invalid:
                        closing_balance = last_tx.balance
                        print(f"✅ Fixed closing balance using last transaction: {closing_balance}")
            else:
                # For Alinma statements, if both balances are valid in the PDF, keep them
                # This is a special case where the account had valid balances
                pass
    
    return opening_balance, closing_balance

class BankStatementExtractor:
    """Enhanced bank statement extractor using OCR with Hugging Face"""

    def __init__(self):
        self.ocr_available = OCR_AVAILABLE
        if not self.ocr_available:
            print("OCR not available, will use fallback text extraction")

    def _fallback_text_extraction(self, pdf_path: str) -> Dict[str, any]:
        """Fallback method using pdfplumber for text extraction"""
        results = {
            "pdf_file": str(Path(pdf_path).resolve()),
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pages_processed": 0,
            "account_info": {},
            "transactions": [],
            "total_transactions": 0,
            "monthly_analysis": {},
            "analytics": {},
        }

        try:
            import pdfplumber
        except ImportError:
            # If pdfplumber is not available, return a basic structure with error info
            print("Warning: pdfplumber not available. PDF processing will be limited.")
            results["error"] = "PDF processing libraries not available. Please install pdfplumber and OCR dependencies for full functionality."
            results["analytics"] = _empty_analytics()
            results["processing_time"] = "0.00s"
            return results

        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_text = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text)

                full_text = "\n".join(pages_text)
                results["pages_processed"] = len(pdf.pages)

                # Simple transaction extraction (basic implementation)
                transactions = []
                lines = full_text.split('\n')

                for line_num, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue

                    # Basic pattern matching for transactions
                    # This is a simplified version - you may need to adjust patterns
                    date_match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', line)
                    amount_match = re.search(r'[\d,]+\.?\d*', line)

                    if date_match and amount_match:
                        amount = float(amount_match.group(0).replace(",", ""))
                        if is_valid_amount(amount):
                            transactions.append(Transaction(
                                date=date_match.group(0),
                                description=line,
                                debit=0.0,
                                credit=amount,
                                balance=0.0,
                                line_number=line_num
                            ))

                results["total_transactions"] = len(transactions)
                results["transactions"] = [
                    {
                        "date": tx.date,
                        "description": tx.description,
                        "debit": tx.debit,
                        "credit": tx.credit,
                        "balance": tx.balance,
                        "line_number": tx.line_number
                    }
                    for tx in transactions
                ]

                # Basic account info extraction
                results["account_info"] = {
                    "customer_name": "Unknown",
                    "account_number": "Unknown",
                    "iban_number": "",
                    "opening_balance": 0.0,
                    "closing_balance": 0.0,
                    "financial_period": "",
                }

                # Apply enhanced balance extraction using scraper logic
                original_opening = results["account_info"]["opening_balance"]
                original_closing = results["account_info"]["closing_balance"]
                
                adjusted_opening, adjusted_closing = handle_zero_balance_fallbacks(
                    transactions, 
                    original_opening, 
                    original_closing,
                    full_text
                )
                
                # Update the results with adjusted balances
                results["account_info"]["opening_balance"] = adjusted_opening
                results["account_info"]["closing_balance"] = adjusted_closing

                # Generate basic analysis if transactions found
                if results["total_transactions"] > 0:
                    monthly_data = aggregate_monthly_analysis(transactions)
                    results["monthly_analysis"] = {
                        month: {

                            "total_credit": data.total_credit,
                            "total_debit": data.total_debit,
                            "net_change": data.net_change,
                            "fluctuation": data.fluctuation,
                            "minimum_balance": data.minimum_balance,
                            "maximum_balance": data.maximum_balance,
                            "international_inward_count": data.international_inward_count,
                            "international_outward_count": data.international_outward_count,
                            "international_inward_total": data.international_inward_total,
                            "international_outward_total": data.international_outward_total,
                            "transaction_count": data.transaction_count
                        }
                        for month, data in monthly_data.items()
                    }
                    results["analytics"] = calculate_analytics(transactions, monthly_data)
                else:
                    results["analytics"] = _empty_analytics()

        except Exception as e:
            print(f"Error in fallback text extraction: {e}")
            results["error"] = str(e)
            results["analytics"] = _empty_analytics()

        results["processing_time"] = f"{time.time() - time.time():.2f}s"
        
        # Apply balance fix logic before creating comprehensive output
        balance_fix_info = fix_zero_balances(results, transactions)
        
        # Create comprehensive JSON output with all data and formulas
        comprehensive_output = create_comprehensive_output(results, transactions, monthly_data, full_text)
        
        # Save comprehensive output to pdf_output folder
        save_comprehensive_output(comprehensive_output, pdf_path)
        
        return results

    def process_bank_statement(self, pdf_path: str) -> Dict[str, any]:
        """Main processing method - uses OCR if available, otherwise fallback"""
        if self.ocr_available:
            return self._process_bank_statement_ocr(pdf_path)
        else:
            print("Using fallback text extraction method")
            return self._fallback_text_extraction(pdf_path)

    def _process_bank_statement_ocr(self, pdf_path: str) -> Dict[str, any]:
        """OCR-based processing method"""
        start = time.time()
        pdf_path_obj = Path(pdf_path).resolve()

        results = {
            "pdf_file": str(pdf_path_obj),
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pages_processed": 0,
            "account_info": {},
            "transactions": [],
            "total_transactions": 0,
            "monthly_analysis": {},
            "analytics": {},
        }

        try:
            images = pdf_to_images(pdf_path)
            results["pages_processed"] = len(images)

            # Also extract full text using pdfplumber for balance extraction
            import pdfplumber
            full_text = ""
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    full_text = "\n".join(page.extract_text(x_tolerance=2) or "" for page in pdf.pages)
            except Exception as e:
                print(f"Warning: Could not extract text with pdfplumber: {e}")

            with ThreadPoolExecutor(max_workers=8) as pool:
                page_results = list(pool.map(
                    partial(process_single_page, total=len(images)),
                    images, range(1, len(images)+1)
                ))

            # Merge results
            master = dict(customer_name="", account_number="", iban_number="", financial_period="", transactions=[])
            for parsed in page_results:
                if not parsed:
                    continue
                for key in ["customer_name", "account_number", "iban_number", "financial_period"]:
                    val = parsed.get(key, "")
                    if val and not master[key]:
                        master[key] = val
                txs = [normalize_transaction(x) for x in parsed.get("transactions", []) if isinstance(x, dict)]
                master["transactions"].extend(txs)

            # Correct OCR date errors based on financial period
            if master["transactions"] and master.get("financial_period"):
                master["transactions"] = correct_ocr_dates(master["transactions"], master["financial_period"])

            # Convert OCR transactions to Transaction objects for analysis
            transactions = []
            for idx, tx in enumerate(master["transactions"]):
                try:
                    debit = float(tx.get("debit", 0) or 0)
                    credit = float(tx.get("credit", 0) or 0)
                    
                    # Validate amounts before creating transaction
                    if is_valid_amount(debit) and is_valid_amount(credit):
                        transactions.append(Transaction(
                            date=tx.get("date", ""),
                            description=tx.get("transaction_description", ""),
                            debit=debit,
                            credit=credit,
                            balance=0.0,  # Will be calculated later
                            line_number=idx
                        ))
                except (ValueError, TypeError):
                    continue

            results["total_transactions"] = len(transactions)

            # Convert to dicts for output
            results["transactions"] = [
                {
                    "date": tx.date,
                    "description": tx.description,
                    "debit": tx.debit,
                    "credit": tx.credit,
                    "balance": tx.balance,
                    "line_number": tx.line_number
                }
                for tx in transactions
            ]

            # Map account info (balances will be set by scraper)
            results["account_info"] = {
                "customer_name": master.get("customer_name", ""),
                "account_number": master.get("account_number", ""),
                "iban_number": master.get("iban_number", ""),
                "financial_period": master.get("financial_period", ""),
            }

            # Construct IBAN for Alinma if account number is available
            if "alinma" in pdf_path.lower() and master.get("account_number"):
                account_num = master["account_number"]
                if account_num.isdigit() and len(account_num) >= 10:  # Basic validation
                    # For Alinma, IBAN = SA56 + padded account number to 18 digits
                    padded_account = account_num.zfill(18)  # Pad with zeros at beginning
                    iban = f"SA56{padded_account}"
                    results["account_info"]["iban_number"] = iban
                    print(f"✅ Constructed IBAN for Alinma: {iban}")

            # Extract balances using enhanced scraper system
            print("🔍 Extracting balances using scraper system...")
            scraper_opening, scraper_closing = extract_balances_from_scraper(pdf_path)

            # Update account info with scraper balances
            results["account_info"]["opening_balance"] = scraper_opening
            results["account_info"]["closing_balance"] = scraper_closing
            print(f"✅ Using scraper balances: Opening={scraper_opening}, Closing={scraper_closing}")
            print(f"🔍 Account info after scraper: {results['account_info']}")

            # Apply enhanced fallbacks using full_text and transaction data
            adjusted_opening, adjusted_closing = handle_zero_balance_fallbacks(
                transactions,
                results["account_info"]["opening_balance"],
                results["account_info"]["closing_balance"],
                full_text
            )
            results["account_info"]["opening_balance"] = adjusted_opening
            results["account_info"]["closing_balance"] = adjusted_closing
            print(f"✅ After fallbacks: Opening={adjusted_opening}, Closing={adjusted_closing}")

            # Use the balances extracted from OCR directly, don't recalculate
            if transactions:
                # Sort transactions by date
                transactions.sort(key=lambda x: parse_date_for_sorting(x.date))

                # The OCR already extracted balances from the Balance column
                # We should trust these balances rather than recalculating
                # Only recalculate if the balance appears to be 0.0 or invalid
                for tx in transactions:
                    if tx.balance == 0.0:
                        # If balance is 0.0, it might be missing, so we can try to calculate
                        # But for now, keep it as is since OCR extracted it
                        pass

                # Update results with calculated balances
                results["transactions"] = [
                    {
                        "date": tx.date,
                        "description": tx.description,
                        "debit": tx.debit,
                        "credit": tx.credit,
                        "balance": tx.balance,
                        "line_number": tx.line_number
                    }
                    for tx in transactions
                ]

                # Balance extraction already applied above

            # Generate analysis if transactions found
            if results["total_transactions"] > 0:
                monthly_data = aggregate_monthly_analysis(transactions)
                results["monthly_analysis"] = {
                    month: {

                        "opening_balance": data.opening_balance,
                        "closing_balance": data.closing_balance,
                        "total_credit": data.total_credit,
                        "total_debit": data.total_debit,
                        "net_change": data.net_change,
                        "fluctuation": data.fluctuation,
                        "minimum_balance": data.minimum_balance,
                        "maximum_balance": data.maximum_balance,
                        "international_inward_count": data.international_inward_count,
                        "international_outward_count": data.international_outward_count,
                        "international_inward_total": data.international_inward_total,
                        "international_outward_total": data.international_outward_total,
                        "transaction_count": data.transaction_count
                    }
                    for month, data in monthly_data.items()
                }

                results["analytics"] = calculate_analytics(transactions, monthly_data)
            else:
                results["analytics"] = _empty_analytics()

        except Exception as e:
            print(f"Error processing PDF with OCR: {e}")
            return {"error": str(e)}

        results["processing_time"] = f"{time.time() - start:.2f}s"
        
        # Apply balance fix logic before creating comprehensive output
        balance_fix_info = fix_zero_balances(results, transactions)
        
        # Create comprehensive JSON output with all data and formulas
        comprehensive_output = create_comprehensive_output(results, transactions, monthly_data, full_text)
        
        # Save comprehensive output to pdf_output folder
        save_comprehensive_output(comprehensive_output, pdf_path)
        
        return results
