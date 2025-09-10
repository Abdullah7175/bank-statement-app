import time
import json
from pathlib import Path
import pdfplumber
import glob


# ===== Import scraper1 functions =====
from alrajhi_large_statement import parse_header as parse_header3, parse_transactions as parse_txn3
# ===== Import scraper2 functions =====
from sbn_large import extract_bank_statement_json, extract_snb_format  # <-- rename your scraper7 file to snb_universal.py

from alrajhi import detect_language, parse_account_summary, parse_transactions as parse_txn1
# ===== Import scraper3 functions =====
from SNB import parse_header as parse_header2, parse_transactions as parse_txn2
# import scapper4 function
from alinma_scrapper import parse_header as parse_header4, parse_transactions as parse_txn4
#import scapper5 function
from alina_en import parse_header as parse_header5, parse_transactions as parse_txn5  # <-- New file
# ===== Import scraper6 functions =====
from alinma_advnce import parse_header as parse_header6, parse_transactions as parse_txn6  # <-- new file
PDF_PATH = "pdf/Alinma EN (1) (1).pdf"
OUTPUT_JSON = "combined_output.json"

def run_scraper1(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join(page.extract_text(x_tolerance=2) or "" for page in pdf.pages)
            account_summary = parse_header3(full_text)
            transactions = parse_txn3(full_text)
            return {
                "pdf_file": str(Path(pdf_path).resolve()),
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_pages": len(pdf.pages),
                "account_summary": account_summary,
                "transactions": transactions,
                "total_transactions": len(transactions),
                "customer_name": account_summary.get("customer_name"),
                "account_number": account_summary.get("account_number"),
                "iban_number": account_summary.get("iban_number"),
                "financial_period": account_summary.get("period"),
                "opening_balance": account_summary.get("opening_balance"),
                "closing_balance": account_summary.get("closing_balance"),
                "pages_processed": len(pdf.pages)
            }
    except Exception as e:
        print(f"❌ Scraper1 failed: {e}")
        return None


    
def run_scraper2(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            detected_lang = detect_language(full_text)
            account_summary = parse_account_summary(full_text)
            transactions = parse_txn1(full_text)
            return {
                "pdf_file": str(Path(pdf_path).resolve()),
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_pages": len(pdf.pages),
                "account_summary": account_summary,
                "transactions": transactions,
                "total_transactions": len(transactions),
                "detected_lang": detected_lang,
                "customer_name": account_summary.get("customer_name"),
                "account_number": account_summary.get("account_number"),
                "iban_number": account_summary.get("iban_number"),
                "financial_period": account_summary.get("financial_period"),
                "opening_balance": account_summary.get("opening_balance"),
                "closing_balance": account_summary.get("closing_balance"),
                "pages_processed": len(pdf.pages)
            }
    except Exception as e:
        print(f"❌ Scraper2 failed: {e}")
        return None


def run_scraper3(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join(page.extract_text(x_tolerance=2) or "" for page in pdf.pages)
            account_summary = parse_header2(full_text)
            transactions = parse_txn2(full_text)
            return {
                "pdf_file": str(Path(pdf_path).resolve()),
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_pages": len(pdf.pages),
                "account_summary": account_summary,
                "transactions": transactions,
                "total_transactions": len(transactions),
                "customer_name": account_summary.get("customer_name"),
                "account_number": account_summary.get("account_number"),
                "iban_number": account_summary.get("iban_number"),
                "financial_period": account_summary.get("period"),
                "opening_balance": account_summary.get("opening_balance"),
                "closing_balance": account_summary.get("closing_balance"),
                "pages_processed": len(pdf.pages)
            }
    except Exception as e:
        print(f"❌ Scraper3 failed: {e}")
        return None

def run_scraper4(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join(page.extract_text(x_tolerance=2) or "" for page in pdf.pages)
            account_summary = parse_header4(full_text)
            transactions = parse_txn4(full_text)
            return {
                "pdf_file": str(Path(pdf_path).resolve()),
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_pages": len(pdf.pages),
                "account_summary": account_summary,
                "transactions": transactions,
                "total_transactions": len(transactions),
                "customer_name": account_summary.get("customer_name"),
                "account_number": account_summary.get("account_number"),
                "iban_number": account_summary.get("iban_number"),
                "financial_period": account_summary.get("period"),
                "opening_balance": account_summary.get("opening_balance"),
                "closing_balance": account_summary.get("closing_balance"),
                "pages_processed": len(pdf.pages)
            }
    except Exception as e:
        print(f"❌ Scraper4 failed: {e}")
        return None


def run_scraper5(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join(page.extract_text(x_tolerance=2) or "" for page in pdf.pages)
            account_summary = parse_header5(full_text)
            transactions = parse_txn5(full_text)
            return {
                "pdf_file": str(Path(pdf_path).resolve()),
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_pages": len(pdf.pages),
                "account_summary": account_summary,
                "transactions": transactions,
                "total_transactions": len(transactions),
                "customer_name": account_summary.get("customer_name"),
                "account_number": account_summary.get("account_number"),
                "iban_number": account_summary.get("iban_number"),
                "financial_period": account_summary.get("period"),
                "opening_balance": account_summary.get("opening_balance"),
                "closing_balance": account_summary.get("closing_balance"),
                "pages_processed": len(pdf.pages)
            }
    except Exception as e:
        print(f"❌ Scraper5 failed: {e}")
        return None

def run_scraper6(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join(page.extract_text(x_tolerance=3, y_tolerance=3) or "" for page in pdf.pages)
            account_summary = parse_header6(full_text)
            transactions = parse_txn6(full_text)
            return {
                "pdf_file": str(Path(pdf_path).resolve()),
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_pages": len(pdf.pages),
                "account_summary": account_summary,
                "transactions": transactions,
                "total_transactions": len(transactions),
                "customer_name": account_summary.get("customer_name"),
                "account_number": account_summary.get("account_number"),
                "iban_number": account_summary.get("iban_number"),
                "financial_period": account_summary.get("period"),
                "opening_balance": account_summary.get("opening_balance"),
                "closing_balance": account_summary.get("closing_balance"),
                "pages_processed": len(pdf.pages)
            }
    except Exception as e:
        print(f"❌ Scraper6 failed: {e}")
        return None

def run_scraper7(pdf_path):
    try:
        transactions = extract_snb_format(pdf_path)
        if not transactions:
            transactions = extract_bank_statement_json(pdf_path)

        return {
            "pdf_file": str(Path(pdf_path).resolve()),
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_pages": None,
            "account_summary": {},  # scraper2 doesn't extract header
            "transactions": transactions,
            "total_transactions": len(transactions),
            "customer_name": None,
            "account_number": None,
            "iban_number": None,
            "financial_period": None,
            "opening_balance": None,
            "closing_balance": None,
            "pages_processed": None
        }
    except Exception as e:
        print(f"❌ Scraper7 failed: {e}")
        return None

def main():
    pdfs = glob.glob("pdf/*.pdf")
    for pdf_path in pdfs:
        start = time.time()
        print(f"▶ Processing {pdf_path}...")

        result = run_scraper1(pdf_path)

        if not result or result["total_transactions"] == 0:
            print("⚠ No transactions found with Scraper 1. Switching to Scraper 2...")
            result = run_scraper2(pdf_path)

        if not result or result["total_transactions"] == 0:
            print("⚠ No transactions found with Scraper 2. Switching to Scraper 3...")
            result = run_scraper3(pdf_path)

        if not result or result["total_transactions"] == 0:
            print("⚠ No transactions found with Scraper 3. Switching to Scraper 4...")
            result = run_scraper4(pdf_path)

        if not result or result["total_transactions"] == 0:
            print("⚠ No transactions found with Scraper 4. Switching to Scraper 5...")
            result = run_scraper5(pdf_path)

        if not result or result["total_transactions"] == 0:
            print("⚠ No transactions found with Scraper 5. Switching to Scraper 6...")
            result = run_scraper6(pdf_path)

        if not result or result["total_transactions"] == 0:
            print("⚠ Switching to Scraper 7...")
            result = run_scraper7(pdf_path)

        if result and result["total_transactions"] > 0:
            print(f"✅ Success with {result['total_transactions']} transactions.")
            result["processing_time"] = f"{time.time() - start:.2f}s"
            output_filename = f"output/{Path(pdf_path).stem}_output.json"
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"📂 Saved output to {output_filename}")
        else:
            print(f"❌ All scrapers failed for {pdf_path}.")


if __name__ == "__main__":
    main()
