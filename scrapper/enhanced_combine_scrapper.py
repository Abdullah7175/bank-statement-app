#!/usr/bin/env python3
"""
Enhanced combine_scrapper with automatic balance calculation
Automatically calculates opening and closing balances from transaction data when missing
"""

import time
import json
from pathlib import Path
import pdfplumber
import glob
import re

# ===== Import scraper1 functions =====
from alrajhi_large_statement import parse_header as parse_header3, parse_transactions as parse_txn3
# ===== Import scraper2 functions =====
from sbn_large import extract_bank_statement_json, extract_snb_format

from alrajhi import detect_language, parse_account_summary, parse_transactions as parse_txn1
# ===== Import scraper3 functions =====
from SNB import parse_header as parse_header2, parse_transactions as parse_txn2
# import scapper4 function
from alinma_scrapper import parse_header as parse_header4, parse_transactions as parse_txn4
#import scapper5 function
from alina_en import parse_header as parse_header5, parse_transactions as parse_txn5
# ===== Import scraper6 function =====
from alinma_advnce import parse_header as parse_header6, parse_transactions as parse_txn6

def extract_balance_from_description(description):
    """Extract balance from transaction description if available"""
    if not description:
        return None
    
    # Look for balance patterns in description
    balance_patterns = [
        r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # Matches numbers like 1,234.56
        r'(\d+(?:\.\d{2})?)',  # Matches numbers like 1234.56
    ]
    
    for pattern in balance_patterns:
        matches = re.findall(pattern, description)
        if matches:
            # Try to find the last balance-like number
            for match in reversed(matches):
                try:
                    # Remove commas and convert to float
                    balance = float(match.replace(',', ''))
                    if balance > 0 and balance < 10000000:  # Reasonable balance range
                        return balance
                except ValueError:
                    continue
    return None

def calculate_balances_from_transactions(transactions):
    """Calculate opening and closing balances from transaction data"""
    if not transactions:
        return None, None
    
    opening_balance = None
    closing_balance = None
    
    # Try to find balances in transaction descriptions
    balances_found = []
    for tx in transactions:
        if 'balance' in tx and tx['balance'] is not None:
            balances_found.append(tx['balance'])
        else:
            # Try to extract from description
            balance = extract_balance_from_description(tx.get('description', ''))
            if balance:
                balances_found.append(balance)
    
    if balances_found:
        # Sort by date if available, otherwise use order
        try:
            # Try to sort by date
            sorted_transactions = sorted(transactions, key=lambda x: x.get('date', ''))
            if sorted_transactions:
                first_tx = sorted_transactions[0]
                last_tx = sorted_transactions[-1]
                
                # For SNB statements, the balance order might be reversed
                # Check if this looks like an SNB statement
                is_snb_format = any('SNB' in str(tx.get('description', '')).upper() for tx in transactions[:5])
                
                if is_snb_format:
                    # For SNB: first transaction balance is closing, last is opening
                    if 'balance' in first_tx and first_tx['balance'] is not None:
                        closing_balance = first_tx['balance']
                    if 'balance' in last_tx and last_tx['balance'] is not None:
                        opening_balance = last_tx['balance']
                else:
                    # Standard format: first transaction balance is opening, last is closing
                    if 'balance' in first_tx and first_tx['balance'] is not None:
                        opening_balance = first_tx['balance']
                    if 'balance' in last_tx and last_tx['balance'] is not None:
                        closing_balance = last_tx['balance']
                
                # Fallback to description extraction if balance field not available
                if not opening_balance:
                    balance = extract_balance_from_description(first_tx.get('description', ''))
                    if balance:
                        opening_balance = balance
                
                if not closing_balance:
                    balance = extract_balance_from_description(last_tx.get('description', ''))
                    if balance:
                        closing_balance = balance
        except:
            # Fallback: use first and last balance found
            if len(balances_found) >= 2:
                opening_balance = balances_found[0]
                closing_balance = balances_found[-1]
            elif len(balances_found) == 1:
                opening_balance = closing_balance = balances_found[0]
    
    return opening_balance, closing_balance

def enhance_account_summary(account_summary, transactions):
    """Enhance account summary - only set to 'none' if balances were never found in PDF"""
    if not account_summary:
        account_summary = {}

    # Only set to "none" if balances are still None (meaning no balance patterns found in PDF)
    # If balances are strings (even "0.00"), they were found in the PDF and should be kept
    if account_summary.get('opening_balance') is None:
        account_summary['opening_balance'] = "none"

    if account_summary.get('closing_balance') is None:
        account_summary['closing_balance'] = "none"

    return account_summary

def run_scraper1(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join(page.extract_text(x_tolerance=2) or "" for page in pdf.pages)
            # Use the correct AlRajhi scraper (parse_account_summary instead of parse_header3)
            account_summary = parse_account_summary(full_text)
            transactions = parse_txn1(full_text)

            # Enhance account summary with calculated balances
            account_summary = enhance_account_summary(account_summary, transactions)

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
                "financial_period": account_summary.get("financial_period"),
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
            
            # Enhance account summary with calculated balances
            account_summary = enhance_account_summary(account_summary, transactions)
            
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
            
            # Enhance account summary with calculated balances
            account_summary = enhance_account_summary(account_summary, transactions)
            
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
            
            # Enhance account summary with calculated balances
            account_summary = enhance_account_summary(account_summary, transactions)
            
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
            
            # Enhance account summary with calculated balances
            account_summary = enhance_account_summary(account_summary, transactions)
            
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
            
            # Enhance account summary with calculated balances
            account_summary = enhance_account_summary(account_summary, transactions)
            
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
        
        # For SNB scraper, try to extract basic info from transactions
        account_summary = {}
        if transactions:
            # Try to find account info in transaction descriptions
            for tx in transactions[:10]:  # Check first 10 transactions
                desc = tx.get('description', '')
                if desc:
                    # Look for account number patterns
                    account_match = re.search(r'(\d{10,})', desc)
                    if account_match and not account_summary.get('account_number'):
                        account_summary['account_number'] = account_match.group(1)

            # Only set to "none" if balances are still None (meaning no balance patterns found)
            if account_summary.get('opening_balance') is None:
                account_summary['opening_balance'] = "none"
            if account_summary.get('closing_balance') is None:
                account_summary['closing_balance'] = "none"

        return {
            "pdf_file": str(Path(pdf_path).resolve()),
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_pages": None,
            "account_summary": account_summary,
            "transactions": transactions,
            "total_transactions": len(transactions) if transactions else 0,
            "customer_name": account_summary.get("customer_name"),
            "account_number": account_summary.get("account_number"),
            "iban_number": account_summary.get("iban_number"),
            "financial_period": account_summary.get("period"),
            "opening_balance": account_summary.get("opening_balance"),
            "closing_balance": account_summary.get("closing_balance"),
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
            
            # Show balance information
            if result.get("opening_balance"):
                print(f"   💰 Opening Balance: {result['opening_balance']}")
            if result.get("closing_balance"):
                print(f"   💰 Closing Balance: {result['closing_balance']}")
            
            result["processing_time"] = f"{time.time() - start:.2f}s"
            output_filename = f"output/enhanced_{Path(pdf_path).stem}_output.json"
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"📂 Saved enhanced output to {output_filename}")
        else:
            print(f"❌ All scrapers failed for {pdf_path}.")

if __name__ == "__main__":
    main()
