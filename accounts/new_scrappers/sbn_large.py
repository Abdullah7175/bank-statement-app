import pdfplumber
import re
import json
from datetime import datetime
import sys

def clean_text(text):
    """
    Clean and normalize text (Arabic/English)
    """
    if not text:
        return None
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    # Remove common noise characters but keep Arabic and English
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFFa-zA-Z0-9\s\.\,\-\:\(\)\/]', '', text)
    
    return text.strip() if text.strip() else None

def extract_monetary_amount(text):
    """
    Extract monetary amounts from text
    """
    if not text:
        return None
    
    # Pattern for monetary amounts (with or without commas, with or without decimals)
    patterns = [
        r'(\d{1,3}(?:,\d{3})*\.\d{2})',  # 1,234.56
        r'(\d{1,3}(?:,\d{3})+)',        # 1,234
        r'(\d+\.\d{2})',                 # 123.45
        r'(\d{4,})'                      # 1234 (4+ digits)
    ]
    
    amounts = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                # Clean and convert to float
                clean_amount = match.replace(',', '')
                amount = float(clean_amount)
                if amount > 0:
                    amounts.append(amount)
            except ValueError:
                continue
    
    return amounts

def determine_transaction_type(text):
    """
    Determine if transaction is credit or debit based on text content
    """
    if not text:
        return None, None
    
    text_lower = text.lower()
    
    # Credit indicators (incoming money)
    credit_keywords = [
        'incoming', 'deposit', 'credit', 'ef', 'transfer in', 'received',
        'salary', 'refund', 'interest', 'dividend'
    ]
    
    # Debit indicators (outgoing money)  
    debit_keywords = [
        'outgoing', 'withdrawal', 'debit', 'charge', 'fee', 'payment',
        'transfer out', 'purchase', 'atm', 'pos', 'online', 'mobile'
    ]
    
    # Arabic keywords (common banking terms)
    arabic_credit_indicators = ['وارد', 'ايداع', 'تحويل وارد', 'راتب']
    arabic_debit_indicators = ['صادر', 'سحب', 'تحويل صادر', 'رسوم', 'دفع']
    
    is_credit = (any(keyword in text_lower for keyword in credit_keywords) or
                 any(keyword in text for keyword in arabic_credit_indicators))
    
    is_debit = (any(keyword in text_lower for keyword in debit_keywords) or
                any(keyword in text for keyword in arabic_debit_indicators))
    
    return is_credit, is_debit

def extract_bank_statement_json(pdf_path):
    """
    Extract bank statement data and return in specified JSON format
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of transaction dictionaries
    """
    
    transactions = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                print(f"Processing page {page_num}...")
                
                # Try table extraction first
                tables = page.extract_tables()
                
                if tables:
                    # Process table data
                    for table in tables:
                        for row_idx, row in enumerate(table):
                            if row_idx == 0:  # Skip header
                                continue
                                
                            if not any(cell for cell in row if cell and cell.strip()):
                                continue  # Skip empty rows
                            
                            # Combine all row data
                            row_text = ' '.join([str(cell) if cell else '' for cell in row])
                            
                            # Extract date
                            date_match = re.search(r'(\d{2}[/\-]\d{2}[/\-]\d{4})', row_text)
                            date_str = None
                            if date_match:
                                date_raw = date_match.group(1)
                                try:
                                    # Convert to YYYY-MM-DD format
                                    if '/' in date_raw:
                                        day, month, year = date_raw.split('/')
                                    else:
                                        day, month, year = date_raw.split('-')
                                    date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                                except:
                                    date_str = None
                            
                            if not date_str:
                                continue
                            
                            # Extract amounts
                            amounts = extract_monetary_amount(row_text)
                            
                            # Determine transaction type
                            is_credit, is_debit = determine_transaction_type(row_text)
                            
                            # Assign credit/debit amounts
                            credit_amount = None
                            debit_amount = None
                            
                            if amounts:
                                if is_credit and not is_debit:
                                    credit_amount = amounts[0]
                                elif is_debit and not is_credit:
                                    debit_amount = amounts[0]
                                elif len(amounts) >= 2:
                                    # If we can't determine type, use position logic
                                    # Typically: [transaction_amount, balance] or [credit, debit, balance]
                                    if len(amounts) >= 3:
                                        credit_amount = amounts[0] if amounts[0] != amounts[-1] else None
                                        debit_amount = amounts[1] if amounts[1] != amounts[-1] else None
                                    else:
                                        # Single transaction amount
                                        if is_credit:
                                            credit_amount = amounts[0]
                                        else:
                                            debit_amount = amounts[0]
                                else:
                                    # Default to debit if unclear
                                    debit_amount = amounts[0]
                            
                            # Clean description
                            description = clean_text(row_text)
                            
                            transaction = {
                                "date": date_str,
                                "description": description,
                                "debit": debit_amount,
                                "credit": credit_amount
                            }
                            
                            transactions.append(transaction)
                
                else:
                    # Fallback to text extraction
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    lines = text.split('\n')
                    
                    for i, line in enumerate(lines):
                        # Look for date patterns
                        date_match = re.search(r'(\d{2}[/\-]\d{2}[/\-]\d{4})', line)
                        
                        if date_match:
                            date_raw = date_match.group(1)
                            try:
                                # Convert to YYYY-MM-DD format
                                if '/' in date_raw:
                                    day, month, year = date_raw.split('/')
                                else:
                                    day, month, year = date_raw.split('-')
                                date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                            except:
                                continue
                            
                            # Combine current line with next 2 lines for context
                            context_lines = lines[i:i+3]
                            combined_text = ' '.join(context_lines)
                            
                            # Extract amounts
                            amounts = extract_monetary_amount(combined_text)
                            
                            # Determine transaction type
                            is_credit, is_debit = determine_transaction_type(combined_text)
                            
                            # Assign amounts
                            credit_amount = None
                            debit_amount = None
                            
                            if amounts:
                                if is_credit and not is_debit:
                                    credit_amount = amounts[0]
                                elif is_debit and not is_credit:
                                    debit_amount = amounts[0]
                                else:
                                    # Use first amount as transaction amount
                                    if is_credit:
                                        credit_amount = amounts[0]
                                    else:
                                        debit_amount = amounts[0]
                            
                            # Clean description
                            description = clean_text(combined_text)
                            
                            transaction = {
                                "date": date_str,
                                "description": description,
                                "debit": debit_amount,
                                "credit": credit_amount
                            }
                            
                            transactions.append(transaction)
    
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return []
    
    # Remove duplicates based on date and amounts
    seen = set()
    unique_transactions = []
    
    for transaction in transactions:
        # Create a key for duplicate detection
        key = (
            transaction["date"],
            transaction.get("debit"),
            transaction.get("credit"),
            transaction.get("description", "")[:50]  # First 50 chars of description
        )
        
        if key not in seen:
            seen.add(key)
            unique_transactions.append(transaction)
    
    # Sort by date
    unique_transactions.sort(key=lambda x: x["date"])
    
    return unique_transactions

def save_json_output(transactions, output_path):
    """
    Save transactions to JSON file with proper formatting
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transactions, f, indent=2, ensure_ascii=False)
        print(f"JSON data saved to {output_path}")
    except Exception as e:
        print(f"Error saving JSON: {str(e)}")

def print_summary(transactions):
    """
    Print summary of extracted transactions
    """
    if not transactions:
        print("No transactions found.")
        return
    
    print(f"\nExtracted {len(transactions)} transactions:")
    print("=" * 60)
    
    total_credit = sum(t.get("credit", 0) or 0 for t in transactions)
    total_debit = sum(t.get("debit", 0) or 0 for t in transactions)
    
    print(f"Total Credits: {total_credit:.2f}")
    print(f"Total Debits: {total_debit:.2f}")
    print(f"Net Amount: {total_credit - total_debit:.2f}")
    
    # Show first few transactions
    print("\nSample transactions:")
    for i, transaction in enumerate(transactions[:5]):
        print(f"\n{i+1}. Date: {transaction['date']}")
        print(f"   Credit: {transaction.get('credit', 'null')}")
        print(f"   Debit: {transaction.get('debit', 'null')}")
        print(f"   Description: {transaction.get('description', 'null')}")

# Enhanced extraction with specific SNB patterns
def extract_snb_format(pdf_path):
    """
    Enhanced extraction specifically for SNB bank statement format
    """
    transactions = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                
                lines = text.split('\n')
                
                for i, line in enumerate(lines):
                    # Skip header lines
                    if any(header in line for header in ['Date', 'Details', 'Credit', 'Debit', 'Balance']):
                        continue
                    
                    # Look for date at start of line
                    date_match = re.match(r'^(\d{2}/\d{2}/\d{4})', line)
                    
                    if date_match:
                        date_raw = date_match.group(1)
                        day, month, year = date_raw.split('/')
                        date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        
                        # Get extended context (current + next 2 lines)
                        context_lines = [line]
                        for j in range(1, 3):
                            if i + j < len(lines):
                                next_line = lines[i + j].strip()
                                if next_line and not re.match(r'^\d{2}/\d{2}/\d{4}', next_line):
                                    context_lines.append(next_line)
                                else:
                                    break
                        
                        combined_text = ' '.join(context_lines)
                        
                        # Extract all monetary amounts
                        amounts = extract_monetary_amount(combined_text)
                        
                        # Determine transaction type based on SNB patterns
                        is_credit = False
                        is_debit = False
                        
                        # SNB specific patterns
                        if 'EF' in combined_text:  # Electronic Funds
                            is_credit = True
                        elif any(keyword in combined_text for keyword in ['Outgoing', 'Charge', 'Fee', 'Transfer Fees']):
                            is_debit = True
                        elif any(keyword in combined_text for keyword in ['Incoming', 'SAR', 'Deposit']):
                            is_credit = True
                        
                        # Assign amounts
                        credit_amount = None
                        debit_amount = None
                        
                        if amounts:
                            if is_credit:
                                credit_amount = amounts[0]
                            elif is_debit:
                                debit_amount = amounts[0]
                            else:
                                # Default logic based on amount patterns in SNB format
                                if len(amounts) == 1:
                                    # Single amount - check context
                                    if any(word in combined_text.lower() for word in ['incoming', 'deposit', 'ef']):
                                        credit_amount = amounts[0]
                                    else:
                                        debit_amount = amounts[0]
                        
                        # Clean description
                        description = clean_text(combined_text)
                        
                        transaction = {
                            "date": date_str,
                            "description": description,
                            "debit": debit_amount,
                            "credit": credit_amount
                        }
                        
                        transactions.append(transaction)
    
    except Exception as e:
        print(f"Error in SNB extraction: {str(e)}")
    
    return transactions

# Main execution
if __name__ == "__main__":
    # Configuration
    pdf_file_path = "pdf/11374826000107 SNB_compressed (2).pdf"  # Replace with your PDF path
    json_output_path = "extracted_transactions.json"
    
    print("Starting SNB bank statement extraction...")
    print("=" * 50)
    
    # Try SNB-specific extraction first
    transactions = extract_snb_format(pdf_file_path)
    
    # Fallback to general extraction if SNB-specific fails
    if not transactions:
        print("SNB extraction returned no results, trying general extraction...")
        transactions = extract_bank_statement_json(pdf_file_path)
    
    if transactions:
        # Print summary
        print_summary(transactions)
        
        # Save to JSON
        save_json_output(transactions, json_output_path)
        
        # Print sample JSON format
        print(f"\nSample JSON output:")
        print(json.dumps(transactions[:2], indent=2, ensure_ascii=False))
        
    else:
        print("No transactions could be extracted from the PDF.")
        print("Please verify:")
        print("1. PDF file path is correct")
        print("2. PDF contains readable text (not scanned image)")
        print("3. PDF format matches expected SNB statement structure")

# Utility function to validate JSON output format
def validate_json_format(transactions):
    """
    Validate that extracted data matches required JSON format
    """
    required_fields = ["date", "description", "debit", "credit"]
    
    for i, transaction in enumerate(transactions):
        for field in required_fields:
            if field not in transaction:
                print(f"Warning: Transaction {i+1} missing field '{field}'")
        
        # Validate date format (YYYY-MM-DD)
        if "date" in transaction and transaction["date"]:
            try:
                datetime.strptime(transaction["date"], "%Y-%m-%d")
            except ValueError:
                print(f"Warning: Transaction {i+1} has invalid date format: {transaction['date']}")
        
        # Validate numeric fields
        for field in ["debit", "credit"]:
            if field in transaction and transaction[field] is not None:
                if not isinstance(transaction[field], (int, float)):
                    print(f"Warning: Transaction {i+1} field '{field}' is not numeric: {transaction[field]}")
    
    print(f"Validation complete for {len(transactions)} transactions.")