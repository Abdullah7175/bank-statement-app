import os
import pandas as pd
import base64
from huggingface_hub import InferenceClient
import fitz  # PyMuPDF
from PIL import Image
import io
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from dateutil import parser
from dotenv import load_dotenv

# Load environment variables from parent directory .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TransactionExtractor:
    def __init__(self):
        """Initialize the extractor with Hugging Face client for transaction extraction only"""
        self.client = InferenceClient(
            provider="fireworks-ai",
            api_key=os.environ["HF_TOKEN"],
        )
        
    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[Image.Image]:
        """Convert PDF pages to images for AI processing"""
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Convert page to image
            mat = fitz.Matrix(dpi/72, dpi/72)  # Scale factor
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
            
        doc.close()
        return images
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def extract_transactions_from_image(self, image: Image.Image, page_num: int) -> Dict[str, Any]:
        """Extract ONLY transaction data from image - no account info"""
        image_url = self.image_to_base64(image)
        
        prompt = """
        Extract ONLY transaction data from this KSA bank statement image. Do NOT extract account information.
        
        CRITICAL TRANSACTION RULES:
        1. Extract EVERY transaction row from transaction tables
        2. Each transaction must have: original-date + description + (debit OR credit) + balance
        3. NEVER have both debit AND credit in same transaction (only one can have value)
        4. Either debit OR credit must be present (both cannot be missing/null)
        5. Balance can be 0.00 but NEVER missing/null
        6. Copy values EXACTLY as shown - no modifications or rounding
        7. Handle English, Arabic, or mixed English/Arabic text
        8. Use "UNCLEAR" only if you cannot read a value clearly
        9. Do NOT extract opening/closing balance summary rows
        10. Do NOT extract from statement summary sections
        11. Do NOT hallucinate or invent any values
        
        Extract Format (JSON object only):
        {
            "transactions": [
                {
                    "original_date": "exact date as shown in original format",
                    "description": "transaction description exactly as written",
                    "debit": "amount or empty string if not present",
                    "credit": "amount or empty string if not present", 
                    "balance": "balance amount (never empty)"
                }
            ]
        }
        
        VERIFICATION CHECKLIST:
        - Count visible transaction rows in image
        - Ensure JSON has same number of transactions
        - Verify no transaction has both debit AND credit
        - Verify every transaction has either debit OR credit
        - Verify no balance is missing
        - Verify no values are invented
        
        Return ONLY the JSON object with transactions array."""
        
        try:
            completion = self.client.chat.completions.create(
                model=os.environ.get("MODEL_ID", "meta-llama/Llama-4-Scout-17B-16E-Instruct"),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0.0
            )
            
            response_text = completion.choices[0].message.content
            logger.info(f"Page {page_num} transaction response (first 200 chars): {response_text[:200]}...")
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # Fallback: try to parse the entire response as JSON
                return json.loads(response_text)
                
        except Exception as e:
            logger.error(f"Error extracting transactions from page {page_num}: {str(e)}")
            return {"transactions": []}
    
    def validate_transaction(self, transaction: Dict, page_num: int) -> bool:
        """Validate transaction based on KSA requirements"""
        try:
            # Check original_date
            date_str = str(transaction.get('original_date', '')).strip()
            if not date_str or date_str.lower() in ['date', 'تاريخ', 'original_date']:
                logger.warning(f"Page {page_num}: Transaction rejected - invalid date: {date_str}")
                return False

            # Check balance - can be zero but not missing/null
            balance = transaction.get('balance')
            if balance is None or str(balance).strip() == '' or str(balance).lower() in ['balance', 'رصيد']:
                logger.warning(f"Page {page_num}: Transaction rejected - missing balance")
                return False

            # Check debit/credit rules
            debit = str(transaction.get('debit', '')).strip()
            credit = str(transaction.get('credit', '')).strip()
            
            # Clean empty values
            has_debit = debit and debit not in ['-', '', 'null', 'none', '0', '0.00']
            has_credit = credit and credit not in ['-', '', 'null', 'none', '0', '0.00']

            # Rule: Either credit OR debit must be present (both cannot be missing)
            if not has_debit and not has_credit:
                logger.warning(f"Page {page_num}: Transaction rejected - both debit and credit missing")
                return False
            
            # Rule: Both credit AND debit cannot be present at the same time
            if has_debit and has_credit:
                logger.warning(f"Page {page_num}: Transaction rejected - both debit and credit present")
                return False

            # Check description exists
            description = str(transaction.get('description', '')).strip()
            if not description:
                logger.warning(f"Page {page_num}: Transaction rejected - missing description")
                return False

            return True

        except Exception as e:
            logger.error(f"Page {page_num}: Error validating transaction: {e}")
            return False
    
    def clean_and_validate_data(self, extracted_data: Dict[str, Any], page_num: int) -> List[Dict]:
        """Clean and validate extracted transaction data"""
        transactions = extracted_data.get("transactions", [])
        
        cleaned_transactions = []
        
        for i, transaction in enumerate(transactions):
            try:
                # Validate transaction first
                if not self.validate_transaction(transaction, page_num):
                    logger.warning(f"Page {page_num}: Skipping invalid transaction {i+1}")
                    continue
                
                # Clean amounts (remove commas, currency symbols, etc.)
                def clean_amount(amount_str):
                    if amount_str is None or amount_str == "" or amount_str in ['-', 'null', 'none']:
                        return ""
                    if isinstance(amount_str, (int, float)):
                        return str(float(amount_str)) if amount_str != 0 else ""
                    
                    # Remove common currency symbols and separators
                    cleaned = str(amount_str).replace(",", "").replace("$", "").replace("€", "")
                    cleaned = cleaned.replace("£", "").replace("₹", "").replace("﷼", "").replace("(SAR)", "").replace("SAR", "")
                    cleaned = cleaned.strip()
                    
                    try:
                        if cleaned and cleaned not in ['0', '0.00']:
                            float(cleaned)  # Validate it's a number
                            return cleaned
                        else:
                            return ""
                    except:
                        return cleaned if cleaned else ""
                
                def clean_balance(balance_str):
                    if balance_str is None or balance_str == "":
                        return "0.00"
                    if isinstance(balance_str, (int, float)):
                        return str(float(balance_str))
                    
                    # Remove common currency symbols and separators
                    cleaned = str(balance_str).replace(",", "").replace("$", "").replace("€", "")
                    cleaned = cleaned.replace("£", "").replace("₹", "").replace("﷼", "").replace("(SAR)", "").replace("SAR", "")
                    cleaned = cleaned.strip()
                    
                    try:
                        return str(float(cleaned)) if cleaned else "0.00"
                    except:
                        return cleaned if cleaned else "0.00"
                
                debit = clean_amount(transaction.get("debit", ""))
                credit = clean_amount(transaction.get("credit", ""))
                balance = clean_balance(transaction.get("balance", ""))
                
                cleaned_transaction = {
                    "original_date": transaction.get("original_date", "").strip(),
                    "description": transaction.get("description", "").strip(),
                    "debit": debit,
                    "credit": credit,
                    "balance": balance
                }
                
                cleaned_transactions.append(cleaned_transaction)
                
            except Exception as e:
                logger.error(f"Page {page_num}: Error cleaning transaction {i+1}: {str(e)}")
                continue
        
        logger.info(f"Page {page_num}: Extracted {len(cleaned_transactions)} valid transactions")
        return cleaned_transactions
    
    def parse_date_for_sorting(self, date_str: str) -> tuple:
        """Parse date string and return sortable tuple (year, month, day, hour, minute)"""
        try:
            date_str = str(date_str).strip()
            
            # Handle common date formats
            # Format: DD/MM/YYYY HH:MM or DD/MM/YY HH:MM (include time if present)
            if re.match(r'\d{1,2}/\d{1,2}/\d{2,4}(\s+\d{1,2}:\d{2})?', date_str):
                parts = date_str.split(' ')
                date_part = parts[0]
                time_part = parts[1] if len(parts) > 1 else "00:00"
                
                # Parse date
                date_components = date_part.split('/')
                day = int(date_components[0])
                month = int(date_components[1])
                year = int(date_components[2])
                
                # Handle 2-digit years
                if year < 100:
                    if year > 50:  # Assume 1900s
                        year += 1900
                    else:  # Assume 2000s
                        year += 2000
                
                # Parse time
                time_components = time_part.split(':')
                hour = int(time_components[0]) if len(time_components) > 0 else 0
                minute = int(time_components[1]) if len(time_components) > 1 else 0
                
                return (year, month, day, hour, minute)
            
            # Format: YYYY-MM-DD HH:MM
            elif re.match(r'\d{4}-\d{1,2}-\d{1,2}(\s+\d{1,2}:\d{2})?', date_str):
                parts = date_str.split(' ')
                date_part = parts[0]
                time_part = parts[1] if len(parts) > 1 else "00:00"
                
                date_components = date_part.split('-')
                year = int(date_components[0])
                month = int(date_components[1])
                day = int(date_components[2])
                
                time_components = time_part.split(':')
                hour = int(time_components[0]) if len(time_components) > 0 else 0
                minute = int(time_components[1]) if len(time_components) > 1 else 0
                
                return (year, month, day, hour, minute)
            
            # Format: DD-MM-YYYY HH:MM (include time if present)
            elif re.match(r'\d{1,2}-\d{1,2}-\d{4}(\s+\d{1,2}:\d{2})?', date_str):
                parts = date_str.split(' ')
                date_part = parts[0]
                time_part = parts[1] if len(parts) > 1 else "00:00"
                
                date_components = date_part.split('-')
                day = int(date_components[0])
                month = int(date_components[1])
                year = int(date_components[2])
                
                time_components = time_part.split(':')
                hour = int(time_components[0]) if len(time_components) > 0 else 0
                minute = int(time_components[1]) if len(time_components) > 1 else 0
                
                return (year, month, day, hour, minute)
            
            # Try to use dateutil parser as fallback
            else:
                try:
                    parsed_date = parser.parse(date_str, dayfirst=True)
                    return (parsed_date.year, parsed_date.month, parsed_date.day, parsed_date.hour, parsed_date.minute)
                except:
                    # If all parsing fails, return the original string for basic string sorting
                    logger.warning(f"Could not parse date: {date_str}, using string sorting")
                    return (9999, 12, 31, 23, 59)  # Put unparseable dates at the end
                    
        except Exception as e:
            logger.warning(f"Error parsing date '{date_str}': {e}")
            return (9999, 12, 31, 23, 59)  # Put problematic dates at the end
    
    def sort_transactions_by_date(self, transactions: List[Dict]) -> List[Dict]:
        """Sort transactions by date in chronological order"""
        try:
            logger.info("Sorting transactions by date...")
            
            # Add sorting key to each transaction
            for transaction in transactions:
                transaction['_sort_key'] = self.parse_date_for_sorting(transaction.get('original_date', ''))
            
            # Sort by the date tuple
            sorted_transactions = sorted(transactions, key=lambda x: x['_sort_key'])
            
            # Remove the temporary sorting key
            for transaction in sorted_transactions:
                del transaction['_sort_key']
            
            logger.info("Successfully sorted transactions by date")
            return sorted_transactions
            
        except Exception as e:
            logger.error(f"Error sorting transactions by date: {e}")
            logger.info("Returning transactions in original order")
            return transactions
    
    def extract_from_pdf(self, pdf_path: str, output_json: Optional[str] = None) -> Dict[str, Any]:
        """Main function to extract transaction data from PDF"""
        logger.info(f"Processing PDF for transactions: {pdf_path}")
        
        # Convert PDF to images
        logger.info("Converting PDF to images...")
        images = self.pdf_to_images(pdf_path)
        
        all_transactions = []
        
        # Process each page
        for i, image in enumerate(images):
            logger.info(f"Processing page {i+1}/{len(images)} for transactions...")
            
            extracted_data = self.extract_transactions_from_image(image, i+1)
            
            # Clean and validate transactions
            if extracted_data.get("transactions"):
                cleaned_transactions = self.clean_and_validate_data(extracted_data, i+1)
                all_transactions.extend(cleaned_transactions)
        
        # Sort transactions by date first
        if all_transactions:
            all_transactions = self.sort_transactions_by_date(all_transactions)
        
        # Remove duplicates based on all fields (maintaining sorted order)
        unique_transactions = []
        seen = set()
        for transaction in all_transactions:
            transaction_key = (
                transaction['original_date'],
                transaction['description'],
                transaction['debit'],
                transaction['credit'],
                transaction['balance']
            )
            if transaction_key not in seen:
                seen.add(transaction_key)
                unique_transactions.append(transaction)
        
        result = {
            "transaction_detail_depth": unique_transactions,
            "total_transactions": len(unique_transactions)
        }
        
        logger.info(f"Successfully extracted {len(unique_transactions)} unique transactions")
        
        # Save to JSON if requested
        if output_json:
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Transaction data saved to: {output_json}")
        
        # Display summary
        if unique_transactions:
            logger.info(f"Transaction Summary:")
            logger.info(f"- Total Transactions: {len(unique_transactions)}")
            
            debit_count = sum(1 for t in unique_transactions if t['debit'])
            credit_count = sum(1 for t in unique_transactions if t['credit'])
            logger.info(f"- Debit Transactions: {debit_count}")
            logger.info(f"- Credit Transactions: {credit_count}")
            
            # Show date range
            first_date = unique_transactions[0]['original_date']
            last_date = unique_transactions[-1]['original_date']
            logger.info(f"- Date Range: {first_date} to {last_date}")
            
            logger.info(f"Sample transactions (sorted by date):")
            for i, trans in enumerate(unique_transactions[:3]):
                logger.info(f"  {i+1}: {trans}")
        
        return result

# Usage example
def main():
    # Initialize extractor
    extractor = TransactionExtractor()
    
    # Extract transaction data from PDF
    pdf_path = "Bank statments/Toyland company , SNB(90).pdf"  # Replace with your PDF path
    output_json = "transactions_output.json"
    
    try:
        result = extractor.extract_from_pdf(pdf_path, output_json)
        return result
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return {}

if __name__ == "__main__":
    # Make sure HF_TOKEN and MODEL_ID environment variables are set
    if not os.environ.get("HF_TOKEN"):
        print("Error: HF_TOKEN environment variable is required")
        print("Please check your .env file")
        exit(1)
    
    if not os.environ.get("MODEL_ID"):
        print("Error: MODEL_ID environment variable is required")
        print("Please check your .env file")
        exit(1)
    
    result = main()
