import os
import base64
from huggingface_hub import InferenceClient
import fitz  # PyMuPDF
from PIL import Image
import io
import json
import re
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv

# Load environment variables from parent directory .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AccountInfoExtractor:
    def __init__(self):
        """Initialize the extractor with Hugging Face client for account info extraction only"""
        self.client = InferenceClient(
            provider="fireworks-ai",
            api_key=os.environ["HF_TOKEN"],
        )
        
    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> tuple:
        """Convert PDF first and last pages to images for account info processing"""
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        first_page_img = None
        last_page_img = None
        
        if total_pages > 0:
            # Get first page
            page = doc[0]
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            first_page_img = Image.open(io.BytesIO(img_data))
            
            # Get last page (if different from first)
            if total_pages > 1:
                page = doc[total_pages - 1]
                mat = fitz.Matrix(dpi/72, dpi/72)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                last_page_img = Image.open(io.BytesIO(img_data))
            
        doc.close()
        return first_page_img, last_page_img, total_pages
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def extract_account_info_from_image(self, image: Image.Image, page_location: str) -> Dict[str, Any]:
        """Extract ONLY account information from image - no transactions"""
        image_url = self.image_to_base64(image)
        
        prompt = f"""
        Extract ONLY account information from this KSA bank statement image ({page_location}). Do NOT extract transaction data.
        
        Handle English, Arabic, or mixed English/Arabic text.
        
        EXTRACT THESE FIELDS ONLY:
        1. Account Name/Holder Name (in English and/or Arabic)
        2. Account Number 
        3. IBAN Number
        4. Statement Date/Period (date range like "01/08/18 - 31/08/18" or "2024-01-01 - 2024-12-31")
        5. Opening Balance (if present in dedicated summary sections)
        6. Closing Balance (if present in dedicated summary sections)
        
        CRITICAL RULES FOR OPENING/CLOSING BALANCES:
        - ONLY extract from DEDICATED SUMMARY SECTIONS that are COMPLETELY SEPARATE from transaction tables
        - NEVER extract from transaction tables, transaction rows, or running balance columns
        - Look for EXPLICIT LABELS: "Opening Balance"/"الرصيد الإفتتاحي" and "Closing Balance"/"رصيد الإقفال"/"الرصيد الختامي"
        - Must be in isolated summary boxes or sections at top/bottom of page
        - If balances appear mixed with transaction data, DO NOT extract them
        - If no clear summary section exists with proper labels, set to null
        - Opening/Closing balances are found together on SAME page (first page OR last page)
        - Look for dedicated summary areas like:
          * Statement summary boxes
          * Account summary sections  
          * Balance summary tables (separate from transaction tables)
        - ABSOLUTELY FORBIDDEN: Any balance value that appears in the same area as dates or transaction descriptions
        
        IMPORTANT NOTES:
        - Account info can be missing/null - don't hallucinate values
        - If opening/closing balance not found, set to null (don't use 0)
        - Don't make up any information that's not clearly visible
        
        Return Format (JSON object only):
        {{
            "account_name": "exact name or null",
            "account_number": "exact number or null", 
            "iban_number": "exact IBAN or null",
            "statement_date": "exact date range or null",
            "opening_balance": "balance value or null",
            "closing_balance": "balance value or null"
        }}
        
        Return ONLY the JSON object with no additional text."""
        
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
                max_tokens=1500,
                temperature=0.0
            )
            
            response_text = completion.choices[0].message.content
            logger.info(f"Account info response from {page_location} (first 200 chars): {response_text[:200]}...")
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # Fallback: try to parse the entire response as JSON
                return json.loads(response_text)
                
        except Exception as e:
            logger.error(f"Error extracting account info from {page_location}: {str(e)}")
            return {
                "account_name": None,
                "account_number": None,
                "iban_number": None,
                "statement_date": None,
                "opening_balance": None,
                "closing_balance": None
            }
    
    def clean_account_field(self, value) -> Optional[str]:
        """Clean account field values"""
        if not value or str(value).strip().lower() in ['null', 'none', '', 'unclear', 'not found']:
            return None
        return str(value).strip()
    
    def clean_balance_field(self, value) -> Optional[str]:
        """Clean balance field values with strict validation"""
        if not value or str(value).strip().lower() in ['null', 'none', '', 'unclear', 'not found']:
            return None
        
        value_str = str(value).strip()
        
        # Only reject if it has clear transaction table context (dates, descriptions, etc.)
        # Simple numbers like "6489.46" or "21825.64" from summary sections should be accepted
        suspicious_patterns = [
            r'\d{2}/\d{2}/\d{4}.*\d+\.\d{2}',  # Contains dates WITH balance values
            r'(debit|credit|مدين|دائن).*\d+\.\d{2}',  # Contains debit/credit terms WITH balance
            r'(transaction|معاملة|عملية).*\d+\.\d{2}',  # Contains transaction terms WITH balance
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, value_str.lower()):
                logger.warning(f"Rejected balance value - appears to be from transaction table context: {value_str}")
                return None
        
        # Remove currency symbols but keep the numeric value
        cleaned = re.sub(r'[^\d.,\-+\s]', '', value_str)
        cleaned = cleaned.strip()
        
        # If nothing left after cleaning, return None
        if not cleaned or cleaned in ['-', '+', '.', ',']:
            return None
            
        return cleaned
    
    def merge_account_info(self, first_page_info: Dict, last_page_info: Dict) -> Dict[str, Any]:
        """Merge account info from first and last pages, handling balance location logic"""
        
        # Initialize result
        result = {
            "account_name": None,
            "account_number": None,
            "iban_number": None,
            "statement_date": None,
            "opening_balance": None,
            "closing_balance": None
        }
        
        # Merge basic account info (prefer first page, fallback to last page)
        for field in ["account_name", "account_number", "iban_number", "statement_date"]:
            if first_page_info and first_page_info.get(field):
                result[field] = self.clean_account_field(first_page_info[field])
            elif last_page_info and last_page_info.get(field):
                result[field] = self.clean_account_field(last_page_info[field])
        
        # Handle opening/closing balance logic
        # Check if first page has balances
        first_has_opening = first_page_info and first_page_info.get("opening_balance")
        first_has_closing = first_page_info and first_page_info.get("closing_balance")
        first_has_balances = first_has_opening or first_has_closing
        
        # Check if last page has balances
        last_has_opening = last_page_info and last_page_info.get("opening_balance")
        last_has_closing = last_page_info and last_page_info.get("closing_balance")
        last_has_balances = last_has_opening or last_has_closing
        
        if first_has_balances and not last_has_balances:
            # Balances are on first page
            logger.info("Using balances from first page")
            result["opening_balance"] = self.clean_balance_field(first_page_info.get("opening_balance"))
            result["closing_balance"] = self.clean_balance_field(first_page_info.get("closing_balance"))
            
        elif last_has_balances and not first_has_balances:
            # Balances are on last page
            logger.info("Using balances from last page")
            result["opening_balance"] = self.clean_balance_field(last_page_info.get("opening_balance"))
            result["closing_balance"] = self.clean_balance_field(last_page_info.get("closing_balance"))
            
        elif first_has_balances and last_has_balances:
            # Both pages have balances - this shouldn't happen according to user requirements
            # Use first page and log warning
            logger.warning("Both first and last page have balances - using first page (this may indicate an issue)")
            result["opening_balance"] = self.clean_balance_field(first_page_info.get("opening_balance"))
            result["closing_balance"] = self.clean_balance_field(first_page_info.get("closing_balance"))
            
        else:
            # No balances found on either page
            logger.info("No opening/closing balances found in dedicated summary sections")
            result["opening_balance"] = None
            result["closing_balance"] = None
        
        return result
    
    def extract_from_pdf(self, pdf_path: str, output_json: Optional[str] = None) -> Dict[str, Any]:
        """Main function to extract account information from PDF"""
        logger.info(f"Processing PDF for account info: {pdf_path}")
        
        # Convert PDF to images (first and last page only)
        logger.info("Converting first and last pages to images...")
        first_page_img, last_page_img, total_pages = self.pdf_to_images(pdf_path)
        
        if not first_page_img:
            logger.error("Could not extract first page from PDF")
            return {
                "account_name": None,
                "account_number": None,
                "iban_number": None,
                "statement_date": None,
                "opening_balance": None,
                "closing_balance": None
            }
        
        # Extract from first page
        logger.info("Extracting account info from first page...")
        first_page_info = self.extract_account_info_from_image(first_page_img, "first page")
        
        # Extract from last page (if different from first)
        last_page_info = None
        if last_page_img and total_pages > 1:
            logger.info("Extracting account info from last page...")
            last_page_info = self.extract_account_info_from_image(last_page_img, "last page")
        
        # Merge account information
        result = self.merge_account_info(first_page_info, last_page_info)
        
        logger.info(f"Final account info: {result}")
        
        # Save to JSON if requested
        if output_json:
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Account info saved to: {output_json}")
        
        # Display summary
        logger.info("Account Info Summary:")
        logger.info(f"- Account Name: {result.get('account_name', 'Not found')}")
        logger.info(f"- Account Number: {result.get('account_number', 'Not found')}")
        logger.info(f"- IBAN: {result.get('iban_number', 'Not found')}")
        logger.info(f"- Statement Date: {result.get('statement_date', 'Not found')}")
        logger.info(f"- Opening Balance: {result.get('opening_balance', 'Not found')}")
        logger.info(f"- Closing Balance: {result.get('closing_balance', 'Not found')}")
        
        return result

# Usage example
def main():
    # Initialize extractor
    extractor = AccountInfoExtractor()
    
    # Extract account info from PDF
    pdf_path = "Bank statments/BAJ AUG 18.pdf"  # Replace with your PDF path
    output_json = "BAJ AUG 18_account.json"
    
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
