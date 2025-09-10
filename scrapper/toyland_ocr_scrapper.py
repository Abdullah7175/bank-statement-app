#!/usr/bin/env python3
"""
Custom OCR Scraper for Toyland Company SNB PDF
Handles image-based bank statements using advanced OCR techniques
"""

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import cv2
import numpy as np
import re
import json
import time
from pathlib import Path
import io

def extract_text_from_image_advanced(image_data):
    """Advanced OCR extraction with multiple preprocessing techniques"""
    try:
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Multiple preprocessing techniques for better OCR
        results = []
        
        # Technique 1: Original image
        try:
            text1 = pytesseract.image_to_string(pil_image, lang='eng+ara')
            if text1.strip():
                results.append(("Original", text1))
        except:
            pass
        
        # Technique 2: Grayscale with thresholding
        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            pil_binary = Image.fromarray(binary)
            text2 = pytesseract.image_to_string(pil_binary, lang='eng+ara')
            if text2.strip():
                results.append(("Binary", text2))
        except:
            pass
        
        # Technique 3: Adaptive thresholding
        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            pil_adaptive = Image.fromarray(adaptive)
            text3 = pytesseract.image_to_string(pil_adaptive, lang='eng+ara')
            if text3.strip():
                results.append(("Adaptive", text3))
        except:
            pass
        
        # Technique 4: Noise reduction + thresholding
        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray)
            _, binary_clean = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            pil_clean = Image.fromarray(binary_clean)
            text4 = pytesseract.image_to_string(pil_clean, lang='eng+ara')
            if text4.strip():
                results.append(("Denoised", text4))
        except:
            pass
        
        # Technique 5: Edge enhancement
        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(gray, -1, kernel)
            _, binary_enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            pil_enhanced = Image.fromarray(binary_enhanced)
            text5 = pytesseract.image_to_string(pil_enhanced, lang='eng+ara')
            if text5.strip():
                results.append(("Enhanced", text5))
        except:
            pass
        
        # Combine all results and find the best one
        if not results:
            return ""
        
        # Choose the result with the most meaningful content
        best_result = ""
        max_score = 0
        
        for technique, text in results:
            # Score based on content quality
            score = 0
            text_lower = text.lower()
            
            # Keywords that indicate bank statement content
            bank_keywords = ['bank', 'account', 'balance', 'transaction', 'debit', 'credit', 'date', 'amount', 'sar', 'riyal']
            for keyword in bank_keywords:
                if keyword in text_lower:
                    score += 1
            
            # Numbers and currency patterns
            if re.search(r'\d+\.\d{2}', text):  # Decimal amounts
                score += 2
            if re.search(r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?', text):  # Formatted amounts
                score += 3
            
            # Date patterns
            if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text):
                score += 2
            
            if score > max_score:
                max_score = score
                best_result = text
        
        return best_result if best_result else results[0][1]
        
    except Exception as e:
        print(f"   OCR processing error: {e}")
        return ""

def parse_toyland_statement(ocr_text):
    """Parse OCR text to extract bank statement information"""
    if not ocr_text:
        return {}, []
    
    print(f"   📝 Parsing OCR text ({len(ocr_text)} characters)...")
    
    # Initialize result structures
    account_summary = {}
    transactions = []
    
    # Split text into lines for processing
    lines = ocr_text.split('\n')
    
    # Extract account information
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Look for account number patterns
        account_match = re.search(r'(\d{10,})', line)
        if account_match and not account_summary.get('account_number'):
            account_summary['account_number'] = account_match.group(1)
        
        # Look for company name
        if 'toyland' in line.lower() or 'toy' in line.lower():
            account_summary['customer_name'] = line.strip()
        
        # Look for currency
        if 'sar' in line.lower() or 'riyal' in line.lower():
            account_summary['currency'] = 'SAR'
        
        # Look for period information
        period_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', line)
        if period_match and not account_summary.get('period'):
            account_summary['period'] = period_match.group(1)
    
    # Extract transaction information
    # Look for transaction patterns in the text
    transaction_patterns = [
        # Pattern: Date Amount Balance Description
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+([+-]?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+(.+)',
        # Pattern: Date Description Amount Balance
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+(.+?)\s+([+-]?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        # Pattern: Amount Balance (simpler)
        r'([+-]?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
    ]
    
    for pattern in transaction_patterns:
        matches = re.findall(pattern, ocr_text)
        if matches:
            for match in matches:
                if len(match) >= 3:
                    try:
                        if len(match) == 4:  # Full pattern
                            date, amount, balance, description = match
                            debit = None
                            credit = None
                            
                            # Determine if it's debit or credit
                            if amount.startswith('-') or amount.startswith('−'):
                                debit = float(amount.replace('-', '').replace('−', '').replace(',', ''))
                            else:
                                credit = float(amount.replace(',', ''))
                            
                            transactions.append({
                                'date': date,
                                'description': description.strip(),
                                'debit': debit,
                                'credit': credit,
                                'balance': float(balance.replace(',', ''))
                            })
                        elif len(match) == 3:  # Amount Balance pattern
                            amount, balance = match
                            debit = None
                            credit = None
                            
                            if amount.startswith('-') or amount.startswith('−'):
                                debit = float(amount.replace('-', '').replace('−', '').replace(',', ''))
                            else:
                                credit = float(amount.replace(',', ''))
                            
                            transactions.append({
                                'date': 'Unknown',
                                'description': 'Transaction',
                                'debit': debit,
                                'credit': credit,
                                'balance': float(balance.replace(',', ''))
                            })
                    except ValueError:
                        continue
    
    # If no structured transactions found, try to extract any numbers that look like amounts
    if not transactions:
        print("   🔍 No structured transactions found, extracting amount patterns...")
        amount_pattern = r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        amounts = re.findall(amount_pattern, ocr_text)
        
        if amounts:
            # Convert to float and sort
            numeric_amounts = []
            for amount in amounts:
                try:
                    numeric_amounts.append(float(amount.replace(',', '')))
                except:
                    continue
            
            if numeric_amounts:
                numeric_amounts.sort()
                
                # Use first and last amounts as potential balances
                if len(numeric_amounts) >= 2:
                    account_summary['opening_balance'] = f"{numeric_amounts[0]:,.2f}"
                    account_summary['closing_balance'] = f"{numeric_amounts[-1]:,.2f}"
                elif len(numeric_amounts) == 1:
                    account_summary['opening_balance'] = f"{numeric_amounts[0]:,.2f}"
                    account_summary['closing_balance'] = f"{numeric_amounts[0]:,.2f}"
    
    return account_summary, transactions

def extract_toyland_data(pdf_path):
    """Main function to extract data from Toyland PDF using OCR"""
    try:
        print(f"🔍 Processing Toyland PDF with custom OCR scraper...")
        
        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"   📄 PDF has {total_pages} pages")
        
        all_ocr_text = ""
        
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            print(f"   📖 Processing page {page_num + 1}...")
            
            # Get images from the page
            image_list = page.get_images()
            print(f"   🖼️  Found {len(image_list)} images on page {page_num + 1}")
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        print(f"   🔍 Processing image {img_index + 1} on page {page_num + 1}...")
                        
                        # Convert to PNG format
                        img_data = pix.tobytes("png")
                        
                        # Extract text using advanced OCR
                        ocr_text = extract_text_from_image_advanced(img_data)
                        
                        if ocr_text.strip():
                            print(f"   ✅ OCR extracted {len(ocr_text)} characters from image {img_index + 1}")
                            print(f"   📝 OCR text: \"{ocr_text[:100]}\"")
                            all_ocr_text += ocr_text + "\n"
                        else:
                            print(f"   ⚠️  No text extracted from image {img_index + 1}")
                        
                        pix = None  # Free memory
                    
                except Exception as e:
                    print(f"   ❌ Error processing image {img_index + 1} on page {page_num + 1}: {e}")
                    continue
            
            page = None  # Free memory
        
        doc.close()
        
        if not all_ocr_text.strip():
            print("   ❌ No text extracted from any images")
            return None
        
        print(f"   📝 Total OCR text extracted: {len(all_ocr_text)} characters")
        print(f"   📝 Full OCR text: \"{all_ocr_text}\"")
        
        # Parse the OCR text
        account_summary, transactions = parse_toyland_statement(all_ocr_text)
        
        # Create result structure
        result = {
            "pdf_file": str(Path(pdf_path).resolve()),
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_pages": total_pages,
            "account_summary": account_summary,
            "transactions": transactions,
            "total_transactions": len(transactions),
            "customer_name": account_summary.get("customer_name"),
            "account_number": account_summary.get("account_number"),
            "iban_number": account_summary.get("iban_number"),
            "financial_period": account_summary.get("period"),
            "opening_balance": account_summary.get("opening_balance"),
            "closing_balance": account_summary.get("closing_balance"),
            "pages_processed": total_pages,
            "ocr_used": True,
            "ocr_text_length": len(all_ocr_text)
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Toyland OCR scraper failed: {e}")
        return None

def main():
    """Test the Toyland OCR scraper"""
    pdf_path = "pdf/Toyland company , SNB.pdf"
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    print("🚀 TOYLAND COMPANY SNB - CUSTOM OCR SCRAPER")
    print("=" * 60)
    
    start_time = time.time()
    result = extract_toyland_data(pdf_path)
    processing_time = time.time() - start_time
    
    if result:
        print(f"\n✅ SUCCESS! Data extracted in {processing_time:.2f} seconds")
        print(f"📊 Account Summary: {result['account_summary']}")
        print(f"💳 Transactions found: {result['total_transactions']}")
        print(f"💰 Opening Balance: {result.get('opening_balance', 'N/A')}")
        print(f"💰 Closing Balance: {result.get('closing_balance', 'N/A')}")
        
        # Save result
        output_filename = f"output/toyland_ocr_output.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"📂 Saved result to {output_filename}")
        
    else:
        print(f"\n❌ FAILED to extract data from Toyland PDF")

if __name__ == "__main__":
    main()
