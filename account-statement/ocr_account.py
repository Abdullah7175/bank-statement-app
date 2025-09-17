import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the account_extract_model folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'account_extract_model'))

from extract_transactions import TransactionExtractor
from extract_account_info import AccountInfoExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CombinedStatementExtractor:
    def __init__(self):
        """Initialize the combined extractor"""
        self.transaction_extractor = TransactionExtractor()
        self.account_extractor = AccountInfoExtractor()
    
    def is_balance_valid(self, balance_value) -> bool:
        """Check if balance value is valid (not null, missing, or zero)"""
        if balance_value is None:
            return False
        
        if isinstance(balance_value, str):
            balance_str = balance_value.strip()
            if not balance_str or balance_str.lower() in ['null', 'none', '', '0', '0.00', '0.0']:
                return False
            try:
                float_val = float(balance_str.replace(',', ''))
                return float_val != 0.0
            except (ValueError, TypeError):
                return False
        
        if isinstance(balance_value, (int, float)):
            return balance_value != 0.0
            
        return False
    
    def extract_balance_from_transactions(self, transactions: List[Dict]) -> tuple:
        """Extract opening and closing balance from transaction list"""
        if not transactions:
            logger.warning("No transactions available for balance extraction")
            return None, None
        
        # First transaction balance is opening balance
        opening_balance = transactions[0].get('balance')
        
        # Last transaction balance is closing balance
        closing_balance = transactions[-1].get('balance')
        
        logger.info(f"Extracted from transactions - Opening: {opening_balance}, Closing: {closing_balance}")
        
        return opening_balance, closing_balance
    
    def safe_float(self, value) -> float:
        """Safely convert value to float"""
        if value is None or value == '':
            return 0.0
        try:
            if isinstance(value, str):
                # Remove commas and currency symbols
                cleaned = value.replace(',', '').replace('SAR', '').replace('﷼', '').strip()
                return float(cleaned) if cleaned else 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object"""
        try:
            # Handle various date formats
            date_str = str(date_str).strip()
            
            # Remove time part if present
            date_part = date_str.split(' ')[0]
            
            # Try different date formats
            formats = ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%d/%m/%y']
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_part, fmt)
                except ValueError:
                    continue
                    
            # If all fail, return a default date
            logger.warning(f"Could not parse date: {date_str}")
            return datetime(1900, 1, 1)

        except Exception as e:
            logger.warning(f"Error parsing date '{date_str}': {e}")
            return datetime(1900, 1, 1)
    
    def is_international_transaction(self, description: str) -> bool:
        """Check if transaction is international based on description"""
        international_keywords = [
            'SWIFT', 'WIRE', 'USD', 'EUR', 'GBP', 'AED', 'international',
            'foreign', 'overseas', 'remittance', 'transfer abroad',
            'incoming wire', 'outgoing wire', 'correspondent bank',
            'nostro', 'vostro', 'cross border'
        ]
        
        description_lower = description.lower()
        return any(keyword.lower() in description_lower for keyword in international_keywords)
    
    def calculate_transaction_summary(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive transaction summary"""
        logger.info("Calculating transaction summary...")
        
        total_deposits_count = 0
        total_deposits_amount = 0.0
        total_withdrawals_count = 0
        total_withdrawals_amount = 0.0
        
        balances = []
        
        for transaction in transactions:
            credit = self.safe_float(transaction.get('credit', ''))
            debit = self.safe_float(transaction.get('debit', ''))
            balance = self.safe_float(transaction.get('balance', ''))
            
            # Count deposits (credits)
            if credit > 0:
                total_deposits_count += 1
                total_deposits_amount += credit
            
            # Count withdrawals (debits)
            if debit > 0:
                total_withdrawals_count += 1
                total_withdrawals_amount += debit
            
            # Collect balances for statistics
            if balance > 0:
                balances.append(balance)
        
        # Calculate net change
        net_change = total_deposits_amount - total_withdrawals_amount
        
        # Balance statistics
        min_balance = min(balances) if balances else 0.0
        max_balance = max(balances) if balances else 0.0
        avg_balance = sum(balances) / len(balances) if balances else 0.0
        
        # Balance verification
        expected_closing = (balances[0] if balances else 0.0) + net_change
        actual_closing = balances[-1] if balances else 0.0
        difference = actual_closing - expected_closing
        
        return {
            "total_deposits": {
                "count": total_deposits_count,
                "amount": round(total_deposits_amount, 2)
            },
            "total_withdrawals": {
                "count": total_withdrawals_count,
                "amount": round(total_withdrawals_amount, 2)
            },
            "net_change": {
                "amount": round(net_change, 2)
            },
            "balance_verification": {
                "expected_closing_balance": round(expected_closing, 2),
                "actual_closing_balance": round(actual_closing, 2),
                "difference": round(difference, 2)
            },
            "balance_statistics": {
                "minimum_balance": round(min_balance, 2),
                "maximum_balance": round(max_balance, 2),
                "average_balance": round(avg_balance, 2)
            }
        }
    
    def calculate_monthly_analysis(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Calculate monthly analysis"""
        logger.info("Calculating monthly analysis...")
        
        monthly_data = defaultdict(lambda: {
            'transactions': [],
            'international_inward': [],
            'international_outward': []
        })
        
        # Group transactions by month
        for transaction in transactions:
            date_obj = self.parse_date(transaction.get('original_date', ''))
            month_key = date_obj.strftime('%Y-%m')
            
            monthly_data[month_key]['transactions'].append(transaction)
            
            # Check for international transactions
            description = transaction.get('description', '')
            credit = self.safe_float(transaction.get('credit', ''))
            debit = self.safe_float(transaction.get('debit', ''))
            
            if self.is_international_transaction(description):
                if credit > 0:
                    monthly_data[month_key]['international_inward'].append(transaction)
                elif debit > 0:
                    monthly_data[month_key]['international_outward'].append(transaction)
        
        # Calculate monthly statistics
        monthly_analysis = {}
        
        for month, data in monthly_data.items():
            month_transactions = data['transactions']
            
            if not month_transactions:
                continue
            
            # Calculate balances and amounts
            balances = [self.safe_float(t.get('balance', '')) for t in month_transactions]
            credits = [self.safe_float(t.get('credit', '')) for t in month_transactions if self.safe_float(t.get('credit', '')) > 0]
            debits = [self.safe_float(t.get('debit', '')) for t in month_transactions if self.safe_float(t.get('debit', '')) > 0]
            
            opening_balance = balances[0] if balances else 0.0
            closing_balance = balances[-1] if balances else 0.0
            total_credit = sum(credits)
            total_debit = sum(debits)
            net_change = total_credit - total_debit
            
            # Calculate fluctuation (max - min balance)
            fluctuation = max(balances) - min(balances) if balances else 0.0
            
            # International transaction statistics
            intl_inward = data['international_inward']
            intl_outward = data['international_outward']
            
            intl_inward_total = sum(self.safe_float(t.get('credit', '')) for t in intl_inward)
            intl_outward_total = sum(self.safe_float(t.get('debit', '')) for t in intl_outward)
            
            monthly_analysis[month] = {
                "opening_balance": round(opening_balance, 2),
                "closing_balance": round(closing_balance, 2),
                "total_credit": round(total_credit, 2),
                "total_debit": round(total_debit, 2),
                "net_change": round(net_change, 2),
                "fluctuation": round(fluctuation, 2),
                "minimum_balance": round(min(balances), 2) if balances else 0.0,
                "maximum_balance": round(max(balances), 2) if balances else 0.0,
                "transaction_count": len(month_transactions),
                "international_inward_count": len(intl_inward),
                "international_outward_count": len(intl_outward),
                "international_inward_total": round(intl_inward_total, 2),
                "international_outward_total": round(intl_outward_total, 2)
            }
        
        return monthly_analysis
    
    def calculate_analytics(self, transactions: List[Dict], monthly_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive analytics"""
        logger.info("Calculating analytics...")
        
        if not transactions:
            return self._empty_analytics()
        
        # Calculate fluctuations for average
        fluctuations = [month_data.get('fluctuation', 0.0) for month_data in monthly_analysis.values()]
        avg_fluctuation = sum(fluctuations) / len(fluctuations) if fluctuations else 0.0
        
        # Calculate net cash flow stability (lower is more stable)
        net_changes = [month_data.get('net_change', 0.0) for month_data in monthly_analysis.values()]
        net_cash_flow_stability = sum(abs(nc) for nc in net_changes) / len(net_changes) if net_changes else 0.0
        
        # Count international transactions
        international_transactions = [t for t in transactions if self.is_international_transaction(t.get('description', ''))]
        total_foreign_transactions = len(international_transactions)
        
        total_foreign_amount = sum(
            self.safe_float(t.get('credit', '')) + self.safe_float(t.get('debit', ''))
            for t in international_transactions
        )
        
        # Calculate overdraft frequency and days
        balances = [self.safe_float(t.get('balance', '')) for t in transactions]
        overdraft_instances = [b for b in balances if b < 0]
        overdraft_frequency = len(overdraft_instances)
        overdraft_total_days = overdraft_frequency  # Simplified assumption
        
        # Calculate inflow/outflow statistics
        credits = [self.safe_float(t.get('credit', '')) for t in transactions if self.safe_float(t.get('credit', '')) > 0]
        debits = [self.safe_float(t.get('debit', '')) for t in transactions if self.safe_float(t.get('debit', '')) > 0]
        
        sum_total_inflow = sum(credits)
        sum_total_outflow = sum(debits)
        avg_total_inflow = sum_total_inflow / len(credits) if credits else 0.0
        avg_total_outflow = sum_total_outflow / len(debits) if debits else 0.0
        
        return {
            "average_fluctuation": round(avg_fluctuation, 2),
            "net_cash_flow_stability": round(net_cash_flow_stability, 2),
            "total_foreign_transactions": total_foreign_transactions,
            "total_foreign_amount": round(total_foreign_amount, 2),
            "overdraft_frequency": overdraft_frequency,
            "overdraft_total_days": overdraft_total_days,
            "sum_total_inflow": round(sum_total_inflow, 2),
            "sum_total_outflow": round(sum_total_outflow, 2),
            "avg_total_inflow": round(avg_total_inflow, 2),
            "avg_total_outflow": round(avg_total_outflow, 2)
        }
    
    def _empty_analytics(self) -> Dict[str, Any]:
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
    
    def merge_account_and_transactions(self, account_info: Dict[str, Any], transaction_data: Dict[str, Any], pdf_path: str, pages_processed: int) -> Dict[str, Any]:
        """Merge account info and transaction data with balance fallback logic and comprehensive analytics"""
        
        transactions = transaction_data.get('transaction_detail_depth', [])
        
        # Check if account_info balances are valid
        opening_valid = self.is_balance_valid(account_info.get('opening_balance'))
        closing_valid = self.is_balance_valid(account_info.get('closing_balance'))
        
        logger.info(f"Account info balance validation - Opening valid: {opening_valid}, Closing valid: {closing_valid}")
        
        # If either balance is invalid, extract from transactions
        if not opening_valid or not closing_valid:
            logger.info("Using transaction balances as fallback for missing/invalid account balances")
            
            transaction_opening, transaction_closing = self.extract_balance_from_transactions(transactions)
            
            # Use transaction balances as fallback
            if not opening_valid and transaction_opening is not None:
                account_info['opening_balance'] = transaction_opening
                logger.info(f"Used transaction opening balance: {transaction_opening}")
                
            if not closing_valid and transaction_closing is not None:
                account_info['closing_balance'] = transaction_closing
                logger.info(f"Used transaction closing balance: {transaction_closing}")
        
        # Format transaction_detail_depth with renamed date field
        formatted_transactions = []
        for transaction in transactions:
            formatted_transaction = {
                "date": transaction.get('original_date'),  # Renamed from original_date
                "description": transaction.get('description'),
                "debit": transaction.get('debit'),
                "credit": transaction.get('credit'),
                "balance": transaction.get('balance')
            }
            formatted_transactions.append(formatted_transaction)
        
        # Calculate comprehensive analytics
        logger.info("=" * 60)
        logger.info("CALCULATING ANALYTICS")
        logger.info("=" * 60)
        
        transaction_summary = self.calculate_transaction_summary(transactions)
        monthly_analysis = self.calculate_monthly_analysis(transactions)
        analytics = self.calculate_analytics(transactions, monthly_analysis)
        
        # Create the comprehensive result
        result = {
            "extraction_metadata": {
                "pdf_file": os.path.basename(pdf_path),
                "processed_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "pages_processed": pages_processed,
                "total_transactions": len(transactions)
            },
            "account_info": {
                "account_name": account_info.get('account_name'),
                "account_number": account_info.get('account_number'),
                "iban_number": account_info.get('iban_number'),
                "statement_date": account_info.get('statement_date'),
                "opening_balance": account_info.get('opening_balance'),
                "closing_balance": account_info.get('closing_balance')
            },
            "transaction_detail_depth": formatted_transactions,
            "transaction_summary": transaction_summary,
            "monthly_analysis": monthly_analysis,
            "analytics": analytics
        }
        
        return result
    
    def extract_from_pdf(self, pdf_path: str, output_json: Optional[str] = None) -> Dict[str, Any]:
        """Extract complete bank statement data from PDF"""
        logger.info(f"Processing complete bank statement: {pdf_path}")
        
        try:
            # Extract transactions
            logger.info("=" * 60)
            logger.info("EXTRACTING TRANSACTIONS")
            logger.info("=" * 60)
            transaction_data = self.transaction_extractor.extract_from_pdf(pdf_path)
            
            # Extract account information
            logger.info("=" * 60)
            logger.info("EXTRACTING ACCOUNT INFORMATION")
            logger.info("=" * 60)
            account_info = self.account_extractor.extract_from_pdf(pdf_path)
            
            # Count pages processed (get from PDF)
            import fitz
            doc = fitz.open(pdf_path)
            pages_processed = len(doc)
            doc.close()

            # Merge the data with analytics
            logger.info("=" * 60)
            logger.info("MERGING DATA WITH BALANCE FALLBACK AND ANALYTICS")
            logger.info("=" * 60)
            combined_result = self.merge_account_and_transactions(account_info, transaction_data, pdf_path, pages_processed)
            
            # Save to JSON if requested
            if output_json:
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(combined_result, f, indent=2, ensure_ascii=False)
                logger.info(f"Comprehensive statement data saved to: {output_json}")
            
            # Display final summary
            logger.info("=" * 60)
            logger.info("COMPREHENSIVE STATEMENT ANALYSIS SUMMARY")
            logger.info("=" * 60)
            
            metadata = combined_result["extraction_metadata"]
            account_info_final = combined_result["account_info"]
            transaction_summary = combined_result["transaction_summary"]
            analytics = combined_result["analytics"]
            
            logger.info("EXTRACTION METADATA:")
            logger.info(f"  • PDF File: {metadata['pdf_file']}")
            logger.info(f"  • Processed At: {metadata['processed_at']}")
            logger.info(f"  • Pages Processed: {metadata['pages_processed']}")
            logger.info(f"  • Total Transactions: {metadata['total_transactions']}")
            
            logger.info("ACCOUNT INFORMATION:")
            logger.info(f"  • Account Name: {account_info_final.get('account_name', 'Not found')}")
            logger.info(f"  • Account Number: {account_info_final.get('account_number', 'Not found')}")
            logger.info(f"  • IBAN: {account_info_final.get('iban_number', 'Not found')}")
            logger.info(f"  • Statement Date: {account_info_final.get('statement_date', 'Not found')}")
            logger.info(f"  • Opening Balance: {account_info_final.get('opening_balance', 'Not found')}")
            logger.info(f"  • Closing Balance: {account_info_final.get('closing_balance', 'Not found')}")
            
            logger.info("TRANSACTION SUMMARY:")
            logger.info(f"  • Total Deposits: {transaction_summary['total_deposits']['count']} transactions, {transaction_summary['total_deposits']['amount']} SAR")
            logger.info(f"  • Total Withdrawals: {transaction_summary['total_withdrawals']['count']} transactions, {transaction_summary['total_withdrawals']['amount']} SAR")
            logger.info(f"  • Net Change: {transaction_summary['net_change']['amount']} SAR")
            logger.info(f"  • Balance Range: {transaction_summary['balance_statistics']['minimum_balance']} - {transaction_summary['balance_statistics']['maximum_balance']} SAR")
            
            logger.info("ANALYTICS HIGHLIGHTS:")
            logger.info(f"  • Foreign Transactions: {analytics['total_foreign_transactions']} ({analytics['total_foreign_amount']} SAR)")
            logger.info(f"  • Average Fluctuation: {analytics['average_fluctuation']} SAR")
            logger.info(f"  • Cash Flow Stability: {analytics['net_cash_flow_stability']} SAR")
            logger.info(f"  • Total Inflow: {analytics['sum_total_inflow']} SAR")
            logger.info(f"  • Total Outflow: {analytics['sum_total_outflow']} SAR")
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return {
                "extraction_metadata": {
                    "pdf_file": os.path.basename(pdf_path) if pdf_path else "unknown",
                    "processed_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "pages_processed": 0,
                    "total_transactions": 0
                },
                "account_info": {
                    "account_name": None,
                    "account_number": None,
                    "iban_number": None,
                    "statement_date": None,
                    "opening_balance": None,
                    "closing_balance": None
                },
                "transaction_detail_depth": [],
                "transaction_summary": {
                    "total_deposits": {"count": 0, "amount": 0.0},
                    "total_withdrawals": {"count": 0, "amount": 0.0},
                    "net_change": {"amount": 0.0},
                    "balance_verification": {
                        "expected_closing_balance": 0.0,
                        "actual_closing_balance": 0.0,
                        "difference": 0.0
                    },
                    "balance_statistics": {
                        "minimum_balance": 0.0,
                        "maximum_balance": 0.0,
                        "average_balance": 0.0
                    }
                },
                "monthly_analysis": {},
                "analytics": self._empty_analytics()
            }

# Usage example
def main():
    # Initialize combined extractor
    extractor = CombinedStatementExtractor()
    
    # Extract complete statement from PDF
    pdf_path = "Bank statments/Toyland company , SNB(90).pdf"  # Replace with your PDF path
    output_json = "merged_statement.json"  # Changed filename as requested
    
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
        sys.exit(1)
    
    if not os.environ.get("MODEL_ID"):
        print("Error: MODEL_ID environment variable is required")
        print("Please check your .env file")
        sys.exit(1)
    
    result = main()
