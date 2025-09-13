import os
import io
import base64
import json
import fitz  # PyMuPDF
from PIL import Image
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from datetime import datetime
import statistics
import time
from collections import defaultdict
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# ---------- Load Environment Variables ----------
load_dotenv()
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-3.1-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Validate required environment variables
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required. Please set it in your .env file or environment.")

# ---------- FastAPI App ----------
app = FastAPI(title="Bank Statement OCR API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- PDF to images ----------
def pdf_to_images(pdf_path, dpi=150):
    doc = fitz.open(pdf_path)
    return [Image.open(io.BytesIO(page.get_pixmap(dpi=dpi).tobytes("png"))) for page in doc]

# ---------- Encode image ----------
def encode_image(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

# ---------- Hugging Face Client ----------
# Initialize Hugging Face client
try:
    client = InferenceClient(provider="fireworks-ai", api_key=HF_TOKEN)
    print("[INFO] Created Hugging Face client with fireworks-ai provider")
except Exception as e:
    print(f"[WARN] Failed to create client with provider, trying standard method: {e}")
    client = InferenceClient(
        model=MODEL_ID,
        token=HF_TOKEN,
    )

SYSTEM_PROMPT = """
You are a universal bank statement parser for English and Arabic.
Extract and return ONLY valid JSON in this schema:

{
  "account_number": str,
  "customer_name": str,
  "iban_number": str,
  "financial_period": str,
  "opening_balance": str,
  "closing_balance": str,
  "transaction_summary_depth": [
    {"date": str, "description": str, "debit": str, "credit": str, "balance": str}
  ]
}

Rules:
- Always return a JSON object.
- Dates must be YYYY-MM-DD (convert if needed).
- If only one column exists, use "-" sign for debit.
- Use strings for numbers.
- If a field is missing, set as null.
"""

def safe_json_parse(text):
    """Try to extract JSON from model output"""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {}

def parse_float(val):
    try:
        return float(str(val).replace(",", "").strip())
    except Exception:
        return 0.0

def parse_date(d):
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(d, fmt)
        except Exception:
            continue
    return None

def pct_fluctuation(values):
    """Percent fluctuation = pstdev(values)/mean(values)*100; safe for mean==0 or small lists."""
    vals = [parse_float(v) for v in values if v is not None]
    if len(vals) <= 1:
        return 0.0
    m = statistics.mean(vals)
    if m == 0:
        return 0.0
    return statistics.pstdev(vals) / m * 100.0

def extract_from_pdf_bytes(pdf_bytes):
    start_time = time.time()
    
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    
    # Save uploaded PDF to temporary file
    temp_pdf_path = f"uploads/temp_{int(time.time())}.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_bytes)
    
    try:
        images = pdf_to_images(temp_pdf_path)
        print(f"\n📑 Starting extraction for uploaded PDF")
        print(f"Total Pages: {len(images)}\n")

        merged = {
            "account_number": None,
            "customer_name": None,
            "iban_number": None,
            "financial_period": None,
            "opening_balance": None,
            "closing_balance": None,
            "transaction_summary_depth": []
        }

        for i, img in enumerate(images, start=1):
            print(f"🔎 Processing Page {i}...")

            # Encode image to base64 for API
            img_b64 = encode_image(img)
            
            try:
                # Try chat completions first (for fireworks-ai provider)
                try:
                    response = client.chat.completions.create(
                        model=MODEL_ID,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": [
                                {"type": "text", "text": "Extract all details and transactions from this statement."},
                                {"type": "image_url", "image_url": {"url": img_b64}}
                            ]}
                        ],
                        temperature=0,
                        max_tokens=3072,
                        response_format={"type": "json_object"}
                    )
                    raw_text = response.choices[0].message["content"]
                except AttributeError:
                    # Fallback to image_to_text + text_generation for standard InferenceClient
                    response = client.image_to_text(
                        image=img_b64,
                    )
                    raw_text = response[0].generated_text
                    
                    # Create a combined prompt for better extraction
                    combined_prompt = f"{SYSTEM_PROMPT}\n\nExtracted text from image:\n{raw_text}\n\nPlease extract structured bank statement data from the above text and return ONLY valid JSON."
                    
                    text_response = client.text_generation(
                        model=MODEL_ID,
                        inputs=combined_prompt,
                        max_new_tokens=4096,
                        temperature=0.0,
                    )
                    raw_text = text_response[0].generated_text
                
                page_json = safe_json_parse(raw_text)
                
            except Exception as e:
                print(f"❌ Error processing page {i}: {e}")
                page_json = {"transactions": []}

            # merge metadata (fill only if empty)
            for key in ["account_number", "customer_name", "iban_number",
                        "financial_period", "opening_balance", "closing_balance"]:
                if not merged[key] and page_json.get(key):
                    merged[key] = page_json.get(key)

            # merge transactions
            txns = page_json.get("transaction_summary_depth") or []
            merged["transaction_summary_depth"].extend(txns)

            print(f"✅ Page {i} extracted, {len(txns)} transactions")

        # sort transactions by date (invalid/missing dates go to the end)
        def sort_key(t):
            d = parse_date(t.get("date", ""))
            return d or datetime.max

        merged["transaction_summary_depth"].sort(key=sort_key)

        # ---------- Fix opening/closing balance ----------
        balance_fix_info = {
            "original_opening_balance": merged["opening_balance"],
            "original_closing_balance": merged["closing_balance"],
            "fixes_applied": []
        }

        if merged["transaction_summary_depth"]:
            first_bal = merged["transaction_summary_depth"][0].get("balance")
            last_bal = merged["transaction_summary_depth"][-1].get("balance")

            if (not merged["opening_balance"]) or (str(merged["opening_balance"]) in ["0", "0.0", "0.00"]):
                if first_bal not in [None, ""]:
                    merged["opening_balance"] = first_bal
                    balance_fix_info["fixes_applied"].append("Opening balance set from first transaction")

            if (not merged["closing_balance"]) or (str(merged["closing_balance"]) in ["0", "0.0", "0.00"]):
                if last_bal not in [None, ""]:
                    merged["closing_balance"] = last_bal
                    balance_fix_info["fixes_applied"].append("Closing balance set from last transaction")

        balance_fix_info["final_opening_balance"] = merged["opening_balance"]
        balance_fix_info["final_closing_balance"] = merged["closing_balance"]

        # ---------- Transaction Summary ----------
        deposits_vals = []
        withdrawals_vals = []
        balances_vals = []

        for t in merged["transaction_summary_depth"]:
            c = parse_float(t.get("credit"))
            d = parse_float(t.get("debit"))
            b = t.get("balance")
            if c > 0:
                deposits_vals.append(c)
            if d > 0:
                withdrawals_vals.append(d)
            if b not in [None, ""]:
                balances_vals.append(parse_float(b))

        total_deposits = sum(deposits_vals)
        total_withdrawals = sum(withdrawals_vals)
        net_change = total_deposits - total_withdrawals

        transaction_summary = {
            "total_deposits": {
                "count": len(deposits_vals),
                "amount": total_deposits,
                "formula": "sum(credit for all transactions where credit > 0)"
            },
            "total_withdrawals": {
                "count": len(withdrawals_vals),
                "amount": total_withdrawals,
                "formula": "sum(debit for all transactions where debit > 0)"
            },
            "net_change": {
                "amount": net_change,
                "formula": "total_deposits - total_withdrawals"
            },
            "balance_verification": {
                "expected_closing_balance": parse_float(merged["opening_balance"]) + net_change,
                "actual_closing_balance": parse_float(merged["closing_balance"]),
                "difference": (parse_float(merged["opening_balance"]) + net_change) - parse_float(merged["closing_balance"]),
                "formula": "opening_balance + net_change = closing_balance"
            }
        }

        # ---------- Balance Statistics ----------
        balance_stats = {
            "minimum_balance": min(balances_vals) if balances_vals else None,
            "maximum_balance": max(balances_vals) if balances_vals else None,
            "average_balance": statistics.mean(balances_vals) if balances_vals else None,
            "formulas": {
                "minimum_balance": "min(transaction_balance for all transactions)",
                "maximum_balance": "max(transaction_balance for all transactions)",
                "average_balance": "mean(transaction_balance for all transactions)"
            }
        }

        # ---------- Monthly Analysis ----------
        monthly_data = defaultdict(list)
        for t in merged["transaction_summary_depth"]:
            d = parse_date(t.get("date", ""))
            if d is None:
                continue
            month = d.strftime("%b")
            monthly_data[month].append(t)

        monthly_analysis = {}
        for month, txns in monthly_data.items():
            credits = [parse_float(t.get("credit")) for t in txns if parse_float(t.get("credit")) > 0]
            debits = [parse_float(t.get("debit")) for t in txns if parse_float(t.get("debit")) > 0]
            balances_month = [parse_float(t.get("balance")) for t in txns if t.get("balance") not in [None, ""]]

            opening_m = balances_month[0] if balances_month else None
            closing_m = balances_month[-1] if balances_month else None
            total_c = sum(credits)
            total_d = sum(debits)
            net_m = total_c - total_d
            fluct_m = pct_fluctuation(balances_month)

            monthly_analysis[month] = {
                "opening_balance": opening_m,
                "closing_balance": closing_m,
                "total_credit": total_c,
                "total_debit": total_d,
                "net_change": net_m,
                "fluctuation": fluct_m,
                "minimum_balance": min(balances_month) if balances_month else None,
                "maximum_balance": max(balances_month) if balances_month else None,
                "transaction_count": len(txns),
                "international_inward_count": 0,
                "international_outward_count": 0,
                "international_inward_total": 0,
                "international_outward_total": 0,
                "formulas": {
                    "opening_balance": "balance at start of month",
                    "closing_balance": "opening_balance + total_credit - total_debit",
                    "net_change": "total_credit - total_debit",
                    "fluctuation": "stdev(balances) / mean(balances) * 100"
                }
            }

            # Logging monthly breakdown
            print(
                f"📊 {month}: Opening={opening_m}, Closing={closing_m}, "
                f"Credits={round(total_c, 2)} ({len(credits)} txns), "
                f"Debits={round(total_d, 2)} ({len(debits)} txns), "
                f"Net={round(net_m, 2)}"
            )

        # ---------- Analytics ----------
        fluct_values = [v["fluctuation"] for v in monthly_analysis.values() if v["fluctuation"] is not None]
        analytics = {
            "average_fluctuation": statistics.mean(fluct_values) if fluct_values else None,
            "net_cash_flow_stability": None,
            "total_foreign_transactions": 0,
            "total_foreign_amount": 0,
            "overdraft_frequency": sum(1 for b in balances_vals if b < 0),
            "overdraft_total_days": sum(1 for b in balances_vals if b < 0),
            "sum_total_inflow": total_deposits,
            "sum_total_outflow": total_withdrawals,
            "avg_total_inflow": total_deposits / len(deposits_vals) if deposits_vals else 0.0,
            "avg_total_outflow": total_withdrawals / len(withdrawals_vals) if withdrawals_vals else 0.0
        }

        # ---------- Validation (placeholder) ----------
        validation = {
            "pdf_summary": {
                "deposit_count": None, "deposit_total": None,
                "withdrawal_count": None, "withdrawal_total": None
            },
            "extracted_summary": {
                "deposit_count": len(deposits_vals),
                "deposit_total": total_deposits,
                "withdrawal_count": len(withdrawals_vals),
                "withdrawal_total": total_withdrawals
            },
            "discrepancies": {
                "deposit_count_difference": None,
                "deposit_total_difference": None,
                "withdrawal_count_difference": None,
                "withdrawal_total_difference": None
            },
            "validation_passed": None
        }

        # ---------- Extraction Metadata ----------
        end_time = time.time()
        extraction_metadata = {
            "pdf_file": "uploaded_file.pdf",
            "processed_at": datetime.now().isoformat(),
            "pages_processed": len(images),
            "total_transactions": len(merged["transaction_summary_depth"]),
            "processing_time": round(end_time - start_time, 2)
        }

        # ---------- Wrap Final Output ----------
        final_output = {
            "extraction_metadata": extraction_metadata,
            "account_information": {
                "customer_name": merged["customer_name"],
                "account_number": merged["account_number"],
                "iban_number": merged["iban_number"],
                "financial_period": merged["financial_period"],
                "opening_balance": merged["opening_balance"],
                "closing_balance": merged["closing_balance"]
            },
            "transaction_summary": {
                **transaction_summary
            },
            "balance_statistics": balance_stats,
            "monthly_analysis": monthly_analysis,
            "analytics": analytics,
            "validation_against_pdf_summary": validation,
            "balance_fix_information": balance_fix_info,
            "transactions_summery_depth": merged["transaction_summary_depth"]
        }

        # Print logging summaries
        print("\n📌 Extraction Metadata:", json.dumps(extraction_metadata, indent=2))
        print("\n💰 Transaction Summary:", json.dumps(transaction_summary, indent=2))
        print("\n📈 Analytics:", json.dumps(analytics, indent=2))
        print(f"\n🏦 Overall Opening Balance: {merged['opening_balance']}, Closing Balance: {merged['closing_balance']}")

        # Mismatch warning if any
        diff = transaction_summary["balance_verification"]["difference"]
        if abs(diff) > 0.01:
            print(f"⚠️ Balance mismatch: expected vs actual differs by {round(diff, 2)}")

        print(f"\n⚡ Execution time: {round(end_time - start_time, 2)} seconds")
        print(f"\n🎯 Extraction finished successfully")

        return final_output

    finally:
        # Clean up temporary file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("frontend.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        pdf_bytes = await file.read()
        result = extract_from_pdf_bytes(pdf_bytes)
        
        # Remove fields that shouldn't be displayed on frontend
        display_result = {k: v for k, v in result.items() 
                        if k not in ["transactions_summery_depth", "validation_against_pdf_summary", "balance_fix_information"]}
        
        return {"success": True, "data": display_result}
    
    except Exception as e:
        print(f"❌ Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
