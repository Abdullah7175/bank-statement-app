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

# ---------- Load Environment Variables ----------
load_dotenv()
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

PDF_PATH = "Bank statments/68200263967000 , Alinma.pdf"
OUTPUT_JSON = "merged_statement.json"

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
client = InferenceClient(
    provider="fireworks-ai",
    api_key=HF_TOKEN,
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

def extract_from_pdf(pdf_path):
    start_time = time.time()
    images = pdf_to_images(pdf_path)
    print(f"\n📑 Starting extraction for: {pdf_path}")
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

        img_b64 = encode_image(img)

        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract structured bank statement data from this page."},
                    {"type": "image_url", "image_url": {"url": img_b64}}
                ]}
            ],
            max_tokens=4096,
            temperature=0.0,
        )

        # NOTE: fireworks returns choices[0].message.content; keep as per user's environment
        page_json = safe_json_parse(response.choices[0].message["content"])

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
        fluct_m = pct_fluctuation(balances_month)  # SAFE: no ZeroDivisionError

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

        # Logging monthly breakdown (with opening/closing)
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
        "net_cash_flow_stability": None,  # placeholder metric
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
        "pdf_file": os.path.basename(pdf_path),
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

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    # Print logging summaries (including overall opening/closing)
    print("\n📌 Extraction Metadata:", json.dumps(extraction_metadata, indent=2))
    print("\n💰 Transaction Summary:", json.dumps(transaction_summary, indent=2))
    print("\n📈 Analytics:", json.dumps(analytics, indent=2))
    print(f"\n🏦 Overall Opening Balance: {merged['opening_balance']}, Closing Balance: {merged['closing_balance']}")

    # Mismatch warning if any
    diff = transaction_summary["balance_verification"]["difference"]
    if abs(diff) > 0.01:
        print(f"⚠️ Balance mismatch: expected vs actual differs by {round(diff, 2)}")

    print(f"\n⚡ Execution time: {round(end_time - start_time, 2)} seconds")
    print(f"\n🎯 Extraction finished. Saved merged JSON to {OUTPUT_JSON}")
    return final_output

if __name__ == "__main__":
    extract_from_pdf(PDF_PATH)

