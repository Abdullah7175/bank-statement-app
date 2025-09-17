import os
import time
import shutil
import platform
import uuid
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the CombinedStatementExtractor class from ocr_account.py
from ocr_account import CombinedStatementExtractor

app = FastAPI(title="Bank Statement OCR API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the extractor globally (since it loads models)
extractor = None

def get_extractor():
    """Get or initialize the statement extractor"""
    global extractor
    if extractor is None:
        # Check environment variables
        if not os.environ.get("HF_TOKEN"):
            raise HTTPException(status_code=500, detail="HF_TOKEN environment variable not found")
        if not os.environ.get("MODEL_ID"):
            raise HTTPException(status_code=500, detail="MODEL_ID environment variable not found")
        
        extractor = CombinedStatementExtractor()
    return extractor

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main frontend page"""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...), request: Request = None):
    """Upload and process PDF bank statement"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Get user system information
    start_time = datetime.now()
    user_agent = request.headers.get("user-agent", "Unknown") if request else "Unknown"
    client_ip = request.client.host if request else "Unknown"
    
    # Get system information
    system_info = {
        "system_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": str(time.tzname),
        "region": os.environ.get("TZ", "UTC"),
        "browser": user_agent.split()[0] if user_agent != "Unknown" else "Unknown",
        "mac_address": ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(0,2*6,2)][::-1]),
        "public_ip": client_ip,
        "operating_system": f"{platform.system()} {platform.release()}",
        "starting_time": start_time.isoformat()
    }
    
    # Log user system information
    print("=" * 80)
    print("🔍 USER SYSTEM INFORMATION")
    print("=" * 80)
    for key, value in system_info.items():
        print(f"📋 {key.replace('_', ' ').title()}: {value}")
    print("=" * 80)
    
    # Create uploads directory if it doesn't exist
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    # Save uploaded file temporarily
    temp_path = uploads_dir / f"temp_{int(time.time())}.pdf"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the PDF using the extractor
        statement_extractor = get_extractor()
        result = statement_extractor.extract_from_pdf(str(temp_path))
        
        # Log ending time
        end_time = datetime.now()
        processing_duration = (end_time - start_time).total_seconds()
        
        print("=" * 80)
        print("🏁 PROCESSING COMPLETED")
        print("=" * 80)
        print(f"📋 Ending Time: {end_time.isoformat()}")
        print(f"📋 Total Processing Duration: {processing_duration:.2f} seconds")
        print("=" * 80)
        
        # Prepare frontend result with proper field names
        frontend_result = {
            "extraction_metadata": result["extraction_metadata"],
            "account_info": result["account_info"],
            "transaction_summary": result["transaction_summary"],
            "monthly_analysis": result["monthly_analysis"],
            "analytics": result["analytics"],
            "system_info": {
                **system_info,
                "ending_time": end_time.isoformat(),
                "processing_duration_seconds": processing_duration
            }
        }
        
        return frontend_result
        
    except Exception as e:
        print(f"❌ Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)