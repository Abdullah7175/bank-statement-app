import os
import time
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Import the extract_from_pdf function from ocr_account.py
from ocr_account import extract_from_pdf

app = FastAPI(title="Bank Statement OCR API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main frontend page"""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF bank statement"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Create uploads directory if it doesn't exist
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    # Save uploaded file temporarily
    temp_path = uploads_dir / f"temp_{int(time.time())}.pdf"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the PDF
        result = extract_from_pdf(str(temp_path))
        
        # Remove fields that shouldn't be displayed on frontend
        frontend_result = {
            "extraction_metadata": result["extraction_metadata"],
            "account_information": result["account_information"],
            "transaction_summary": result["transaction_summary"],
            "balance_statistics": result["balance_statistics"],
            "monthly_analysis": result["monthly_analysis"],
            "analytics": result["analytics"]
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
    uvicorn.run(app, host="0.0.0.0", port=8000)