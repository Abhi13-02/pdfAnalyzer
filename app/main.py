from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from app.services.pdf_processor import extract_page_data


app = FastAPI()

# Configure CORS (adjust allowed origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] for all origins
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI!"}

@app.post("/analyze-pdf/")
async def analyze_pdf(file: UploadFile = File(...)):
    """
    Analyze a PDF file page by page by extracting:
      - Page text (using PyMuPDF)
      - OCR text (from rendered page image)
      - Embedded images and their descriptions (using Pollinations API)
    """
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    pages = extract_page_data(file_path)
    
    response_data = {"pages": pages}
    
    return JSONResponse(content=response_data)
