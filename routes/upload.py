import os
from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
from services.pdf_service import extract_pdf_text
from services.pdf_data_extractor import extract_patient_data
import services.session_store as store

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

PDF_CONTENT_TYPES = {"application/pdf", "text/plain"}
IMAGE_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/tiff", "image/bmp"}

router = APIRouter(prefix="/upload", tags=["Upload"])

@router.post("/pdf")
async def upload_pdf(file: UploadFile = File(...)):

    filename = os.path.basename(file.filename)
    content_type = file.content_type.lower()
    is_pdf = content_type == "application/pdf" or filename.lower().endswith(".pdf")
    is_text = content_type == "text/plain" or filename.lower().endswith(".txt")

    if not (is_pdf or is_text):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF or TXT uploads are allowed.")

    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if is_pdf:
        extracted_text = extract_pdf_text(file_path)
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as text_file:
            extracted_text = text_file.read()

    patient_data = extract_patient_data(
    extracted_text
)
    store.latest_patient_data = patient_data

    return {
    "message": "File Uploaded",
    "file_path": file_path,
    "extracted_text": extracted_text,
    "patient_data": patient_data
    }

@router.post("/image")
async def upload_image(file: UploadFile = File(...)):

    filename = os.path.basename(file.filename)
    content_type = file.content_type.lower()

    if content_type not in IMAGE_CONTENT_TYPES and not any(filename.lower().endswith(ext) for ext in [".jpeg", ".jpg", ".png", ".tiff", ".bmp"]):
        raise HTTPException(status_code=400, detail="Invalid file type. Only image uploads are allowed.")

    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "Image Uploaded",
        "file_path": file_path
    }