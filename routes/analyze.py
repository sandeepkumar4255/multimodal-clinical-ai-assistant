from fastapi import APIRouter

from models.schemas import PatientRequest
from services.agent_service import run_clinical_pipeline

import services.session_store as store

router = APIRouter(
    prefix="/analyze",
    tags=["Analyze"]
)


@router.post("/")
def analyze_patient(request: PatientRequest):

    patient_data = store.latest_patient_data

    if not patient_data:

        return {
            "status": "error",
            "message": "Please upload a patient PDF first."
        }

    result = run_clinical_pipeline(
        text=request.text,
        patient_data=patient_data,
        image_path=request.image_path,
        pdf_path=request.pdf_path
    )

    store.latest_analysis = result

    return {
        "status": "success",
        "result": result
    }