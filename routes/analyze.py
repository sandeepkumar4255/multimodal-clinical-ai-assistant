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

    try:

        print("=" * 50)
        print("ANALYZE API HIT")
        print("Text:", request.text)
        print("Image Path:", request.image_path)
        print("PDF Path:", request.pdf_path)
        print("=" * 50)

        patient_data = store.latest_patient_data

        print("Patient Data:", patient_data)

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

        print("ANALYSIS COMPLETED SUCCESSFULLY")

        return {
            "status": "success",
            "result": result
        }

    except Exception as e:

        print("ANALYZE ERROR:", str(e))

        return {
            "status": "error",
            "message": str(e)
        }
