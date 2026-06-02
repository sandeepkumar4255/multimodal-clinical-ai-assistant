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

    patient_data = {
        "age": request.age,
        "sex": request.sex,
        "cp": request.cp,
        "trestbps": request.trestbps,
        "chol": request.chol,
        "fbs": request.fbs,
        "restecg": request.restecg,
        "thalach": request.thalach,
        "exang": request.exang,
        "oldpeak": request.oldpeak,
        "slope": request.slope,
        "ca": request.ca,
        "thal": request.thal
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