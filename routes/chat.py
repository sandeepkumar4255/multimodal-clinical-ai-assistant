from fastapi import APIRouter
from pydantic import BaseModel

from services.llm_service import generate_response
from services.rag_service import retrieve_documents
import services.session_store as store

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)


class ChatRequest(BaseModel):
    message: str


@router.post("/")
def chat(request: ChatRequest):

    patient_context = ""

    if store.latest_analysis:

        patient_context = f"""
Patient Analysis:

Disease Status:
{store.latest_analysis.get('disease_status')}

Risk Score:
{store.latest_analysis.get('risk_score')}

Symptoms:
{', '.join(store.latest_analysis.get('symptoms', []))}

Conditions:
{', '.join(store.latest_analysis.get('conditions', []))}

Medications:
{', '.join(store.latest_analysis.get('medications', []))}

Recommendation:
{store.latest_analysis.get('recommendation')}

Xray Result:
{store.latest_analysis.get('image_analysis', {}).get('prediction')}
"""

    question = request.message.lower()

    patient_keywords = [
        "above patient",
        "this patient",
        "patient",
        "medication",
        "medications",
        "risk",
        "risk score",
        "condition",
        "conditions",
        "symptoms",
        "recommendation",
        "xray",
        "x-ray"
    ]

    use_patient_context = any(
        keyword in question
        for keyword in patient_keywords
    )

    evidence = []

    if not use_patient_context:

        evidence = retrieve_documents(
            request.message,
            top_k=4
        )

    evidence_text = (
        "\n\n".join(evidence)
        if evidence
        else "No additional medical evidence."
    )

    context = f"""
You are an AI Clinical Assistant.

IMPORTANT RULES:

1. If the question refers to the current patient,
use Patient Analysis first.

2. Do not invent diseases,
medications, or treatments.

3. If Medications are empty,
say no medications are currently prescribed.

4. If medications exist,
only mention those medications.

5. Never suggest new medications
unless explicitly prescribed in the
patient analysis.

6. If asked for medication,
return only the medications listed
in Patient Analysis.

7. Use Medical Evidence only for
general medical questions.

Patient Analysis:
{patient_context}

Medical Evidence:
{evidence_text}
"""

    response = generate_response(
        context=context,
        question=request.message
    )

    return {
        "response": response,
        "evidence": evidence
    }