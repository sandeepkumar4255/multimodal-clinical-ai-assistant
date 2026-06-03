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

    # Check if patient analysis exists
    if store.latest_analysis is None:

        return {
            "response": (
                "Please upload and analyze a patient report first "
                "before asking patient-specific questions."
            ),
            "evidence": []
        }

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
        "x-ray",
        "diagnosis",
        "treatment",
        "report"
    ]

    use_patient_context = any(
        keyword in question
        for keyword in patient_keywords
    )

    evidence = []

    # Only use RAG for general medical questions
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

    patient_context = f"""
Current Patient Analysis:

{store.latest_analysis}
"""

    context = f"""
You are an AI Clinical Assistant.

STRICT RULES:

1. Use Patient Analysis as the primary source.
2. Do NOT invent diseases, medications, diagnoses, or treatments.
3. If medications are empty, say:
   "No medications are currently prescribed."
4. Only mention medications present in Patient Analysis.
5. If patient is LOW RISK, clearly state that.
6. If patient is healthy, clearly state that.
7. Use Medical Evidence only for general medical questions.
8. If information is unavailable, say:
   "Information not available in the patient analysis."

PATIENT ANALYSIS:
{patient_context}

MEDICAL EVIDENCE:
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