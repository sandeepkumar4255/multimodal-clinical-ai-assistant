from pydantic import BaseModel
from typing import List, Optional


class PatientRequest(BaseModel):

    text: str = ""

    image_path: Optional[str] = None
    pdf_path: Optional[str] = None


class ChatRequest(BaseModel):
    message: str


class AnalysisResponse(BaseModel):
    symptoms: List[str]
    conditions: List[str]
    medications: List[str]
    risk_score: float
    pdf_summary: Optional[str] = None
    evidence: List[str]
    recommendation: str
    disease_status: str
    image_analysis: dict