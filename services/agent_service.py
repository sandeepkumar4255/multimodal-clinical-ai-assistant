from services.nlp_service import extract_medical_entities
from services.ml_service import predict_heart_risk
from services.rag_service import retrieve_documents
from services.image_service import analyze_xray
from services.pdf_service import extract_pdf_text
from services.llm_service import generate_response


def run_clinical_pipeline(
    text,
    patient_data,
    image_path=None,
    pdf_path=None
):

    patient_notes = text or ""
    pdf_summary = None

    if pdf_path:
        try:
            pdf_text = extract_pdf_text(pdf_path)
            pdf_summary = pdf_text[:1200]
            patient_notes = (
                f"{patient_notes}\n\n{pdf_text}"
                if patient_notes
                else pdf_text
            )
        except Exception:
            pdf_summary = None

    # NLP Extraction
    entities = extract_medical_entities(patient_notes)

    # ML Prediction
    risk_score = predict_heart_risk(patient_data)

    # RAG Evidence
    evidence = retrieve_documents(patient_notes)

    # LLM Recommendation
    recommendation = generate_response(
        context="\n\n".join(evidence) if evidence else patient_notes,
        question=patient_notes
    )

    image_result = {
        "prediction": "UNKNOWN",
        "confidence": 0.0
    }

    if risk_score < 0.30:
        disease_status = "LOW RISK"
    elif risk_score < 0.65:
        disease_status = "MODERATE RISK"
    else:
        disease_status = "HIGH RISK"

    if image_path:

        image_result = analyze_xray(image_path)

        if image_result["prediction"] == "NORMAL":

            if disease_status == "LOW RISK":

                recommendation = (
                    "Image appears normal. Continue routine monitoring "
                    "and follow a healthy lifestyle."
                )

            else:

                recommendation = (
                    "The X-ray appears normal, but clinical risk factors "
                    "indicate elevated cardiovascular risk. "
                    "Recommend specialist review and follow-up."
                )

        elif image_result["prediction"] == "PNEUMONIA":

            disease_status = "HIGH RISK"

            recommendation = (
                "Image findings suggest pneumonia. "
                "Urgent clinical review and treatment are advised."
            )

    return {
        "disease_status": disease_status,
        "symptoms": entities["symptoms"],
        "conditions": entities["conditions"],
        "medications": entities["medications"],
        "risk_score": round(float(risk_score), 2),
        "image_analysis": image_result,
        "pdf_summary": pdf_summary,
        "evidence": evidence,
        "recommendation": recommendation
    }