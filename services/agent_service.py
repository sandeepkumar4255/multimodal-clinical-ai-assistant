from services.nlp_service import extract_medical_entities
from services.ml_service import predict_heart_risk
from services.pdf_service import extract_pdf_text
from services.llm_service import generate_response


def run_clinical_pipeline(
    text,
    patient_data,
    image_path=None,
    pdf_path=None
):

    print("PIPELINE STARTED")

    patient_notes = text or ""
    pdf_summary = None

    if pdf_path:
        try:
            print("PDF EXTRACTION")

            pdf_text = extract_pdf_text(pdf_path)

            pdf_summary = pdf_text[:1200]

            patient_notes = (
                f"{patient_notes}\n\n{pdf_text}"
                if patient_notes
                else pdf_text
            )

        except Exception as e:
            print("PDF ERROR:", str(e))
            pdf_summary = None

    print("NLP EXTRACTION")

    entities = extract_medical_entities(patient_notes)

    print("HEART RISK PREDICTION")

    risk_score = predict_heart_risk(patient_data)

    print("RISK SCORE:", risk_score)

    evidence = []

    print("LLM RESPONSE")

    recommendation = generate_response(
        context=patient_notes,
        question=patient_notes
    )

    if risk_score < 0.30:
        disease_status = "LOW RISK"
    elif risk_score < 0.65:
        disease_status = "MODERATE RISK"
    else:
        disease_status = "HIGH RISK"

    # Render-safe image response
    image_result = {
        "prediction": "ANALYSIS DISABLED IN CLOUD VERSION",
        "confidence": 0.0
    }

    print("PIPELINE COMPLETED")

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
