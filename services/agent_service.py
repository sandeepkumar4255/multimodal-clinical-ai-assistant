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

    print("STEP 0 - PIPELINE STARTED")

    patient_notes = text or ""
    pdf_summary = None

    if pdf_path:
        try:
            print("STEP 1 - PDF EXTRACTION")

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

    print("STEP 2 - NLP")

    entities = extract_medical_entities(patient_notes)

    print("STEP 3 - ML")

    risk_score = predict_heart_risk(patient_data)

    print("RISK SCORE:", risk_score)

    if risk_score < 0.30:

        evidence = []

    else:

        print("STEP 4 - RAG")

        evidence = retrieve_documents(
            patient_notes,
            top_k=2
        )

    print("STEP 5 - LLM")

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

        print("STEP 6 - XRAY")

        # TEMPORARY TEST
        image_result = {
            "prediction": "TEST",
            "confidence": 1.0
        }

        # COMMENT THIS FOR NOW
        # image_result = analyze_xray(image_path)

    print("STEP 7 - RETURNING RESULT")

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
