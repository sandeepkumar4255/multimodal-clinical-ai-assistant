SYMPTOMS = [
    "chest pain",
    "shortness of breath",
    "fever",
    "cough",
    "fatigue",
    "dizziness",
    "nausea",
    "palpitations"
]

CONDITIONS = [
    "hypertension",
    "diabetes",
    "pneumonia",
    "coronary artery disease",
    "heart disease",
    "asthma",
    "copd",
    "stroke"
]

MEDICATIONS = [
    "aspirin",
    "atorvastatin",
    "statin",
    "metformin",
    "insulin",
    "amoxicillin",
    "lisinopril"
]


def is_negated(text, keyword):

    text = text.lower()

    negation_patterns = [
        f"no {keyword}",
        f"no signs of {keyword}",
        f"no evidence of {keyword}",
        f"without {keyword}",
        f"negative for {keyword}",
        f"absence of {keyword}"
    ]

    return any(pattern in text for pattern in negation_patterns)


def extract_medical_entities(text):

    normalized = text.lower()
    normalized = normalized.replace(
    "chest pain type",
    ""
    )

    symptoms = []
    conditions = []
    medications = []

    # Symptoms
    for symptom in SYMPTOMS:

        if symptom in normalized:

            if not is_negated(normalized, symptom):

                symptoms.append(symptom)

    # Conditions
    for condition in CONDITIONS:

        if condition in normalized:

            # Handle reports like:
            # "No signs of coronary artery disease, hypertension, diabetes, or pneumonia"

            if "no signs of" in normalized:
                continue

            if "no evidence of" in normalized:
                continue

            if not is_negated(normalized, condition):

                conditions.append(condition)

    # Medications
    for medication in MEDICATIONS:

        if medication in normalized:

            medications.append(medication)

    return {
        "symptoms": list(set(symptoms)),
        "conditions": list(set(conditions)),
        "medications": list(set(medications))
    }