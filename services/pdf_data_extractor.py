import re


def extract_patient_data(text):

    patient = {}

    # Age
    age = re.search(r"Age:\s*(\d+)", text, re.I)
    patient["age"] = int(age.group(1)) if age else None

    # Sex
    sex = re.search(r"Gender:\s*(Male|Female)", text, re.I)

    if sex:
        patient["sex"] = 1 if sex.group(1).lower() == "male" else 0
    else:
        patient["sex"] = None

    # Chest Pain Type
    cp_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-Anginal Pain": 2,
        "Asymptomatic": 3
    }

    cp = re.search(
        r"Chest Pain Type:\s*(.+)",
        text
    )

    patient["cp"] = (
        cp_map.get(cp.group(1).strip(), 0)
        if cp else None
    )

    # Blood Pressure
    bp = re.search(
        r"Resting Blood Pressure:\s*(\d+)",
        text
    )

    patient["trestbps"] = (
        int(bp.group(1))
        if bp else None
    )

    # Cholesterol
    chol = re.search(
        r"Serum Cholesterol:\s*(\d+)",
        text
    )

    patient["chol"] = (
        int(chol.group(1))
        if chol else None
    )

    # FBS
    fbs = re.search(
        r"Fasting Blood Sugar Above 120 mg/dL:\s*(Yes|No)",
        text,
        re.I
    )

    patient["fbs"] = (
        1 if fbs and fbs.group(1).lower() == "yes"
        else 0
    )

    # Rest ECG
    restecg_map = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }

    ecg = re.search(
        r"Resting ECG Result:\s*(.+)",
        text
    )

    patient["restecg"] = (
        restecg_map.get(ecg.group(1).strip(), 0)
        if ecg else 0
    )

    # Thalach
    thalach = re.search(
        r"Maximum Heart Rate Achieved:\s*(\d+)",
        text
    )

    patient["thalach"] = (
        int(thalach.group(1))
        if thalach else None
    )

    # Exang
    exang = re.search(
        r"Exercise Induced Angina:\s*(Yes|No)",
        text,
        re.I
    )

    patient["exang"] = (
        1 if exang and exang.group(1).lower() == "yes"
        else 0
    )

    # Oldpeak
    oldpeak = re.search(
        r"ST Depression Induced By Exercise:\s*([\d.]+)",
        text
    )

    patient["oldpeak"] = (
        float(oldpeak.group(1))
        if oldpeak else 0.0
    )

    # Slope
    slope_map = {
        "Downsloping": 0,
        "Flat": 1,
        "Upsloping": 2
    }

    slope = re.search(
        r"Slope Of Peak Exercise ST Segment:\s*(.+)",
        text
    )

    patient["slope"] = (
        slope_map.get(
            slope.group(1).strip(),
            1
        )
        if slope else 1
    )

    # CA
    ca = re.search(
        r"Number Of Major Vessels Colored By Fluoroscopy:\s*(\d+)",
        text
    )

    patient["ca"] = (
        int(ca.group(1))
        if ca else 0
    )

    # Thal
    thal_map = {
        "Normal": 2,
        "Fixed Defect": 1,
        "Reversible Defect": 3
    }

    thal = re.search(
        r"Thalassemia Status:\s*(.+)",
        text
    )

    patient["thal"] = (
        thal_map.get(
            thal.group(1).strip(),
            2
        )
        if thal else 2
    )

    return patient