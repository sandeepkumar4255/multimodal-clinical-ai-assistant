import os

def retrieve_documents(query, top_k=2):

    query = query.lower()

    keyword_mapping = {
        "heart disease": "datasets/pubmed/heart_disease.txt",
        "coronary artery disease": "datasets/pubmed/heart_disease.txt",
        "chest pain": "datasets/pubmed/heart_disease.txt",
        "hypertension": "datasets/pubmed/heart_disease.txt",

        "pneumonia": "datasets/pubmed/pneumonia.txt",
        "cough": "datasets/pubmed/pneumonia.txt",
        "fever": "datasets/pubmed/pneumonia.txt",
        "shortness of breath": "datasets/pubmed/pneumonia.txt"
    }

    selected_files = set()

    for keyword, file_path in keyword_mapping.items():
        if keyword in query:
            selected_files.add(file_path)

    results = []

    for file_path in selected_files:

        if os.path.exists(file_path):

            with open(file_path, "r", encoding="utf-8") as file:
                results.append(file.read()[:1000])

    return results[:top_k]