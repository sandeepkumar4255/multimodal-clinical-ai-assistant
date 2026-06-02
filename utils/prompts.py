CLINICAL_PROMPT = """
You are an AI clinical assistant.

Analyze:
- symptoms
- disease risk
- medical evidence

Give evidence-backed recommendations.
"""

RAG_PROMPT = """
Context:
{context}

Question:
{question}

Generate a medical recommendation.
"""