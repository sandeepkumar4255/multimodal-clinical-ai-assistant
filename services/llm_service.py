from groq import Groq
from config.settings import GROQ_API_KEY

if GROQ_API_KEY:
    print("GROQ KEY:", GROQ_API_KEY[:10] + "...")
else:
    print("GROQ API KEY NOT FOUND")

client = Groq(
    api_key=GROQ_API_KEY
) if GROQ_API_KEY else None

def generate_response(context, question):

    prompt = f"""
You are an AI clinical assistant.

Patient Analysis:
{context}

User Question:
{question}

IMPORTANT:
Only answer using the patient analysis above.
Do not invent diseases, medications, or diagnoses.
If the patient is low risk, say so.
If no medications are prescribed, say so.

Provide:
1. Possible condition
2. Recommendation
3. Supporting evidence
"""

    if client is None:
        return (
            "No LLM API key is configured. "
            "Use the extracted evidence and clinical notes to advise follow-up care, risk monitoring, and specialist referral if required."
        )

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=500
        )

        return completion.choices[0].message.content
    except Exception as e:
        print("LLM ERROR:", str(e))
        return f"LLM Error: {str(e)}"
    
