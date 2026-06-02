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
You are an AI-powered clinical assistant.

Use the provided medical context to answer the user's question.

Medical Context:
{context}

User Question:
{question}

Give:
- possible condition
- recommendation
- evidence-backed response

Keep the response concise and professional.
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
    
