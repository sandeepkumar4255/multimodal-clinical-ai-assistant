from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).resolve().parent.parent / ".env"

load_dotenv(env_path, override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")