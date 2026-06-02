from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.upload import router as upload_router
from routes.analyze import router as analyze_router
from routes.chat import router as chat_router


app = FastAPI(
    title="AI Clinical Assistant API"
)


# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include Routes
app.include_router(upload_router)
app.include_router(analyze_router)
app.include_router(chat_router)


@app.get("/")
def home():

    return {
        "message": "Backend Running Successfully"
    }