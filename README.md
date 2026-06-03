# AI Clinical Assistant - Backend

## Overview

The AI Clinical Assistant Backend is a FastAPI-based healthcare application that combines machine learning, document processing, medical information retrieval, and large language models to assist in clinical decision-making.

## Features

* Patient PDF upload and processing
* Medical report text extraction
* Patient data extraction from reports
* Heart disease risk prediction using Machine Learning
* Chest X-ray analysis using CNN
* Medical document retrieval (RAG)
* AI-powered clinical recommendations using Groq LLM
* REST API endpoints for frontend integration

## Tech Stack

### Backend Framework

* FastAPI
* Uvicorn

### Machine Learning

* Scikit-Learn
* XGBoost
* Joblib

### Deep Learning

* PyTorch
* Torchvision

### NLP & AI

* Groq API
* Llama 3.1 8B Instant

### Document Processing

* PyMuPDF
* Python-Docx

## Project Structure

backend/

├── routes/

│ ├── upload.py

│ ├── analyze.py

│ └── chat.py

├── services/

│ ├── agent_service.py

│ ├── ml_service.py

│ ├── llm_service.py

│ ├── rag_service.py

│ ├── pdf_service.py

│ └── image_service.py

├── models/

│ ├── heart_model.pkl

│ └── xray_model.pth

├── uploads/

├── datasets/

├── config/

├── main.py

└── requirements.txt

## API Endpoints

### Upload PDF

POST /upload/pdf

### Upload Image

POST /upload/image

### Analyze Patient

POST /analyze/

### Clinical Chat

POST /chat/

## Machine Learning Pipeline

1. Upload patient report
2. Extract clinical information
3. Predict cardiovascular risk
4. Retrieve relevant medical evidence
5. Generate AI recommendation
6. Return structured clinical response

## Deployment

Backend deployed on Render:

https://multimodal-clinical-ai-assistant.onrender.com

## Future Improvements

* Vector database integration
* LangChain implementation
* Multi-patient session management
* Advanced medical RAG
* Cloud-based X-ray inference

## Author

Sandeep Kumar
