from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import httpx
import pandas as pd
from typing import Dict, Optional, List
import json

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="Kanoon API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CSV_STORAGE_PATH = "case_metadata.csv"
HUGGINGFACE_SPACE_URL = os.getenv("HUGGINGFACE_SPACE_URL")
HF_API_KEY = os.getenv("HF_API_KEY")

# Initialize CSV storage if not exists
if not os.path.exists(CSV_STORAGE_PATH):
    pd.DataFrame(columns=["id", "source", "title", "judges", "date", "summary", "pdf_path", "summary_path"]).to_csv(CSV_STORAGE_PATH, index=False)

# Helper functions for CSV operations
def get_case_metadata(case_id: str) -> Optional[Dict]:
    try:
        df = pd.read_csv(CSV_STORAGE_PATH, dtype={'id': str})
        case_data = df[df["id"] == str(case_id)]
        return case_data.iloc[0].to_dict() if not case_data.empty else None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def save_case_metadata(data: Dict):
    try:
        df = pd.read_csv(CSV_STORAGE_PATH, dtype={'id': str})
        new_id = str(data["id"])
        # Check for existing ID
        if new_id in df["id"].values:
            print(f"Case ID {new_id} already exists. Skipping save.")
            return
            
        # Append new data
        new_df = pd.DataFrame([data])
        new_df["id"] = new_df["id"].astype(str)  # Ensure ID is string
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(CSV_STORAGE_PATH, index=False)
    except Exception as e:
        print(f"Error saving to CSV: {e}")

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=10, description="Number of top results to return")

class AddDocumentsRequest(BaseModel):
    file_path: str

class StructuredQueryRequest(BaseModel):
    text: str
    source: str

class ChatInitRequest(BaseModel):
    case_id: str  

class ChatMessageRequest(BaseModel):
    case_id: str
    question: str

# HTTP client for making requests to Hugging Face Space
async def get_http_client():
    async with httpx.AsyncClient(timeout=60.0) as client:
        yield client

# API endpoints
@app.post("/query")
async def query_documents(request: QueryRequest, client: httpx.AsyncClient = Depends(get_http_client)):
    """Endpoint to search legal documents"""
    try:
        # Forward request to Hugging Face Space
        response = await client.post(
            f"{HUGGINGFACE_SPACE_URL}/query",
            json={"query": request.query, "top_k": request.top_k},
            headers={"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        result = response.json()
        
        # Cache results locally
        for item in result["results"]:
            save_case_metadata(item)
            
        return result
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error communicating with Hugging Face Space: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-documents")
async def add_documents(request: AddDocumentsRequest):
    """Endpoint to add new documents to the collection"""
    # This would require file upload functionality
    # For now, we'll just return a message
    return {
        "message": "Document addition is handled through the admin interface",
        "status": "not_implemented"
    }

@app.post("/structured_query")
async def structured_query(request: StructuredQueryRequest, client: httpx.AsyncClient = Depends(get_http_client)):
    """Processes text and answers predefined questions."""
    try:
        # Check if we already have this case in our CSV
        case_id = request.source.split("/")[-1].split(".")[0] if request.source else None
        if case_id:
            cached_data = get_case_metadata(case_id)
            if cached_data:
                return {
                    "status": "success",
                    "source": "csv_cache",
                    "data": cached_data
                }
        
        # Forward request to Hugging Face Space
        response = await client.post(
            f"{HUGGINGFACE_SPACE_URL}/structured_query",
            json={"text": request.text, "source": request.source},
            headers={"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        result = response.json()
        
        # Cache result locally
        if "data" in result:
            save_case_metadata(result["data"])
            
        return result
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error communicating with Hugging Face Space: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_init")
async def initialize_chat_session(request: ChatInitRequest):
    """Initialize chat session with document text"""
    # This is now a lightweight operation - we just check if the case exists
    case_id = request.case_id
    case_data = get_case_metadata(case_id)
    
    if not case_data:
        raise HTTPException(status_code=404, detail="Case not found")
    
    return {"status": "ready", "message": "Chat initialized successfully"}

@app.post("/chat_query")
async def handle_chat_query(request: ChatMessageRequest, client: httpx.AsyncClient = Depends(get_http_client)):
    """Handle chat query for initialized session"""
    try:
        # Forward request to Hugging Face Space
        response = await client.post(
            f"{HUGGINGFACE_SPACE_URL}/chat_query",
            json={"case_id": request.case_id, "question": request.question},
            headers={"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        return response.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error communicating with Hugging Face Space: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cases/{case_id}")
async def get_case_details(case_id: str):
    """Get complete case details including file paths"""
    try:
        case_data = get_case_metadata(case_id)
        if not case_data:
            raise HTTPException(status_code=404, detail="Case not found")
            
        return {
            "id": case_data["id"],
            "title": case_data["title"],
            "judges": case_data["judges"],
            "date": case_data["date"],
            "summary": case_data["summary"],
            "pdf_path": f"{case_data['pdf_path']}", 
            "summary_path": f"{case_data['summary_path']}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))