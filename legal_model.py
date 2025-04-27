from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Any
import os
import pandas as pd
import uuid
import httpx
import redis
import logging
import time
import json
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from functools import lru_cache
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("kanoon-api")

# Constants
CSV_STORAGE_PATH = "case_metadata.csv"
TEXT_FILE_DIR = "./supreme_court_cleaned_texts"
FAISS_INDEX_BASE = "faiss_index"
COLLECTION_NAME = "legal_documents"

# Configuration
class Settings:
    HUGGINGFACE_API_URL = os.getenv("HUGGINGFACE_API_URL")
    REDIS_URL = os.getenv("REDIS_URL", None)
    QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_CLOUD_API_KEY")
    API_KEY = os.getenv("API_KEY")  # For simple API key auth

@lru_cache()
def get_settings():
    return Settings()

# Initialize Redis if available
redis_client = None
if get_settings().REDIS_URL:
    try:
        redis_client = redis.from_url(get_settings().REDIS_URL)
        logger.info("Redis client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {e}")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(title="Kanoon API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize CSV storage if not exists
if not os.path.exists(CSV_STORAGE_PATH):
    pd.DataFrame(columns=["id", "source", "title", "judges", "date", "summary", "pdf_path", "summary_path"]).to_csv(CSV_STORAGE_PATH, index=False)

os.makedirs(FAISS_INDEX_BASE, exist_ok=True)

# HTTP client for external API calls
@lru_cache()
def get_http_client():
    return httpx.AsyncClient(timeout=60.0)

# Helper functions for CSV operations
def get_case_metadata(case_id: str) -> Optional[Dict]:
    try:
        # Try Redis cache first
        if redis_client:
            cached_data = redis_client.get(f"case:{case_id}")
            if cached_data:
                logger.info(f"Cache hit for case {case_id}")
                return json.loads(cached_data)
        
        # Fall back to CSV
        df = pd.read_csv(CSV_STORAGE_PATH, dtype={'id': str})
        case_data = df[df["id"] == str(case_id)]
        
        # Check if any rows exist before accessing
        if not case_data.empty:
            result = case_data.iloc[0].to_dict()
            
            # Cache in Redis if available
            if redis_client:
                redis_client.setex(
                    f"case:{case_id}", 
                    3600,  # 1 hour expiry
                    json.dumps(result)
                )
            
            return result
        return None
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return None

def save_case_metadata(data: Dict):
    try:
        df = pd.read_csv(CSV_STORAGE_PATH, dtype={'id': str})
        new_id = str(data["id"])
        
        # Check for existing ID
        if new_id in df["id"].values:
            logger.info(f"Case ID {new_id} already exists. Skipping save.")
            return
            
        # Append new data
        new_df = pd.DataFrame([data])
        new_df["id"] = new_df["id"].astype(str)  # Ensure ID is string
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(CSV_STORAGE_PATH, index=False)
        
        # Update Redis cache if available
        if redis_client:
            redis_client.setex(
                f"case:{new_id}", 
                3600,  # 1 hour expiry
                json.dumps(data)
            )
            
        logger.info(f"Saved metadata for case {new_id}")
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")

# Pydantic models
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

# Simple API key authentication
async def verify_api_key(request: Request):
    api_key = get_settings().API_KEY
    if not api_key:
        return True  # Skip auth if no API key is set
        
    if request.headers.get("X-API-Key") != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# External API functions
async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings from external service"""
    client = get_http_client()
    try:
        response = await client.post(
            f"{get_settings().HUGGINGFACE_API_URL}/embeddings",
            json={"texts": texts},
            timeout=60.0
        )
        response.raise_for_status()
        return response.json()["embeddings"]
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting embeddings: {str(e)}")

async def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 20) -> List[str]:
    """Split text using external service"""
    client = get_http_client()
    try:
        response = await client.post(
            f"{get_settings().HUGGINGFACE_API_URL}/split-text",
            json={"text": text, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
            timeout=60.0
        )
        response.raise_for_status()
        return response.json()["chunks"]
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        raise HTTPException(status_code=500, detail=f"Error splitting text: {str(e)}")

async def get_answer_from_context(context: List[str], question: str) -> str:
    """Get answer from external QA service"""
    client = get_http_client()
    try:
        response = await client.post(
            f"{get_settings().HUGGINGFACE_API_URL}/qa",
            json={"context": context, "question": question},
            timeout=60.0
        )
        response.raise_for_status()
        return response.json()["answer"]
    except Exception as e:
        logger.error(f"Error getting answer: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting answer: {str(e)}")

# Qdrant client functions
async def query_qdrant(query: str, top_k: int = 5) -> List[Dict]:
    """Query Qdrant directly via API"""
    client = get_http_client()
    try:
        # First get embeddings for the query
        query_embedding = await get_embeddings([query])
        
        # Then search Qdrant
        response = await client.post(
            f"{get_settings().QDRANT_CLOUD_URL}/collections/{COLLECTION_NAME}/points/search",
            headers={"api-key": get_settings().QDRANT_API_KEY},
            json={
                "vector": query_embedding[0],
                "limit": top_k,
                "with_payload": True
            }
        )
        response.raise_for_status()
        results = response.json()
        
        # Extract and format results
        formatted_results = []
        for hit in results.get("result", []):
            payload = hit.get("payload", {})
            formatted_results.append({
                "score": hit.get("score", 0),
                "page_content": payload.get("page_content", ""),
                "metadata": {
                    "source": payload.get("metadata", {}).get("source", "")
                }
            })
        
        return formatted_results
    except Exception as e:
        logger.error(f"Error querying Qdrant: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying Qdrant: {str(e)}")

# Cache decorator
def cache_response(ttl_seconds=3600):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if not redis_client:
                return await func(*args, **kwargs)
            
            # Create a cache key from function name and arguments
            cache_key = f"cache:{func.__name__}:{str(args)}:{str(kwargs)}"
            cached_result = redis_client.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache hit for {cache_key}")
                return json.loads(cached_result)
            
            result = await func(*args, **kwargs)
            
            # Cache the result
            redis_client.setex(
                cache_key,
                ttl_seconds,
                json.dumps(result)
            )
            
            return result
        return wrapper
    return decorator

# Extract case ID from source path
def extract_case_id(source_path: str) -> str:
    """Extract numeric ID from paths like 'supreme_court_cleaned_texts/8.txt'"""
    try:
        return source_path.split("/")[-1].split(".")[0]
    except:
        raise ValueError("Invalid source path format")

# API endpoints
@app.post("/query")
@limiter.limit("10/minute")
@cache_response(ttl_seconds=300)  # Cache for 5 minutes
async def query_documents(request: Request, query_request: QueryRequest): 
    """Endpoint to search legal documents"""
    try:
        # Verify top_k is within allowed range
        if not (1 <= request.top_k <= 10):
            raise HTTPException(
                status_code=400,
                detail="top_k must be between 1 and 10"
            )
        
        # Query Qdrant
        results = await query_qdrant(request.query, request.top_k)
        
        structured_results = []
        for doc in results:
            source = doc["metadata"]["source"]
            case_id = extract_case_id(source)
            
            # Check cache first
            cached_data = get_case_metadata(case_id)
            if cached_data:
                structured_results.append(cached_data)
                continue
                
            # Process with RAG if not in cache
            response = await structured_query(StructuredQueryRequest(text=doc["page_content"], source=source))
            structured_data = {
                "id": case_id,
                "source": source,
                "title": response["data"]["title"],
                "judges": response["data"]["judges"],
                "date": response["data"]["date"],
                "summary": response["data"]["summary"]
            }
            save_case_metadata(structured_data)
            structured_results.append(structured_data)
            
        return {
            "query": request.query,
            "top_k": request.top_k,
            "results": structured_results
        }
    except Exception as e:
        logger.error(f"Error in query_documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/structured_query")
@limiter.limit("20/minute")
@cache_response(ttl_seconds=3600)  # Cache for 1 hour
async def structured_query(request: StructuredQueryRequest, req: Request = None, api_key: bool = Depends(verify_api_key)):
    """Processes text and extracts structured information"""
    try:
        case_id = extract_case_id(request.source)

        # First check cache
        cached_data = get_case_metadata(case_id)
        if cached_data:
            return {
                "status": "success",
                "source": "csv_cache",
                "data": cached_data
            }
        
        # Structured questions
        questions = [
            "What is the title of the case?",
            "Who are the judges?",
            "What is the date of the case?",
            "Provide a 50-word summary of the case."
        ]
        
        # Split text into chunks
        chunks = await split_text(request.text)
        
        # Get answers for each question
        responses = {}
        for q in questions:
            responses[q] = await get_answer_from_context(chunks, q)

        # Create structured data
        structured_data = {
            "id": case_id,
            "source": request.source,
            "title": responses["What is the title of the case?"],
            "judges": responses["Who are the judges?"],
            "date": responses["What is the date of the case?"],
            "summary": responses["Provide a 50-word summary of the case."],
            "pdf_path": f"{case_id}.pdf",
            "summary_path": f"{case_id}.txt" 
        }

        # Save to CSV
        save_case_metadata(structured_data)

        return {
            "status": "success",
            "source": "rag_processed",
            "data": structured_data
        }

    except Exception as e:
        logger.error(f"Error in structured_query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_init")
@limiter.limit("10/minute")
async def initialize_chat_session(request: ChatInitRequest, req: Request, api_key: bool = Depends(verify_api_key)):
    """Initialize chat session with document text"""
    case_id = request.case_id
    text_file_path = os.path.join(TEXT_FILE_DIR, f"{case_id}.txt")
    
    if not os.path.exists(text_file_path):
        raise HTTPException(status_code=404, detail="Case text file not found")
    
    try:
        with open(text_file_path, "r") as file:
            text = file.read()
    except Exception as e:
        logger.error(f"Error reading text file: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading text file: {str(e)}")
    
    try:
        # Split text and store chunks in Redis
        chunks = await split_text(text)
        
        # Store chunks in Redis if available
        if redis_client:
            redis_client.setex(
                f"chunks:{case_id}",
                3600 * 24,  # 24 hour expiry
                json.dumps(chunks)
            )
        
        return {"status": "ready", "message": "Chat initialized successfully"}
    except Exception as e:
        logger.error(f"Error initializing chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error initializing chat: {str(e)}")

@app.post("/chat_query")
@limiter.limit("20/minute")
async def handle_chat_query(request: ChatMessageRequest, req: Request, api_key: bool = Depends(verify_api_key)):
    """Handle chat query for initialized session"""
    try:
        # Try to get chunks from Redis
        chunks = None
        if redis_client:
            cached_chunks = redis_client.get(f"chunks:{request.case_id}")
            if cached_chunks:
                chunks = json.loads(cached_chunks)
        
        # If not in Redis, read from file and process
        if not chunks:
            text_file_path = os.path.join(TEXT_FILE_DIR, f"{request.case_id}.txt")
            if not os.path.exists(text_file_path):
                raise HTTPException(status_code=404, detail="Case text file not found")
                
            with open(text_file_path, "r") as file:
                text = file.read()
                
            chunks = await split_text(text)
            
            # Store in Redis for future use
            if redis_client:
                redis_client.setex(
                    f"chunks:{request.case_id}",
                    3600 * 24,  # 24 hour expiry
                    json.dumps(chunks)
                )
        
        # Get answer from external service
        answer = await get_answer_from_context(chunks, request.question)
        return {"answer": answer}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in handle_chat_query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cases/{case_id}")
@limiter.limit("30/minute")
@cache_response(ttl_seconds=3600)  # Cache for 1 hour
async def get_case_details(case_id: str, req: Request, api_key: bool = Depends(verify_api_key)):
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
        logger.error(f"Error in get_case_details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check the health of the service"""
    health_data = {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": time.time(),
        "services": {
            "redis": "connected" if redis_client else "not_configured",
        }
    }
    
    # Check HuggingFace service
    try:
        client = get_http_client()
        response = await client.get(f"{get_settings().HUGGINGFACE_API_URL}/health")
        if response.status_code == 200:
            health_data["services"]["huggingface"] = "healthy"
        else:
            health_data["services"]["huggingface"] = "unhealthy"
    except Exception:
        health_data["services"]["huggingface"] = "unreachable"
    
    return health_data

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Generate request ID for tracking
    request_id = str(uuid.uuid4())
    logger.info(f"Request {request_id} started: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Request {request_id} completed: {response.status_code} in {process_time:.4f}s")
        
        # Add custom headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request {request_id} failed: {str(e)} in {process_time:.4f}s")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
