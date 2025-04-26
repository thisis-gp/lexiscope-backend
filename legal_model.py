from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_qdrant import Qdrant
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
import google.generativeai as genai
import pandas as pd
import os
import uuid
from datetime import datetime
from typing import Dict, Optional

# In-memory session storage (replace with database in production)
sessions = {}

# Add CSV storage path
CSV_STORAGE_PATH = "case_metadata.csv"

TEXT_FILE_DIR = r"./supreme_court_cleaned_texts"
FAISS_INDEX_BASE = "faiss_index"

# Initialize CSV storage if not exists
if not os.path.exists(CSV_STORAGE_PATH):
    pd.DataFrame(columns=["id","source", "title", "judges", "date", "summary", "pdf_path", "summary_path"]).to_csv(CSV_STORAGE_PATH, index=False)

os.makedirs(FAISS_INDEX_BASE, exist_ok=True)

# Load environment variables
load_dotenv()

google_gemini_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
genai.configure(api_key=google_gemini_api_key)

# Create FastAPI app
app = FastAPI(title="Kanoon API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL for security (e.g., ["http://localhost:5173"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")  
QDRANT_API_KEY = os.getenv("QDRANT_CLOUD_API_KEY")
COLLECTION_NAME = "legal_documents" 

# Helper functions for CSV operations
def get_case_metadata(case_id: str) -> Optional[Dict]:
    try:
        df = pd.read_csv(CSV_STORAGE_PATH,dtype={'id': str})
        case_data = df[df["id"] == str(case_id)]
        print(case_data)
        # Check if any rows exist before accessing
        return case_data.iloc[0].to_dict() if not case_data.empty else None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def save_case_metadata(data: Dict):
    try:
        df = pd.read_csv(CSV_STORAGE_PATH,dtype={'id': str})
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

try:
    # Initialize components that will be reused
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )


    # Initialize Qdrant client
    qdrant_client = QdrantClient(
        url=QDRANT_CLOUD_URL,
        api_key=QDRANT_API_KEY,
    )

    # Check if collection exists before connecting
    existing_collections = [col.name for col in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in existing_collections:
        raise ValueError(f"Collection '{COLLECTION_NAME}' not found in Qdrant Cloud.")

    # Create Qdrant instance connected to existing collection
    qdrant = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )

    print(f"✅ Connected to Qdrant collection: {COLLECTION_NAME}")

except Exception as e:
    raise Exception(f"Failed to initialize components: {e}")

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=10, description="Number of top results to return")

class AddDocumentsRequest(BaseModel):
    file_path: str



# Global initialization of reusable components
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
prompt_template = """
Answer the question as detailed as possible from the provided context. 
If the answer is not available in the context, say "answer is not available in the context".

Context: \n {context}?\n
Question: \n {question}\n
Answer:
"""

# Pydantic Model for Structured Query
class StructuredQueryRequest(BaseModel):
    text: str
    source: str

# Pydantic Model for Chatbot Query
class ChatQueryRequest(BaseModel):
    text: str
    question: str

# API endpoints
@app.post("/query")
async def query_documents(request: QueryRequest):
    """Endpoint to search legal documents"""
    try:
        # Verify top_k is within allowed range
        if not (1 <= request.top_k <= 10):
            raise HTTPException(
                status_code=400,
                detail="top_k must be between 1 and 10"
            )
        
        retriever = qdrant.as_retriever(search_kwargs={"k": request.top_k})
        results = retriever.invoke(request.query)
        structured_results = []
        for doc in results:
            source = doc.metadata.get("source")
            case_id = extract_case_id(source)
            
            # Check CSV cache first
            cached_data = get_case_metadata(case_id)
            print(f"Cached data: {cached_data}")
            if cached_data:
                structured_results.append(cached_data)
                continue
                
            # Process with RAG if not in cache
            response = await structured_query(StructuredQueryRequest(text=doc.page_content, source=source))
            case_id = extract_case_id(source)
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
            
        print(f"Structured results: {structured_results}")
            
        return {
            "query": request.query,
            "top_k": request.top_k,
            "results": structured_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/add-documents")
async def add_documents(request: AddDocumentsRequest):
    """Endpoint to add new documents to the collection"""
    try:
        # Load and process new documents
        loader = DirectoryLoader(request.file_path)
        new_docs = loader.load()
        split_new_docs = text_splitter.split_documents(new_docs)
        
        # Add to existing collection
        qdrant.add_documents(split_new_docs)
        
        return {
            "message": f"Successfully added {len(split_new_docs)} document chunks",
            "new_chunks": len(split_new_docs),
            "collection": COLLECTION_NAME
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# First API - Structured Query API
@app.post("/structured_query")
async def structured_query(request: StructuredQueryRequest):
    """Processes text, trains the model, and answers predefined questions."""
    try:

        case_id = extract_case_id(request.source)

        # First check CSV cache
        cached_data = get_case_metadata(request.source)
        if cached_data:
            return {
                "status": "success",
                "source": "csv_cache",
                "data": cached_data
            }
        
        # Check if vector store exists
        if not vector_store_exists(case_id):
            # Create vector store only if needed
            create_vector_store(case_id, request.text)

        # Structured questions
        questions = [
            "What is the title of the case?",
            "Who are the judges?",
            "What is the date of the case?",
            "Provide a 50-word summary of the case."
        ]
        
        responses = {q: query_vector_store(q, case_id) for q in questions}

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
        raise HTTPException(status_code=500, detail=str(e))
    
# New models for chat
class ChatInitRequest(BaseModel):
    case_id: str  

class ChatMessageRequest(BaseModel):
    case_id: str
    question: str


# Function to split text into chunks
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to create vector store
def create_vector_store(case_id: str, text: str):
    index_path = os.path.join(FAISS_INDEX_BASE, case_id)
    if vector_store_exists(case_id):
        return index_path
    
    chunks = split_text_into_chunks(text)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    os.makedirs(index_path, exist_ok=True)

    vector_store.save_local(index_path)
    return index_path

# Add these new functions for vector store caching
def vector_store_exists(case_id: str) -> bool:
    index_dir = os.path.join(FAISS_INDEX_BASE, case_id)
    return os.path.exists(os.path.join(index_dir, "index.faiss"))

# Function to load QA model
def load_qa_model():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available in the context, say "answer is not available in the context".

    Context: \n {context}?\n
    Question: \n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to process user query
def query_vector_store(user_question, case_id):
    index_path = os.path.join(FAISS_INDEX_BASE, case_id)
    
    if not os.path.exists(index_path):
        raise ValueError(f"FAISS index not found for case {case_id}")
    
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    docs = vector_store.similarity_search(user_question)
    chain = load_qa_model()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def get_answer(case_id: str, question: str):
    index_dir = os.path.join(FAISS_INDEX_BASE, case_id)
    if not vector_store_exists(case_id):
        raise HTTPException(status_code=404, detail="Vector index not found. Initialize chat first.")
    
    try:
        vector_store = FAISS.load_local(
            index_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading vector store: {str(e)}")

    docs = vector_store.similarity_search(question)
    
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    try:
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

# Modified chat endpoint
@app.post("/chat_init")
async def initialize_chat_session(request: ChatInitRequest):
    """Initialize chat session with document text"""
    case_id = request.case_id
    text_file_path = os.path.join(TEXT_FILE_DIR, f"{case_id}.txt")
    
    if not os.path.exists(text_file_path):
        raise HTTPException(status_code=404, detail="Case text file not found")
    
    try:
        with open(text_file_path, "r") as file:
            text = file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading text file: {str(e)}")
    
    try:
        index_path = create_vector_store(case_id, text)
        return {"status": "ready", "message": "Chat initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing chat: {str(e)}")

@app.post("/chat_query")
async def handle_chat_query(request: ChatMessageRequest):
    """Handle chat query for initialized session"""
    try:
        answer = get_answer(request.case_id, request.question)
        return {"answer": answer}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_case_id(source_path: str) -> str:
    """Extract numeric ID from paths like 'supreme_court_cleaned_texts/8.txt'"""
    try:
        return source_path.split("/")[-1].split(".")[0]
    except:
        raise ValueError("Invalid source path format")

class CaseDetailsRequest(BaseModel):
    source: str

# Add to API endpoints
@app.get("/cases/{case_id}")
async def get_case_details(case_id: str):
    print(case_id)
    """Get complete case details including file paths"""
    try:
        case_data = get_case_metadata(case_id)
        print(case_data)
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
    uvicorn.run(app, host="0.0.0.0", port=8000)