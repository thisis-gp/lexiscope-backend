from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_community.vectorstores import Qdrant
import google.generativeai as genai
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()
FILE_PATH = 'data/Object_casedocs_500/'
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")  
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "legal_documents" 

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap=20,
#     length_function=len,
#     is_separator_regex=False,
# )

# Load legal case docs
loader = DirectoryLoader(FILE_PATH)
docs = loader.load()
# docs = text_splitter.split_documents(docs)

# define embedding model
emb_model="sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=emb_model,
    cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME'),
)

output_dim = 384

# Define vectors configuration
vector_params = VectorParams(
    size=output_dim,  # Set to the size of your embeddings
    distance=Distance.EUCLID
)

qdrant_client = QdrantClient(
url=QDRANT_CLOUD_URL,
api_key=QDRANT_API_KEY,
)

# Check if collection exists and create it if not
collections = [collection.name for collection in qdrant_client.get_collections().collections]
print("Existing collections:", collections)

if COLLECTION_NAME not in collections:
    print(f"Creating collection: {COLLECTION_NAME}")
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=vector_params
    )
    
    # Insert documents into Qdrant with embeddings
    for i,doc in enumerate(docs):
        point = {
            "id": i,  # Ensure each doc has a unique identifier
            "payload": {"content": doc.page_content},
            "vector": embeddings.embed_documents(doc.page_content)[0]
        }
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )
    print("Documents inserted successfully")
else:
    print(f"Collection {COLLECTION_NAME} already exists")

# construct a retriever on top of the vector store


def get_relevant_legal_docs(query:str):
    try:
        embedded_query = embeddings.embed_query(query)
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedded_query,
            limit=5
        )
        documents = "\n".join([result.payload["content"] for result in search_result])
        print(documents)
        return documents if documents else "No relevant documents found."
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return "Error retrieving relevant documents."

# Gemini Model
genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
    "temperature":0.4,
    "top_p":0.9,
    "top_k":50,
    "max_output_tokens":2048,
    "response_mime_type":"text/plain",
}

model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    generation_config=generation_config,
    safety_settings={
        "harassment": "block_none",
        "hate": "block_none",
        "sexual": "block_none",
        "dangerous": "block_none",
    },
    system_instruction="""
    You are an AI legal assistant designed to assist both lawyers and the public in legal case discovery. Your role is to provide the information exactly as it appears in the legal documents, cases, or data. Do not summarize, simplify, or modify any part of the information. Present all details, including names, dates, numeric values, legal terminology, and case facts, exactly as they are provided.

Important guidelines:
- **No summarization**: Do not summarize or paraphrase any content. Provide the information word-for-word as given.
- **No omissions**: Do not omit any details, including names, case facts, dates, and numbers.
- **No added interpretation**: Do not add or infer any new information. Only the exact content that is provided must be relayed to the user.
- **Maintain precision**: If specific names, numbers, or legal conditions are given, present them exactly as they appear. Do not remove any names or alter the text in any way.
- If any information is unclear, inform the user that you cannot provide additional explanations and can only relay the content exactly as it was provided.

Your goal is to deliver the information as accurately as possible, in the exact format it appears, including all names and details.

""",
)

print("Complete")

async def legal_model(query:str,conversation_history:list):
    documents = get_relevant_legal_docs(query)
    if documents == "No relevant documents found." or documents == "Error retrieving relevant documents.":
        return documents
    else:
        input_text = f"Legal Question: {query}\n\nRelavant Documents:\n{documents}"
        conversation_history = [{"role":"user","parts":input_text}]

        response = model.start_chat(history=conversation_history)
        ai_response = response.send_message(query)
        conversation_history.append({"role":"model","parts":ai_response})
        return ai_response.text
