from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from model.legal_model import legal_model
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a request model
class UserMessage(BaseModel):
    user_id: str
    message: str

conversation_history = []

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat/{user_id}")
async def legal_assistance(user_id:str, user_message:UserMessage):
    global conversation_history

    user_input = user_message.message

    try:
        response = await legal_model(user_input,conversation_history)

        return JSONResponse(content={"response": response}, status_code=200)
    except Exception as e:
        print(e)
        raise JSONResponse(content={"response":"Error occured"}, status_code=400)
