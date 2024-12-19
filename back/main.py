# app/main.py
from fastapi import FastAPI, Request
from training_answer import handle_chat_pt_request  # 요청 처리 모듈 임포트
from diet_answer import handle_chat_diet_request  # 요청 처리 모듈 임포트

app = FastAPI()

@app.post("/chat-pt")
async def chat(request: Request):
    try:
        # 채팅 요청 처리
        return await handle_chat_pt_request(request)
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat-diet")
async def chat(request: Request):
    try:
        # 채팅 요청 처리
        return await handle_chat_diet_request(request)
    except Exception as e:
        return {"error": str(e)}