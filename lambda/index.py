# fastapi/index.py
import json
import os
import re
from typing import List, Dict, Optional
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import boto3
from botocore.exceptions import ClientError

# FastAPIアプリケーションの初期化
app = FastAPI()

# CORSミドルウェアの追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデルID
MODEL_ID = "https://9d77-34-138-193-2.ngrok-free.app"

# リクエスト用のPydanticモデル
class ChatRequest(BaseModel):
    message: str
    conversationHistory: Optional[List[Dict[str, str]]] = []

# レスポンス用のPydanticモデル
class ChatResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    conversationHistory: Optional[List[Dict[str, str]]] = None
    error: Optional[str] = None

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        message = request.message
        conversation_history = request.conversationHistory or []
        
        print("Processing message:", message)
        print("Using model:", MODEL_ID)
        
        # 会話履歴を使用
        messages = conversation_history.copy()
        
        # ユーザーメッセージを追加
        messages.append({
            "role": "user",
            "content": message
        })
        
        # FastAPIを使用したLLM呼び出し処理
        # ここで外部APIまたはローカルモデルを呼び出します
        async with httpx.AsyncClient() as client:
            # FastAPIで提供されるAPIエンドポイントを呼び出す
            # 例: http://localhost:8000/api/llm
            bedrock_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    bedrock_messages.append({
                        "role": "user",
                        "content": [{"text": msg["content"]}]
                    })
                elif msg["role"] == "assistant":
                    bedrock_messages.append({
                        "role": "assistant", 
                        "content": [{"text": msg["content"]}]
                    })
            
            # LLM APIへのリクエストペイロード
            request_payload = {
                "messages": bedrock_messages,
                "inferenceConfig": {
                    "maxTokens": 512,
                    "stopSequences": [],
                    "temperature": 0.7,
                    "topP": 0.9
                }
            }
            
            print("Calling LLM API with payload:", json.dumps(request_payload))
            
            # FastAPIのLLMエンドポイントを呼び出す
            response = await client.post(
                "http://localhost:8000/api/llm",  # APIエンドポイントを適切に設定
                json=request_payload,
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="LLM API returned an error")
            
            # レスポンスを解析
            response_data = response.json()
            
            # 応答の検証
            if not response_data.get('output') or not response_data['output'].get('message') or not response_data['output']['message'].get('content'):
                raise Exception("No response content from the model")
            
            # アシスタントの応答を取得
            assistant_response = response_data['output']['message']['content'][0]['text']
            
            # アシスタントの応答を会話履歴に追加
            messages.append({
                "role": "assistant",
                "content": assistant_response
            })
            
            # 成功レスポンスの返却
            return ChatResponse(
                success=True,
                response=assistant_response,
                conversationHistory=messages
            )
        
    except Exception as error:
        print("Error:", str(error))
        return ChatResponse(
            success=False,
            error=str(error)
        )

# オプション: 同期APIのサポート（Lambdaの互換性のため）
def lambda_handler(event, context):
    """
    Lambda互換のハンドラー関数
    FastAPIアプリケーションから呼び出すこともできます
    """
    try:
        # リクエストボディの解析
        body = json.loads(event['body'])
        request = ChatRequest(**body)
        
        # FastAPIエンドポイントを同期的に呼び出す
        import asyncio
        response = asyncio.run(chat(request))
        
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps(response.dict())
        }
        
    except Exception as error:
        print("Error:", str(error))
        
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": str(error)
            })
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
