from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.chat.model_manager import ModelManager
import asyncio
from sse_starlette.sse import EventSourceResponse
import time

app = FastAPI(title="Model Inference API")
model_manager = ModelManager()

class InferenceRequest(BaseModel):
    prompt: str
    model_name: str
    stream: bool = True
    
class InferenceResponse(BaseModel):
    text: str
    model_name: str
    time_taken: float
    tokens: int
    tokens_per_second: float

@app.get("/models")
async def list_models():
    """获取所有可用模型列表"""
    return {
        "models": list(model_manager.get_available_models())
    }

@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """非流式推理接口"""
    if request.stream:
        raise HTTPException(status_code=400, detail="此接口不支持流式输出，请使用 /infer/stream")
        
    try:
        start_time = time.time()
        text = ""
        token_count = 0
        
        for output in model_manager.generate_stream(request.prompt, request.model_name):
            if not output.finished:
                text += output.text
                token_count += 1
                
        time_taken = time.time() - start_time
        
        return InferenceResponse(
            text=text,
            model_name=request.model_name,
            time_taken=time_taken,
            tokens=token_count,
            tokens_per_second=token_count/time_taken if time_taken > 0 else 0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer/stream")
async def infer_stream(request: InferenceRequest):
    """流式推理接口"""
    if not request.stream:
        raise HTTPException(status_code=400, detail="此接口仅支持流式输出，请使用 /infer")
        
    async def generate():
        try:
            for output in model_manager.generate_stream(request.prompt, request.model_name):
                if output.finished:
                    break
                yield {
                    "data": output.text
                }
        except Exception as e:
            yield {
                "error": str(e)
            }
    
    return EventSourceResponse(generate())

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True) 