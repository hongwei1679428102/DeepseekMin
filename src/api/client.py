import requests
import sseclient
import json
from typing import Optional

class InferenceClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        
    def list_models(self):
        """获取可用模型列表"""
        response = requests.get(f"{self.base_url}/models")
        return response.json()
        
    def infer(self, prompt: str, model_name: str, stream: bool = True):
        """进行推理"""
        if stream:
            return self.infer_stream(prompt, model_name)
        else:
            return self.infer_sync(prompt, model_name)
            
    def infer_sync(self, prompt: str, model_name: str):
        """同步推理"""
        response = requests.post(
            f"{self.base_url}/infer",
            json={
                "prompt": prompt,
                "model_name": model_name,
                "stream": False
            }
        )
        return response.json()
        
    def infer_stream(self, prompt: str, model_name: str):
        """流式推理"""
        response = requests.post(
            f"{self.base_url}/infer/stream",
            json={
                "prompt": prompt,
                "model_name": model_name,
                "stream": True
            },
            stream=True
        )
        
        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.data:
                try:
                    data = json.loads(event.data)
                    if "error" in data:
                        raise Exception(data["error"])
                    yield data
                except json.JSONDecodeError:
                    yield {"data": event.data} 