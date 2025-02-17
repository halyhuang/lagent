import sys
import uvicorn
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
from typing import List, Dict, Any

class OllamaLLM:
    def __init__(self):
        self.base_url = "http://localhost:11434/v1"
        self.default_model = "gpt-4o"
        self.available_models = [{
            "id": "gpt-4o",
            "object": "model",
            "created": 1677610602,
            "owned_by": "ollama",
            "permission": [],
            "root": "ollama",
            "parent": None,
        }]
        
    async def forward_request(self, request_data: dict) -> dict:
        """
        转发请求到 Ollama 服务
        """
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=request_data
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Ollama API 调用失败: {str(e)}", file=sys.stderr)
            raise HTTPException(status_code=503, detail="Ollama 服务不可用")
        except Exception as e:
            print(f"处理请求时出错: {str(e)}", file=sys.stderr)
            raise HTTPException(status_code=500, detail="内部服务器错误")

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        获取可用的模型列表
        """
        try:
            # 直接返回预定义的模型列表
            return self.available_models
        except Exception as e:
            print(f"处理请求时出错: {str(e)}", file=sys.stderr)
            raise HTTPException(status_code=500, detail="内部服务器错误")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = OllamaLLM()

@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    try:
        request_data = await request.json()
        result = await llm.forward_request(request_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.get("/v1/models")
async def list_models():
    """
    获取所有可用的模型列表
    """
    try:
        models = await llm.list_models()
        return {
            "object": "list",
            "data": models
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    port = 8000
    print(f"启动 OpenAI 兼容服务在端口 {port}...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )

if __name__ == '__main__':
    main() 