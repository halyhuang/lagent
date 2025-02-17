import sys
from typing import Optional, Union, List, Dict
import time
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import requests

class Message(BaseModel):
    role: str = Field(..., description="消息角色：system, user, 或 assistant")
    content: str = Field(..., description="消息内容")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="模型名称")
    messages: List[Message] = Field(..., description="消息历史")
    temperature: float = Field(0.7, description="温度参数", ge=0.0, le=2.0)
    stream: bool = Field(False, description="是否使用流式响应")

class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="响应ID")
    object: str = "chat.completion"
    created: int = Field(..., description="创建时间戳")
    model: str = Field(..., description="使用的模型")
    choices: List[Dict] = Field(..., description="响应选项")
    usage: Dict = Field(..., description="token 使用统计")

class OllamaLLM:
    def __init__(self, model_name="deepseek-r1:1.5b", temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = "http://localhost:11434/api"
        
    def chat(self, messages: List[Message]) -> str:
        """
        处理聊天请求
        :param messages: 消息列表
        :return: 模型的回复
        """
        try:
            # 构建 prompt
            prompt = ""
            for msg in messages:
                if msg.role == "system":
                    prompt += f"System: {msg.content}\n"
                elif msg.role == "assistant":
                    prompt += f"Assistant: {msg.content}\n"
                else:
                    prompt += f"Human: {msg.content}\n"
            prompt += "Assistant: "

            # 调用 Ollama API
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"]
            
        except requests.exceptions.RequestException as e:
            print(f"Ollama API 调用失败: {str(e)}", file=sys.stderr)
            raise HTTPException(status_code=503, detail="Ollama 服务不可用")
        except Exception as e:
            print(f"处理请求时出错: {str(e)}", file=sys.stderr)
            raise HTTPException(status_code=500, detail="内部服务器错误")

# 创建 FastAPI 应用
app = FastAPI(
    title="OpenAI Compatible Chat Service",
    description="OpenAI 兼容的聊天服务",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该限制
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

# 创建全局 LLM 实例
llm = OllamaLLM(model_name="deepseek-r1:1.5b", temperature=0.7)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    OpenAI 兼容的聊天补全接口
    """
    try:
        # 获取模型响应
        response_text = llm.chat(request.messages)
        
        # 构建 OpenAI 格式的响应
        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": -1,  # 实际应用中需要实现 token 计数
                "completion_tokens": -1,
                "total_tokens": -1
            }
        )
        return JSONResponse(content=response.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """
    列出可用模型
    """
    return JSONResponse(content={
        "data": [{
            "id": llm.model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "ollama"
        }]
    })

@app.get("/")
async def root():
    """
    服务健康检查接口
    """
    return JSONResponse(content={
        "status": "running",
        "model": llm.model_name,
        "api_compatibility": "openai",
        "version": "1.0.0"
    })

@app.get("/favicon.ico")
async def favicon():
    """
    处理 favicon 请求
    """
    return Response(status_code=204)

def main():
    # 启动服务器
    port = 8000  # 服务监听端口
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