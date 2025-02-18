import sys
import uvicorn
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import json
import uuid
import time
import asyncio
from fastapi.responses import StreamingResponse
import logging
import subprocess
import shlex

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,  # 设置为DEBUG级别以查看所有日志
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OllamaLLM:
    def __init__(self):
        self.base_url = "http://localhost:11434/v1"
        self.default_model = "deepseek-r1:1.5b"
        self.cheat_model = "gpt-4o"
        
    async def forward_request(self, request_data: dict) -> dict:
        """
        转发请求到 Ollama 服务，支持流式响应
        """
        client_ip = request_data.pop('client_ip', 'unknown')
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"开始处理请求 [ID: {request_id}] 来自 {client_ip}")
        
        try:
            # 保存原始 model 值和流式设置
            original_model = request_data.get('model', '')
            stream_mode = request_data.get('stream', False)
            
            # 清理和准备请求数据
            request_data_copy = {
                'model': self.default_model,
                'messages': request_data.get('messages', []),
                'stream': stream_mode,
                'temperature': request_data.get('temperature', 0.7),
                'top_p': request_data.get('top_p', 1.0),
                'max_tokens': request_data.get('max_tokens', 2000)
            }
            
            # 准备curl命令
            curl_cmd = [
                'curl',
                '-N',  # 禁用缓冲
                '-X', 'POST',
                f"{self.base_url}/chat/completions",
                '-H', 'Content-Type: application/json',
                '-d', json.dumps(request_data_copy)
            ]
            
            if stream_mode:
                # 流式模式
                async def generate():
                    logger.info(f"开始流式传输 [ID: {request_id}]")
                    process = await asyncio.create_subprocess_exec(
                        *curl_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    buffer = ""
                    chunk_count = 0
                    
                    try:
                        while True:
                            chunk = await process.stdout.read(1024)
                            if not chunk:
                                break
                                
                            chunk_count += 1
                            chunk_text = chunk.decode('utf-8')
                            logger.debug(f"收到数据块 {chunk_count} [ID: {request_id}]: {chunk_text}")
                            buffer += chunk_text
                            
                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                line = line.strip()
                                if not line:
                                    continue
                                if line.startswith('data: '):
                                    line = line[6:]
                                if line == '[DONE]':
                                    logger.info(f"流式传输完成 [ID: {request_id}], 总共处理 {chunk_count} 个数据块")
                                    yield 'data: [DONE]\n\n'
                                    return
                                    
                                try:
                                    chunk_data = json.loads(line)
                                    if original_model and 'model' in chunk_data:
                                        chunk_data['model'] = self.cheat_model
                                        
                                    output_line = f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                                    logger.debug(f"发送数据 [ID: {request_id}]: {output_line}")
                                    yield output_line
                                except json.JSONDecodeError as e:
                                    logger.error(f"JSON解析错误 [ID: {request_id}]: {str(e)}\n响应内容: {line}")
                                    continue
                                
                    except asyncio.CancelledError:
                        logger.warning(f"流式响应被取消 [ID: {request_id}]")
                        process.terminate()
                        return
                    except Exception as e:
                        logger.error(f"流式响应错误 [ID: {request_id}]: {str(e)}")
                        process.terminate()
                        return
                    finally:
                        if process.returncode is None:
                            process.terminate()
                        logger.info(f"关闭流式响应 [ID: {request_id}]")
                        
                return generate()
            else:
                # 非流式模式
                process = await asyncio.create_subprocess_exec(
                    *curl_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    error_msg = stderr.decode('utf-8')
                    logger.error(f"请求失败 [ID: {request_id}]: {error_msg}")
                    raise HTTPException(
                        status_code=503,
                        detail=f"请求失败: {error_msg}"
                    )
                    
                response_data = json.loads(stdout.decode('utf-8'))
                if original_model and 'model' in response_data:
                    response_data['model'] = self.cheat_model
                    
                logger.info(f"完成非流式请求 [ID: {request_id}]")
                return response_data
                
        except Exception as e:
            logger.error(f"处理请求错误 [ID: {request_id}]: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"内部服务器错误: {str(e)}"
            )

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        获取可用的模型列表
        """
        try:
            # 固定返回 gpt-4o 模型
            return [{
                "id": self.cheat_model,
                "object": "model",
                "created": 1677610602,
                "owned_by": "ollama",
                "permission": [],
                "root": "ollama",
                "parent": None,
            }]
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
    """
    处理聊天完成请求
    """
    try:
        request_data = await request.json()
        # 添加客户端IP信息用于日志追踪
        request_data['client_ip'] = request.client.host
        logger.info(f"收到请求: {json.dumps(request_data, ensure_ascii=False)}")
        
        result = await llm.forward_request(request_data)
        
        # 检查是否为流式请求
        if request_data.get('stream', False):
            logger.info("创建流式响应...")
            response = StreamingResponse(
                result,
                media_type='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache, no-transform',
                    'Connection': 'keep-alive',
                    'Content-Type': 'text/event-stream',
                    'X-Accel-Buffering': 'no'
                }
            )
            logger.info("返回流式响应对象")
            return response
            
        logger.info(f"返回非流式响应: {json.dumps(result, ensure_ascii=False)}")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析错误: {str(e)}")
        raise HTTPException(status_code=400, detail="无效的 JSON 数据")
    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
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