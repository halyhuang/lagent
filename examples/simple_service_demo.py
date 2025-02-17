import sys
from typing import Optional, Union, List, Dict
import requests

class OllamaLLM:
    def __init__(self, model_name="deepseek-r1:1.5b", temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = "http://localhost:11434/api"
        
    def chat(self, messages: Union[str, List[Dict[str, str]]]) -> str:
        """
        处理聊天请求
        :param messages: 可以是字符串或消息列表
        :return: 模型的回复
        """
        try:
            # 构建 prompt
            if isinstance(messages, str):
                prompt = messages
            else:
                # 将消息列表转换为适合 deepseek 的格式
                prompt = ""
                for msg in messages:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    if role == 'system':
                        prompt += f"System: {content}\n"
                    elif role == 'assistant':
                        prompt += f"Assistant: {content}\n"
                    else:
                        prompt += f"Human: {content}\n"
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
            return "抱歉，我现在无法回答。请确保 Ollama 服务正在运行。"
        except Exception as e:
            print(f"处理请求时出错: {str(e)}", file=sys.stderr)
            return "抱歉，处理您的请求时出现错误。"

def main():
    # 创建 Ollama 客户端
    llm = OllamaLLM(model_name="deepseek-r1:1.5b", temperature=0.7)
    
    # 测试消息
    messages = [
        "你好，请介绍一下你自己",
        "请帮我写一个简单的Python函数，实现两个数相加",
    ]
    
    # 发送消息并获取回复
    for msg in messages:
        print(f"\n发送消息: {msg}")
        response = llm.chat(msg)
        print(f"收到回复: {response}")

if __name__ == '__main__':
    main() 