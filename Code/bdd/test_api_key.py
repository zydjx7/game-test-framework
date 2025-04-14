import os
from dotenv import load_dotenv
from openai import OpenAI
import sys
from loguru import logger

# 设置日志
logger.add("logs/api_test.log", level="DEBUG", rotation="1 MB")

def test_api_key():
    """测试API密钥是否有效"""
    # 加载环境变量
    load_dotenv()
    
    # 获取API密钥
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("错误：未找到API密钥，请确保.env文件中设置了OPENAI_API_KEY")
        return False
    
    # 获取模型名称（默认为gpt-4o）
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    # 创建客户端
    client = OpenAI(api_key=api_key)
    
    # 测试API调用
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        logger.info(f"API密钥设置成功！使用模型:{model_name}")
        logger.info(f"收到回复: {response.choices[0].message.content}")
        return True
    except Exception as e:
        logger.error(f"API调用失败: {e}")
        return False

if __name__ == "__main__":
    success = test_api_key()
    if success:
        print("✅ API密钥验证成功")
    else:
        print("❌ API密钥验证失败")