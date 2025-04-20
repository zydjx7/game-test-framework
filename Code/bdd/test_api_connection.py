import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from loguru import logger
import codecs

# 设置日志
logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("logs/api_test.log", level="DEBUG", rotation="1 MB")

# DeepSeek API基础URL
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# 默认模型
DEFAULT_MODEL = "deepseek-chat"

def load_env_safe():
    """安全加载环境变量，处理不同编码"""
    try:
        # 尝试使用常规方式加载
        logger.info("尝试使用默认编码加载.env文件")
        load_dotenv()
        return True
    except UnicodeDecodeError as e:
        logger.warning(f"加载.env文件失败(UTF-8): {e}")
        
        # 查找.env文件
        dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
        if not os.path.exists(dotenv_path):
            logger.error(f".env文件不存在: {dotenv_path}")
            return False
            
        # 尝试不同的编码
        encodings = ['utf-16', 'utf-16-le', 'utf-16-be', 'latin1', 'cp1252', 'gbk']
        for encoding in encodings:
            try:
                logger.info(f"尝试使用 {encoding} 编码读取.env文件")
                with codecs.open(dotenv_path, 'r', encoding=encoding) as file:
                    content = file.read()
                    
                # 解析键值对并设置环境变量
                for line in content.splitlines():
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
                logger.info(f"成功使用 {encoding} 编码加载.env文件")
                return True
            except Exception as e:
                logger.debug(f"{encoding} 编码尝试失败: {e}")
                
        logger.error("所有编码尝试都失败了")
        return False

def test_connection():
    """测试API连接"""
    # 加载环境变量
    load_env_safe()
    
    # 获取API配置
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    base_url = os.getenv("OPENAI_BASE_URL", "")
    api_type = os.getenv("API_TYPE", "deepseek")
    
    logger.info(f"开始测试API连接，API类型: {api_type}，使用模型: {model_name}")
    
    if not api_key:
        logger.error("未找到API密钥环境变量")
        return False
        
    # 创建客户端
    client_args = {"api_key": api_key}
    if base_url:
        client_args["base_url"] = base_url
        logger.info(f"使用API基础URL: {base_url}")
    
    client = OpenAI(**client_args)
    
    try:
        # 测试API调用
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个游戏测试助手"},
                {"role": "user", "content": "返回'API连接成功'"}
            ],
            max_tokens=10
        )
        logger.info(f"API调用成功！使用{api_type} API，模型: {model_name}")
        logger.info(f"响应内容: {response.choices[0].message.content}")
        return True
    except Exception as e:
        logger.error(f"API调用失败: {str(e)}")
        return False

if __name__ == "__main__":
    # 安全加载环境变量
    if not load_env_safe():
        logger.warning("从命令行输入API密钥")
        os.environ["OPENAI_API_KEY"] = input("请输入你的API密钥: ")
        api_type = input("请输入API类型(openai或deepseek，默认deepseek): ") or "deepseek"
        os.environ["API_TYPE"] = api_type
        
        if api_type.lower() == "deepseek":
            os.environ["OPENAI_BASE_URL"] = DEEPSEEK_BASE_URL
            os.environ["OPENAI_MODEL"] = input(f"请输入模型名称(默认{DEFAULT_MODEL}): ") or DEFAULT_MODEL
        else:
            os.environ["OPENAI_BASE_URL"] = ""
            os.environ["OPENAI_MODEL"] = input("请输入模型名称(默认gpt-4o): ") or "gpt-4o"
    
    success = test_connection()
    if success:
        print("✅ API连接测试成功")
    else:
        print("❌ API连接测试失败")