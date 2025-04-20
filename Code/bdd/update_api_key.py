#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
更新API密钥设置（支持OpenAI和DeepSeek）
"""

import os
import argparse
import sys
import codecs
from dotenv import load_dotenv, find_dotenv, set_key
from openai import OpenAI
from loguru import logger

# 设置日志
logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("logs/api_update.log", level="DEBUG", rotation="1 MB")

# DeepSeek API基础URL
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# 默认模型
DEFAULT_MODEL = "deepseek-chat"

def load_env_safe():
    """安全加载.env文件，处理不同编码格式"""
    try:
        # 尝试使用默认编码加载
        logger.info("尝试使用默认编码加载.env文件")
        load_dotenv()
        return True
    except UnicodeDecodeError as e:
        logger.warning(f"加载.env文件失败(UTF-8): {e}")
        
        # 查找.env文件
        dotenv_path = find_dotenv()
        if not dotenv_path:
            logger.error(f".env文件不存在")
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

def update_api_key(api_key, api_type="deepseek"):
    """更新.env文件中的API密钥
    @param api_key: API密钥
    @param api_type: API类型，可选值：'openai'或'deepseek'，默认为'deepseek'
    """
    # 首先尝试找到现有的.env文件
    dotenv_path = find_dotenv()
    
    # 检查文件是否存在及其编码
    current_encoding = "utf-8"  # 默认编码
    if dotenv_path and os.path.exists(dotenv_path):
        # 尝试检测文件编码
        encodings = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin1', 'cp1252', 'gbk']
        for encoding in encodings:
            try:
                with codecs.open(dotenv_path, 'r', encoding=encoding) as f:
                    f.read()
                current_encoding = encoding
                logger.info(f"检测到.env文件编码为: {encoding}")
                break
            except UnicodeDecodeError:
                continue
    
    if not dotenv_path:
        # 如果.env文件不存在，创建新文件
        dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
        with open(dotenv_path, 'w', encoding='utf-8') as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")
            f.write(f"OPENAI_BASE_URL={DEEPSEEK_BASE_URL if api_type == 'deepseek' else ''}\n")
            f.write(f"OPENAI_MODEL={DEFAULT_MODEL if api_type == 'deepseek' else 'gpt-4o'}\n")
            f.write(f"API_TYPE={api_type}\n")
        logger.info(f"创建了新的.env文件(UTF-8): {dotenv_path}")
    else:
        # 如果文件存在，以检测到的编码读取内容
        try:
            with codecs.open(dotenv_path, 'r', encoding=current_encoding) as f:
                content = f.read()
                
            # 解析并更新内容
            lines = content.splitlines()
            updated_content = []
            api_key_found = False
            model_found = False
            base_url_found = False
            api_type_found = False
            
            for line in lines:
                if line.startswith("OPENAI_API_KEY="):
                    updated_content.append(f"OPENAI_API_KEY={api_key}")
                    api_key_found = True
                elif line.startswith("OPENAI_MODEL="):
                    updated_content.append(f"OPENAI_MODEL={DEFAULT_MODEL if api_type == 'deepseek' else 'gpt-4o'}")
                    model_found = True
                elif line.startswith("OPENAI_BASE_URL="):
                    updated_content.append(f"OPENAI_BASE_URL={DEEPSEEK_BASE_URL if api_type == 'deepseek' else ''}")
                    base_url_found = True
                elif line.startswith("API_TYPE="):
                    updated_content.append(f"API_TYPE={api_type}")
                    api_type_found = True
                else:
                    updated_content.append(line)
            
            # 如果没找到相关配置，添加它们
            if not api_key_found:
                updated_content.append(f"OPENAI_API_KEY={api_key}")
            if not model_found:
                updated_content.append(f"OPENAI_MODEL={DEFAULT_MODEL if api_type == 'deepseek' else 'gpt-4o'}")
            if not base_url_found:
                updated_content.append(f"OPENAI_BASE_URL={DEEPSEEK_BASE_URL if api_type == 'deepseek' else ''}")
            if not api_type_found:
                updated_content.append(f"API_TYPE={api_type}")
            
            # 写回文件，保持原有编码
            with codecs.open(dotenv_path, 'w', encoding=current_encoding) as f:
                f.write("\n".join(updated_content))
            
            logger.info(f"更新了API密钥({api_type})，保持原编码({current_encoding}): {dotenv_path}")
        except Exception as e:
            logger.error(f"更新API密钥失败: {e}")
            return False
    
    # 重新加载环境变量
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_MODEL"] = DEFAULT_MODEL if api_type == 'deepseek' else 'gpt-4o'
    os.environ["OPENAI_BASE_URL"] = DEEPSEEK_BASE_URL if api_type == 'deepseek' else ''
    os.environ["API_TYPE"] = api_type
    
    return True

def test_api_key():
    """测试API密钥是否有效"""
    # 安全加载环境变量
    load_env_safe()
    
    # 获取API密钥和模型名称
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    base_url = os.getenv("OPENAI_BASE_URL", "")
    api_type = os.getenv("API_TYPE", "deepseek")
    
    if not api_key:
        logger.error("错误：未找到API密钥，请确保.env文件中设置了OPENAI_API_KEY")
        return False
    
    # 创建客户端
    client_args = {"api_key": api_key}
    if base_url:
        client_args["base_url"] = base_url
        
    client = OpenAI(**client_args)
    
    # 测试API调用
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=5
        )
        logger.info(f"API密钥设置成功！使用{api_type} API，模型: {model_name}")
        logger.info(f"收到回复: {response.choices[0].message.content}")
        return True
    except Exception as e:
        logger.error(f"API调用失败: {e}")
        return False

def update_model(model_name):
    """更新.env文件中的模型名称"""
    # 获取.env文件路径和编码
    dotenv_path = find_dotenv()
    if not dotenv_path:
        logger.error("找不到.env文件，请先设置API密钥")
        return False
    
    # 尝试检测文件编码
    current_encoding = "utf-8"  # 默认编码
    encodings = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin1', 'cp1252', 'gbk']
    for encoding in encodings:
        try:
            with codecs.open(dotenv_path, 'r', encoding=encoding) as f:
                content = f.read()
            current_encoding = encoding
            logger.info(f"检测到.env文件编码为: {encoding}")
            
            # 解析并更新内容
            lines = content.splitlines()
            updated_content = []
            model_found = False
            
            for line in lines:
                if line.startswith("OPENAI_MODEL="):
                    updated_content.append(f"OPENAI_MODEL={model_name}")
                    model_found = True
                else:
                    updated_content.append(line)
            
            # 如果没找到模型配置，添加它
            if not model_found:
                updated_content.append(f"OPENAI_MODEL={model_name}")
            
            # 写回文件，保持原有编码
            with codecs.open(dotenv_path, 'w', encoding=current_encoding) as f:
                f.write("\n".join(updated_content))
            
            logger.info(f"更新了模型名称为 {model_name}，保持原编码({current_encoding})")
            
            # 设置环境变量
            os.environ["OPENAI_MODEL"] = model_name
            
            return True
        except UnicodeDecodeError:
            continue
    
    logger.error("无法检测.env文件编码，更新模型失败")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="更新API设置")
    parser.add_argument("--key", help="新的API密钥")
    parser.add_argument("--model", help="指定使用的模型，例如deepseek-chat, deepseek-reasoner等")
    parser.add_argument("--api-type", choices=["openai", "deepseek"], default="deepseek", help="指定API类型，默认为deepseek")
    parser.add_argument("--test", action="store_true", help="测试API密钥是否有效")
    parser.add_argument("--create-utf8", action="store_true", help="创建一个新的UTF-8编码的.env文件")
    
    args = parser.parse_args()
    
    # 创建UTF-8编码的.env文件
    if args.create_utf8:
        try:
            # 先加载现有环境变量
            load_env_safe()
            api_key = os.getenv("OPENAI_API_KEY", "")
            model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
            base_url = os.getenv("OPENAI_BASE_URL", DEEPSEEK_BASE_URL)
            api_type = os.getenv("API_TYPE", "deepseek")
            
            # 创建新文件
            dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
            with open(dotenv_path, 'w', encoding='utf-8') as f:
                f.write(f"OPENAI_API_KEY={api_key}\n")
                f.write(f"OPENAI_MODEL={model}\n")
                f.write(f"OPENAI_BASE_URL={base_url}\n")
                f.write(f"API_TYPE={api_type}\n")
            logger.info(f"已创建UTF-8编码的.env文件: {dotenv_path}")
        except Exception as e:
            logger.error(f"创建UTF-8编码的.env文件失败: {e}")
    
    # 更新API密钥
    if args.key and update_api_key(args.key, args.api_type):
        logger.info(f"API密钥已成功更新，API类型: {args.api_type}")
    
    # 更新模型名称
    if args.model and update_model(args.model):
        logger.info(f"模型已更新为: {args.model}")
    
    # 如果要求测试API密钥
    if args.test or (not args.key and not args.model and not args.create_utf8):
        if test_api_key():
            logger.info("✅ API密钥测试通过")
        else:
            logger.error("❌ API密钥测试失败")