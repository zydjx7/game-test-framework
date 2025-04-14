from typing import Dict, Any, Optional
import yaml
import os
from loguru import logger
from dotenv import load_dotenv

class ConfigManager:
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load_config()
    
    def load_config(self, config_path: str = "config.yaml") -> None:
        """加载配置文件和环境变量"""
        try:
            # 加载环境变量
            load_dotenv()
            
            # 从文件加载基本配置
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
                
            # 使用环境变量覆盖配置
            self._config['api']['openai']['api_key'] = os.getenv('OPENAI_API_KEY', self._config['api']['openai']['api_key'])
            self._config['rivergame']['host'] = os.getenv('RIVER_GAME_HOST', self._config['rivergame']['host'])
            self._config['rivergame']['port'] = int(os.getenv('RIVER_GAME_PORT', self._config['rivergame']['port']))
            
            self._setup_logging()
            logger.info("配置加载完成")
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            raise
    
    def _setup_logging(self) -> None:
        """配置日志系统"""
        log_config = self._config.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_file = log_config.get('file', 'logs/app.log')
        
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 配置logger
        logger.add(
            log_file,
            rotation="500 MB",
            retention="10 days",
            level=log_level
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        try:
            keys = key.split('.')
            value = self._config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def validate_config(self) -> bool:
        """验证配置是否完整"""
        required_keys = [
            'api.openai.api_key',
            'api.openai.model',
            'rivergame.host',
            'rivergame.port'
        ]
        
        try:
            for key in required_keys:
                if self.get(key) is None:
                    logger.error(f"缺少必要的配置项: {key}")
                    return False
            return True
        except Exception as e:
            logger.error(f"验证配置失败: {e}")
            return False