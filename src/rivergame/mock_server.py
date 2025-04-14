from aiohttp import web
import json
from typing import Dict, Any
from loguru import logger

class MockGameServer:
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.game_state = {
            'player_position': {'x': 0, 'y': 0},
            'platforms': [
                {'x': 100, 'y': 100},
                {'x': 200, 'y': 100},
                {'x': 300, 'y': 100}
            ],
            'game_status': 'ready'
        }
        self._setup_routes()
        
    def _setup_routes(self):
        """设置路由"""
        self.app.router.add_post('/setup', self.handle_setup)
        self.app.router.add_post('/action', self.handle_action)
        self.app.router.add_get('/state', self.handle_get_state)
        
    async def handle_setup(self, request: web.Request) -> web.Response:
        """处理游戏状态设置请求"""
        try:
            data = await request.json()
            state_desc = data.get('state', '')
            
            # 根据描述设置初始状态
            self.game_state['game_status'] = 'ready'
            self.game_state['player_position'] = {'x': 0, 'y': 0}
            
            return web.json_response({
                'success': True,
                'message': '游戏状态已设置',
                'state': self.game_state
            })
        except Exception as e:
            logger.error(f"设置游戏状态失败: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
            
    async def handle_action(self, request: web.Request) -> web.Response:
        """处理游戏动作请求"""
        try:
            data = await request.json()
            action = data.get('action', '')
            
            # 模拟动作执行
            if 'jump' in action.lower():
                self.game_state['player_position']['y'] += 50
            elif 'move' in action.lower():
                self.game_state['player_position']['x'] += 100
                
            return web.json_response({
                'success': True,
                'message': f'执行动作: {action}',
                'state': self.game_state
            })
        except Exception as e:
            logger.error(f"执行动作失败: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
            
    async def handle_get_state(self, request: web.Request) -> web.Response:
        """获取游戏当前状态"""
        return web.json_response(self.game_state)
        
    async def start(self):
        """启动服务器"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"模拟游戏服务器已启动：http://{self.host}:{self.port}")
        return runner
        
    @classmethod
    async def create_and_start(cls, host: str = "localhost", port: int = 8080):
        """创建并启动服务器的工厂方法"""
        server = cls(host, port)
        runner = await server.start()
        return server, runner