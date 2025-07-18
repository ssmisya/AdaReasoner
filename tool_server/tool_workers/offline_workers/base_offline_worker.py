# base_offline_worker.py
import traceback
import time
import json
from abc import ABC, abstractmethod

from tool_server.utils.server_utils import build_logger
from tool_server.utils.error_codes import *

logger = build_logger("base_offline_worker")

class BaseOfflineWorker(ABC):
    """
    基础离线工具工作类
    提供与在线工具一致的接口，但直接以函数调用方式实现
    """
    def __init__(self, model_name=None):
        """
        初始化离线工具
        
        Args:
            model_name (str): 工具名称，如未指定则使用类名
        """
        self.model_name = model_name if model_name else self.__class__.__name__
        self.instruction = self._get_default_instruction()
        logger.info(f"Initialized offline worker: {self.model_name}")
        
    def _get_default_instruction(self):
        """
        返回默认工具说明，子类应重写此方法
        """
        return {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Base offline tool. This should be overridden by subclasses.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    
    def generate(self, params):
        """
        工具生成函数，与在线工具接口一致
        
        Args:
            params (dict): 工具参数
            
        Returns:
            dict: 工具执行结果
        """
        tool_reward = 2.0
        start_time = time.time()
        
        try:
            # 参数验证
            param_keys = set(params.keys())
            required_keys = set(self.instruction["function"]["parameters"].get("required", []))
            
            if required_keys:
                parameter_name_match_reward = len(param_keys & required_keys) / len(required_keys | param_keys)
                tool_reward += parameter_name_match_reward
                
                # 参数名称不完全匹配，直接返回错误
                if parameter_name_match_reward < 1:
                    return {
                        "tool_response_from": self.model_name,
                        "status": "failed",
                        "message": f"Invalid parameters. Expected: {required_keys}, got: {param_keys}",
                        "error_code": INVALID_PARAMETERS,
                        "tool_reward": tool_reward,
                        "execution_time": time.time() - start_time
                    }
            
            # 执行工具核心逻辑
            result = self._execute(params)
            
            # 添加工具标识
            if isinstance(result, dict) and "tool_response_from" not in result:
                result["tool_response_from"] = self.model_name
            if isinstance(result, dict) and "execution_time" not in result:
                result["execution_time"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error in {self.model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"Tool execution failed: {str(e)}",
                "error_code": TOOL_RUN_FAILED,
                "tool_reward": tool_reward,
                "execution_time": time.time() - start_time
            }
    
    @abstractmethod
    def _execute(self, params):
        """
        工具核心执行逻辑，需要被子类实现
        
        Args:
            params (dict): 工具参数
            
        Returns:
            dict: 执行结果
        """
        pass
    
    def get_tool_instruction(self):
        """返回工具说明"""
        return self.instruction