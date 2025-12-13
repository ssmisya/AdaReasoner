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
    
    def generate(self, params, tool_manager = None):
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
            tool_reward = 2
            missing_params_res = self.whether_missing_tool_parameter(params)
            no_missing_params = missing_params_res["no_missing_params"]
            tool_reward += missing_params_res["params_name_match_reward"]
            if not no_missing_params:
                return {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Missing some required parameters. expected keys: {self.instruction['function']['parameters'].get('required', [])}",
                    "error_code": MISSING_REQUIRED_PARAMETER,
                    "tool_reward": tool_reward,
                    "execution_time": time.time() - start_time
                }
            
            params_qualify_res = self.verify_tool_parameter(params)
            params_qualified = params_qualify_res["params_qualified"]
            tool_reward += params_qualify_res["params_qualified_reward"]
            if not params_qualified:
                error_message = params_qualify_res.get("error_info", "Invalid parameters.")
                return {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": error_message,
                    "error_code": INVALID_PARAMETERS,
                    "tool_reward": tool_reward,
                    "execution_time": time.time() - start_time
                }
            else:
                new_params = params_qualify_res["new_params"] 
            
            if tool_manager:
                new_params["tool_manager"] = tool_manager
                
            # 执行工具核心逻辑
            result = self._execute(new_params)
            
            # 添加工具标识
            if isinstance(result, dict) and "tool_response_from" not in result:
                result["tool_response_from"] = self.model_name
            if isinstance(result, dict) and "execution_time" not in result:
                result["execution_time"] = time.time() - start_time
            if isinstance(result, dict) and "tool_reward" not in result:
                result["tool_reward"] = tool_reward
            if isinstance(result, dict) and "error_code" not in result:
                result["error_code"] = SUCCESS
            
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
    
    def whether_missing_tool_parameter(self, params):
        param_keys = set(params.keys())
        required_keys = set(self.instruction["function"]["parameters"].get("required", []))
        parameter_name_match_reward = len(param_keys & required_keys) / len(required_keys)
        no_missing_params = parameter_name_match_reward >= 1
        res = {
            "params_name_match_reward": parameter_name_match_reward,
            "no_missing_params": no_missing_params
        }
        
        return res
    
    def verify_tool_parameter(self, params):
        res = {
            "params_qualified_reward":1,
            "params_qualified": True,
            "new_params":params
        }
        return res
    

    
