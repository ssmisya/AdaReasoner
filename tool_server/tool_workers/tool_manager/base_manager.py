import os
import requests

from ..offline_workers import get_tool_generate_fn, get_available_tools, get_all_tool_instructions
from ..offline_workers import get_tool_instruction as get_offline_tool_instruction
from tool_server.utils.utils import load_json_file
from tool_server.utils.server_utils import build_logger
from tool_server.utils.prompts import (
    one_tool_call_wo_toollist, 
    tool_planning_model_prompt_no_tool_call,
    multi_tool_call_wo_toollist,
    tool_desc_dict
)
from contextlib import contextmanager
import signal
import time

logger = build_logger("tool_manager")
class TimeoutException(Exception): pass
def _timeout_handler(signum, frame):
    raise TimeoutException("chat() timed out.")
signal.signal(signal.SIGALRM, _timeout_handler)

class ToolManager(object):
    def __init__(self, controller_url_location=None, tools=None):
        """
        初始化工具管理器
        
        Args:
            controller_url_location (str, optional): 控制器URL位置
            tools (list, optional): 指定要初始化的工具列表，不指定则初始化全部工具
        """
        self.controller_url_location = controller_url_location
        self.tools = tools  # 保存用户指定的工具列表
        
        self.init_offline_tools(tools)
        self.init_online_tools(self.controller_url_location)
        self.init_online_tool_addr_dict()
        
        logger.info(f"ToolManager is initialized.")
        self.available_tools = self.available_online_tools + self.available_offline_tools
        print("available_tools", self.available_tools)
        self.headers = {"User-Agent": "LLaVA-Plus Client"}
        
    def init_online_tools(self, controller_url_location=None):
        """初始化在线工具"""
        self.available_online_tools = []
        if controller_url_location is None:
            current_file_path = os.path.dirname(os.path.abspath(__file__))
            self.controller_addr_location = f"{current_file_path}/../online_workers/controller_addr/controller_addr.json"
            logger.info("controller_addr is None, using default from controller_addr_location")
        else:
            self.controller_addr_location = controller_url_location
            logger.info(f"controller_addr exsits, controller_url_location is {controller_url_location}")
        
        if os.path.exists(self.controller_addr_location):
            self.controller_addr = load_json_file(self.controller_addr_location)["controller_addr"]
        else:
            self.controller_addr = self.controller_addr_location

        with self.disable_proxy():
            if self.controller_addr is not None and isinstance(self.controller_addr, str):
                session = requests.Session()
                session.trust_env = False
                try:
                    ret = session.post(self.controller_addr + "/list_models", proxies={},timeout=(6000,60000))
                    online_tools = ret.json()["models"]
                    
                    # 如果指定了工具列表，只保留指定的在线工具
                    if self.tools is not None:
                        online_tools = [tool for tool in online_tools if tool in self.tools]
                        
                    logger.info(f"Online Tools: {online_tools}")
                    self.available_online_tools = online_tools
                except Exception as e:
                    logger.error(f"Failed to connect to controller: {e}")
                    self.available_online_tools = []
            
        # 检查常用工具是否可用
        required_tools = []
        if self.tools is None:
            # 如果未指定工具，检查所有常用工具
            required_tools = ["GroundingDINO", "OCR", "SegmentRegionAroundPoint", "Point", 
                             "Crop", "DrawLine", "DrawShape", "HighlightBox", "MaskBox", 
                             "GetSubplotInfo", "GetBarInfo"]
        else:
            # 仅检查指定的常用工具
            required_tools = [tool for tool in self.tools]
                
        miss_tool = [tool for tool in required_tools if tool not in self.available_online_tools]
        if len(miss_tool) == 0:
            logger.info("All required online tools are prepared successfully")
        else:
            logger.info(f"Not all required online tools are prepared successfully, missing: {miss_tool}")        
    
    def init_offline_tools(self, tools=None):
        """
        初始化离线工具
        
        Args:
            tools (list, optional): 指定要初始化的工具列表，不指定则初始化全部工具
        """
        # 获取所有可用的离线工具
        all_available_tools = get_available_tools()
        
        # 如果指定了工具列表，只保留指定的工具
        if tools is not None:
            self.available_offline_tools = [tool for tool in all_available_tools if tool in tools]
        else:
            self.available_offline_tools = all_available_tools
            
        logger.info(f"Offline Tools: {self.available_offline_tools}")
    
    def init_online_tool_addr_dict(self):
        """初始化在线工具地址字典"""
        self.online_tool_addr_dict = {}
        for model_name in self.available_online_tools:
            with self.disable_proxy():
                session = requests.Session()
                session.trust_env = False
                try:
                    ret = session.post(self.controller_addr + "/get_worker_address",
                                      json={"model": model_name}, proxies={})
                    worker_addr = ret.json()["address"]
                    if worker_addr == "":
                        logger.error(f"worker_addr for {model_name} is empty")
                        continue
                    self.online_tool_addr_dict[model_name] = worker_addr
                except Exception as e:
                    logger.error(f"Failed to get worker address for {model_name}: {e}")
    
    def get_online_tool_instruction(self, tool_name):
        """
        从在线工具获取指令说明
        
        Args:
            tool_name (str): 工具名称
            
        Returns:
            str: 工具的指令说明，如果获取失败则返回None
        """
        # 检查缓存
        self.online_tool_instructions = getattr(self, 'online_tool_instructions', {})
        
        # 如果工具地址字典中没有该工具，返回None
        if tool_name not in self.online_tool_addr_dict:
            logger.warning(f"Tool {tool_name} not found in online_tool_addr_dict")
            return None
            
        tool_worker_addr = self.online_tool_addr_dict[tool_name]
        try:
            with self.disable_proxy():
                session = requests.Session()
                session.trust_env = False
                ret = session.post(tool_worker_addr + "/tool_instruction", 
                                   headers=self.headers, proxies={})
                
                if ret.status_code == 200:
                    response_data = ret.json()
                    if response_data.get("status") == "success" and "tool_instruction" in response_data:
                        # 保存到缓存
                        instruction = response_data["tool_instruction"]
                        self.online_tool_instructions[tool_name] = instruction
                        return instruction
                        
            logger.warning(f"Failed to get instruction for online tool {tool_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting instruction for online tool {tool_name}: {e}")
            return None
    
    def get_tool_instructions(self, tools=None):
        """
        获取指定工具的指令说明
        
        Args:
            tools (list, optional): 指定要获取的工具列表，不指定则获取所有已初始化的工具
            
        Returns:
            dict: 工具名称到instruction的映射
        """
        instructions = {}
        
        # 确定要获取指令的工具列表
        tool_list = tools if tools is not None else (self.available_offline_tools + self.available_online_tools)
        
        # 获取工具指令
        for tool_name in tool_list:
            # 获取离线工具指令
            if tool_name in self.available_offline_tools:
                instruction = get_offline_tool_instruction(tool_name)
                if instruction:
                    instructions[tool_name] = instruction
                    
            # 获取在线工具指令
            elif tool_name in self.available_online_tools:
                # 先尝试从在线工具获取指令
                instruction = self.get_online_tool_instruction(tool_name)
                if instruction:
                    instructions[tool_name] = instruction
                # 如果无法从在线工具获取，尝试从预定义字典获取
                elif tool_name in tool_desc_dict:
                    raise ValueError(f"Tool {tool_name} not found in online tools, but exists in tool_desc_dict.")
                    instructions[tool_name] = tool_desc_dict[tool_name]
                    
        return instructions
    
    def get_tool_prompt(self, prompt_type="one_tool_call", tools=None):
        """
        获取带有工具说明的提示语
        
        Args:
            prompt_type (str): 提示语类型，可选 "one_tool_call"、"no_tool_call" 或 "multi_tool_call"
            tools (list, optional): 指定要包含在提示语中的工具列表，不指定则包含所有已初始化的工具
            
        Returns:
            str: 带有工具说明的提示语
        """
        # 获取工具指令
        tool_instructions = self.get_tool_instructions(tools)
        
        # 将工具指令拼接成字符串
        tool_list_str = "\n".join([f"{desc}" for name, desc in tool_instructions.items()])
        
        # 根据prompt类型选择基础提示语
        if prompt_type == "one_tool_call":
            base_prompt = one_tool_call_wo_toollist
        elif prompt_type == "no_tool_call":
            return tool_planning_model_prompt_no_tool_call
        elif prompt_type == "multi_tool_call":
            base_prompt = multi_tool_call_wo_toollist
        else:
            raise ValueError(f"Unsupported prompt type: {prompt_type}")
        
        # 将工具列表插入到提示语中
        prompt = base_prompt.replace("{tool_list}", tool_list_str)
        
        return prompt
    
    def call_tool(self, tool_name, params):
        """调用工具并返回结果"""
        timeout_sec = 6000000  # timeout per attempt
        ret_message = {"text": f"Failed to call tool {tool_name} for unknown reason, ", "error_code": 1}
        try:
            signal.alarm(timeout_sec)
            if tool_name in self.available_offline_tools:
                try:
                    tool_generate_fn = get_tool_generate_fn(tool_name)
                    if tool_generate_fn is None:
                        ret_message = {"text": f"Tool {tool_name} not found.", "error_code": 1}
                    else:
                        ret_message = tool_generate_fn(params)
                except Exception as e:
                    logger.error(f"Failed to call tool {tool_name}: {e}")
                    ret_message = {"text": f"Failed to call tool {tool_name}: {e}", "error_code": 1}
                
            elif tool_name in self.available_online_tools:
                try:
                    tool_worker_addr = self.online_tool_addr_dict[tool_name]
                    with self.disable_proxy():
                        session = requests.Session()
                        session.trust_env = False
                        ret = session.post(tool_worker_addr + "/worker_generate", 
                                          headers=self.headers, json=params, proxies={},
                                          timeout=(3000, 3000))
                    ret_message = ret.json()
                except Exception as e:
                    logger.error(f"Failed to call tool {tool_name}: {e}")
                    ret_message = {"text": f"Failed to call tool {tool_name}: {e}", "error_code": 1}
            else:
                ret_message = {"text": f"Tool {tool_name} not found.", "error_code": 1}
            signal.alarm(0)
        except TimeoutException as te:
            logger.error(f"Timeout calling tool {tool_name}: {te}")
            ret_message = {"text": f"Timeout calling tool {tool_name}: {te}", "error_code": 1}
        finally:
            signal.alarm(0)
            return ret_message
            
    @contextmanager
    def disable_proxy(self):
        """临时禁用代理设置的上下文管理器"""
        # 保存代理环境变量
        old_HTTP = os.environ.get("HTTP_PROXY")
        old_HTTPS = os.environ.get("HTTPS_PROXY")
        old_http = os.environ.get("http_proxy")
        old_https = os.environ.get("https_proxy")
        
        # 移除代理设置
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        
        try:
            yield
        finally:
            # 恢复代理设置
            if old_http is not None:
                os.environ["http_proxy"] = old_http
            if old_https is not None:
                os.environ["https_proxy"] = old_https
            if old_HTTP is not None:
                os.environ["HTTP_PROXY"] = old_HTTP
            if old_HTTPS is not None:
                os.environ["HTTPS_PROXY"] = old_HTTPS