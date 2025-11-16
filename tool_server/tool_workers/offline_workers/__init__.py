# __init__.py
from tool_server.tool_workers.offline_workers.astar import AStarWithPixelCoordinate
from tool_server.tool_workers.offline_workers.draw_path import Draw2DPath
from tool_server.tool_workers.offline_workers.turn_into_text_map import TurnCoordinateIntoTextMap
from tool_server.tool_workers.offline_workers.detect_black_area_experiment import DetectBlackArea
from tool_server.tool_workers.offline_workers.insert_image import InsertImage
from tool_server.tool_workers.offline_workers.get_weather import GetWeather

# 工具实例注册表
offline_tool_instances = {
    "AStarWithPixelCoordinate": AStarWithPixelCoordinate(),
    "Draw2DPath": Draw2DPath(),
    "DetectBlackArea": DetectBlackArea(),
    "InsertImage": InsertImage(),
    "GetWeather": GetWeather(),
    # "TurnCoordinateIntoTextMap": TurnCoordinateIntoTextMap(),
    # 其他工具实例...
}

def get_tool_generate_fn(tool_name):
    """获取工具生成函数"""
    if tool_name in offline_tool_instances:
        return offline_tool_instances[tool_name].generate
    
    return None

def get_tool_instruction(tool_name):
    """
    获取工具的指令说明mei
    
    Args:
        tool_name (str): 工具名称
        
    Returns:
        dict: 工具的instruction字典，如果工具不存在则返回None
    """
    if tool_name in offline_tool_instances:
        return offline_tool_instances[tool_name].get_tool_instruction()
    
    return None

def get_all_tool_instructions():
    """
    获取所有已注册工具的指令说明
    
    Returns:
        dict: 工具名称到instruction的映射
    """
    instructions = {}
    
    for tool_name, tool_instance in offline_tool_instances.items():
        instructions[tool_name] = tool_instance.get_tool_instruction()
    
    return instructions

def get_available_tools():
    """
    获取所有可用工具的名称列表
    
    Returns:
        list: 工具名称列表
    """
    return list(offline_tool_instances.keys())