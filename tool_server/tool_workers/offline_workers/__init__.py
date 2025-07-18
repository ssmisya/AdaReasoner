# __init__.py
from tool_server.tool_workers.offline_workers.astar import AStarWithPixelCoordinate

# 工具实例注册表
offline_tool_instances = {
    "AStarWithPixelCoordinate": AStarWithPixelCoordinate(),
    # 其他工具实例...
}

# 向后兼容的工具名称映射
offline_tool_workers = {
    "crop_de": "crop_worker",
    "drawline_de": "drawline_worker",
}

def get_tool_generate_fn(tool_name):
    """获取工具生成函数"""
    # 首先检查是否有新式工具实例
    if tool_name in offline_tool_instances:
        return offline_tool_instances[tool_name].generate
    
    # 向后兼容旧式工具
    if tool_name not in offline_tool_workers:
        return None
    
    module = __import__(f"tool_server.tool_workers.offline_workers.{offline_tool_workers[tool_name]}", fromlist=["generate"])
    return getattr(module, "generate")