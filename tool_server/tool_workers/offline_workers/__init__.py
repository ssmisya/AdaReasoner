
offline_tool_workers = {
    "crop_de":"crop_worker",
    "drawline_de":"drawline_worker",
}

def get_tool_generate_fn(tool_name):
    if tool_name not in offline_tool_workers:
        return None
    module = __import__(f"tool_server.tool_workers.offline_workers.{offline_tool_workers[tool_name]}", fromlist=["generate"])
    return getattr(module, "generate")