#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试工具管理器的自定义初始化和获取工具说明功能
"""

import os
from PIL import Image, ImageDraw
from tool_server.tool_workers.tool_manager.base_manager import ToolManager
from tool_server.utils.utils import pil_to_base64, base64_to_pil

def test_default_initialization():
    """测试默认初始化（全部工具）"""
    print("\n=== 测试默认初始化（全部工具）===")
    
    # 默认初始化（所有工具）
    full_manager = ToolManager()
    
    print(f"可用的离线工具: {full_manager.available_offline_tools}")
    print(f"可用的在线工具: {full_manager.available_online_tools}")
    print(f"所有可用工具: {full_manager.available_tools}")
    
    return full_manager

def test_custom_initialization():
    """测试自定义初始化（仅特定工具）"""
    print("\n=== 测试自定义初始化（仅特定工具）===")
    
    # 自定义初始化（仅特定工具）
    custom_manager = ToolManager(tools=["AStarWithPixelCoordinate", "Draw2DPath", "OCR"])
    
    print(f"可用的离线工具: {custom_manager.available_offline_tools}")
    print(f"可用的在线工具: {custom_manager.available_online_tools}")
    print(f"所有可用工具: {custom_manager.available_tools}")
    
    return custom_manager

def test_get_tool_instructions(manager):
    """测试获取工具说明"""
    print("\n=== 测试获取工具说明 ===")
    
    # 获取所有已初始化工具的说明
    all_instructions = manager.get_tool_instructions()
    print(f"全部工具的说明数量: {len(all_instructions)}")
    
    # 显示特定工具的说明
    astar_instruction = manager.get_tool_instructions(tools=["AStarWithPixelCoordinate"])
    print("--- A*寻路工具说明 ---")
    print(astar_instruction.get("AStarWithPixelCoordinate", "未找到说明"))
    
    # 测试获取多个特定工具的说明
    specific_tools = ["AStarWithPixelCoordinate", "Draw2DPath"]
    specific_instructions = manager.get_tool_instructions(tools=specific_tools)
    
    print(f"\n获取特定工具说明 ({', '.join(specific_tools)}):")
    for tool_name, instruction in specific_instructions.items():
        print(f"- {tool_name}: {instruction[:100]}..." if len(instruction) > 100 else f"- {tool_name}: {instruction}")
    
    return all_instructions

def test_get_tool_prompt(manager):
    """测试获取工具提示语"""
    print("\n=== 测试获取工具提示语 ===")
    
    # 获取特定工具的单工具调用提示语
    astar_draw_prompt = manager.get_tool_prompt(
        prompt_type="one_tool_call", 
        tools=["AStarWithPixelCoordinate", "Draw2DPath"]
    )
    
    # 显示提示语片段
    print("--- 单工具调用提示语片段 ---")
    prompt_preview = astar_draw_prompt[:300] + "..." if len(astar_draw_prompt) > 300 else astar_draw_prompt
    print(prompt_preview)
    
    # 获取无工具调用提示语
    no_tool_prompt = manager.get_tool_prompt(prompt_type="no_tool_call")
    print("\n--- 无工具调用提示语 ---")
    print(no_tool_prompt)
    
    return astar_draw_prompt

def test_tool_call(manager):
    """测试调用工具"""
    print("\n=== 测试调用工具 ===")
    
    # 测试调用A*寻路工具
    astar_params = {
        "start": [0, 0],
        "goal": [2, 0],
        "obstacles": [[1, 0]]
    }
    
    print("调用 AStarWithPixelCoordinate 工具...")
    astar_result = manager.call_tool("AStarWithPixelCoordinate", astar_params)
    print(f"结果状态: {astar_result.get('status')}")
    print(f"生成的控制序列: {astar_result.get('path')}")
    
    # 使用Draw2DPath绘制路径
    # 创建一个空白图片作为基础
    base_img = Image.new('RGB', (636, 133), color=(255, 255, 255))
    draw = ImageDraw.Draw(base_img)
    
    # 绘制网格和障碍物
    cell_size = 64
    for i in range(10):
        draw.line([(i*cell_size, 0), (i*cell_size, 133)], fill=(200, 200, 200), width=1)
    for i in range(3):
        draw.line([(0, i*cell_size), (636, i*cell_size)], fill=(200, 200, 200), width=1)
    
    # 绘制障碍物
    draw.rectangle([cell_size+1, 1, 2*cell_size-1, cell_size-1], fill=(100, 100, 100))
    
    # 标记起点和终点
    draw.ellipse([5, 5, cell_size-5, cell_size-5], fill=(0, 255, 0))
    draw.ellipse([2*cell_size+5, 5, 3*cell_size-5, cell_size-5], fill=(255, 0, 0))
    
    # 转换为base64
    img_base64 = pil_to_base64(base_img)
    
    # 使用Draw2DPath绘制路径
    draw_params = {
        "image": img_base64,
        "start_point": [32, 32],  # 起点中心
        "directions": astar_result.get('path')  # 使用A*的结果
    }
    
    print("\n调用 Draw2DPath 工具...")
    draw_result = manager.call_tool("Draw2DPath", draw_params)
    print(f"结果状态: {draw_result.get('status')}")
    
    # 保存结果图片
    if 'edited_image' in draw_result:
        result_img = base64_to_pil(draw_result['edited_image'])
        result_img_path = "path_result.png"
        result_img.save(result_img_path)
        print(f"路径绘制结果已保存至: {os.path.abspath(result_img_path)}")
    
    return astar_result, draw_result

def main():
    """主测试函数"""
    print("开始测试 ToolManager 功能...")
    
    # 1. 测试默认初始化
    full_manager = test_default_initialization()
    
    # 2. 测试自定义初始化
    custom_manager = test_custom_initialization()
    
    # 3. 测试获取工具说明
    instructions = test_get_tool_instructions(full_manager)
    
    # 4. 测试获取工具提示语
    prompt = test_get_tool_prompt(custom_manager)
    
    # 5. 测试工具调用
    astar_result, draw_result = test_tool_call(custom_manager)
    
    print("\n=== 测试完成 ===")
    print(f"初始化的工具总数: {len(full_manager.available_tools)}")
    print(f"自定义工具总数: {len(custom_manager.available_tools)}")
    print(f"获取到的工具说明数量: {len(instructions)}")
    
    # 打印测试结论
    print("\n测试结论:")
    if len(custom_manager.available_tools) < len(full_manager.available_tools):
        print("✅ 自定义初始化工具成功，只加载了指定工具")
    else:
        print("❌ 自定义初始化工具失败，加载了全部工具")
    
    if astar_result.get('status') == 'success' and draw_result.get('status') == 'success':
        print("✅ 工具调用成功，A*寻路和路径绘制均正常工作")
    else:
        print("❌ 工具调用存在问题，请检查错误信息")
    
    if len(prompt) > 100:
        print("✅ 成功获取工具提示语，包含了工具说明")
    else:
        print("❌ 获取工具提示语失败或内容不完整")

if __name__ == "__main__":
    main()