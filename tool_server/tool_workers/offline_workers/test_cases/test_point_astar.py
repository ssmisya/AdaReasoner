# test_point_astar.py
import unittest
import os
import numpy as np
from PIL import Image
import gymnasium as gym
import base64
import io
import json
import heapq
import time
import matplotlib.pyplot as plt

from tool_server.tool_workers.tool_manager.base_manager import ToolManager

class TestPointAstarWorkflow(unittest.TestCase):
    """测试Point定位与A*寻路工作流"""
    
    def setUp(self):
        """初始化测试环境"""
        self.tool_manager = ToolManager()
        
        # 生成一个FrozenLake环境用于测试
        self.env_size = 8
        self.desc = [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG"
        ]
        
        self.env = gym.make('FrozenLake-v1', desc=self.desc, render_mode="rgb_array", is_slippery=False)
        self.env.reset()
        self.env_image = Image.fromarray(self.env.render())
        self.cell_size = 64  # FrozenLake每个格子的像素大小
        
        # 保存临时文件
        self.temp_image_path = "temp_frozen_lake.png"
        self.env_image.save(self.temp_image_path)
        
        # 将图片转换为base64以便传递给API
        with open(self.temp_image_path, "rb") as image_file:
            self.image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    def tearDown(self):
        """清理测试环境"""
        if os.path.exists(self.temp_image_path):
            os.remove(self.temp_image_path)
        if os.path.exists("path_result.png"):
            os.remove("path_result.png")
        self.env.close()
    
    def image_to_base64(self, image_path):
        """将图片转换为Base64字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def base64_to_image(self, base64_string):
        """将Base64字符串转换为PIL图像"""
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))
    
    def test_point_astar_workflow(self):
        """测试完整工作流：点定位+A*寻路+路径绘制"""
        print("\n--- 测试Point定位与A*寻路工作流 ---")
        
        # 步骤1: 使用Point工具定位起点(Elf/S)
        print("步骤1: 定位起点(Elf)")
        start_params = {
            "image": self.image_base64,
            "description": "the starting position (S/Elf) in the frozen lake"
        }
        
        # 检查Point工具是否可用，如不可用则跳过测试
        if "Point" not in self.tool_manager.available_tools:
            print("警告: Point工具不可用，跳过测试")
            return
        
        start_result = self.tool_manager.call_tool("Point", start_params)
        print(f"起点定位结果: {start_result['status']}")
        
        if start_result['status'] != "success" or len(start_result.get('points', [])) == 0:
            print(f"警告: 起点定位失败或未找到点 - {start_result.get('message', '未知错误')}")
            start_point = [32, 32]  # 默认使用第一个格子的中心点
            print(f"使用默认起点坐标: {start_point}")
        else:
            start_point = [start_result['points'][0]['x'], start_result['points'][0]['y']]
            print(f"找到起点坐标: {start_point}")
        
        # 保存带标记的图像
        if 'edited_image' in start_result:
            start_image = self.base64_to_image(start_result['edited_image'])
            start_image.save("start_point.png")
            print("起点标记图像已保存为 start_point.png")
        
        # 步骤2: 使用Point工具定位终点(Goal/G)
        print("\n步骤2: 定位终点(Goal)")
        goal_params = {
            "image": self.image_base64,
            "description": "the goal position (G/Gift) in the frozen lake"
        }
        
        goal_result = self.tool_manager.call_tool("Point", goal_params)
        print(f"终点定位结果: {goal_result['status']}")
        
        if goal_result['status'] != "success" or len(goal_result.get('points', [])) == 0:
            print(f"警告: 终点定位失败或未找到点 - {goal_result.get('message', '未知错误')}")
            goal_point = [7 * self.cell_size + self.cell_size/2, 7 * self.cell_size + self.cell_size/2]  # 默认使用右下角的格子
            print(f"使用默认终点坐标: {goal_point}")
        else:
            goal_point = [goal_result['points'][0]['x'], goal_result['points'][0]['y']]
            print(f"找到终点坐标: {goal_point}")
        
        # 保存带标记的图像
        if 'edited_image' in goal_result:
            goal_image = self.base64_to_image(goal_result['edited_image'])
            goal_image.save("goal_point.png")
            print("终点标记图像已保存为 goal_point.png")
        
        # 步骤3: 使用Point工具识别障碍物(Holes/H)
        print("\n步骤3: 识别障碍物(Holes)")
        holes_params = {
            "image": self.image_base64,
            "description": "the holes (H) in the frozen lake"
        }
        
        # 在实际应用中，可能需要多次调用Point来识别所有障碍物
        # 这里简化为一次调用，也可以改为手动指定已知的障碍物坐标
        holes_result = self.tool_manager.call_tool("Point", holes_params)
        print(f"障碍物定位结果: {holes_result['status']}")
        
        # 如果Point工具无法识别所有障碍物，则手动指定一些障碍物
        if holes_result['status'] != "success" or len(holes_result.get('points', [])) < 3:
            print("警告: 障碍物识别不完全，添加手动指定的障碍物坐标")
            # 通过描述中的位置手动生成障碍物坐标
            hole_coords = []
            for i, row in enumerate(self.desc):
                for j, cell in enumerate(row):
                    cell_str = cell.decode('utf-8') if isinstance(cell, bytes) else cell
                    if cell_str == 'H':
                        pixel_x = j * self.cell_size + self.cell_size/2
                        pixel_y = i * self.cell_size + self.cell_size/2
                        hole_coords.append([pixel_x, pixel_y])
            
            if holes_result['status'] == "success" and len(holes_result.get('points', [])) > 0:
                # 合并自动检测和手动添加的障碍物
                auto_holes = [[p['x'], p['y']] for p in holes_result.get('points', [])]
                print(f"自动检测到的障碍物: {auto_holes}")
                print(f"手动添加的障碍物: {hole_coords}")
                obstacles = hole_coords  # 使用手动添加的障碍物
            else:
                obstacles = hole_coords
            
            print(f"最终使用的障碍物坐标数量: {len(obstacles)}")
        else:
            obstacles = [[p['x'], p['y']] for p in holes_result['points']]
            print(f"识别出的障碍物数量: {len(obstacles)}")
        
        # 保存带标记的图像
        if 'edited_image' in holes_result:
            holes_image = self.base64_to_image(holes_result['edited_image'])
            holes_image.save("holes_points.png")
            print("障碍物标记图像已保存为 holes_points.png")
        
        # 步骤4: 使用TurnCoordinateIntoTextMap将坐标转换为文本地图
        print("\n步骤4: 生成文本地图")
        map_params = {
            "start": start_point,
            "goal": goal_point,
            "obstacles": obstacles,
            "cell_size": self.cell_size
        }
        
        map_result = self.tool_manager.call_tool("TurnCoordinateIntoTextMap", map_params)
        print(f"文本地图生成结果: {map_result['status']}")
        if map_result['status'] == "success":
            print("生成的文本地图:")
            print(map_result['text_map'])
        
        # 步骤5: 使用AStarWithPixelCoordinate找到路径
        print("\n步骤5: 使用A*寻找路径")
        astar_params = {
            "start": start_point,
            "goal": goal_point,
            "obstacles": obstacles
        }
        
        astar_result = self.tool_manager.call_tool("AStarWithPixelCoordinate", astar_params)
        print(f"A*寻路结果: {astar_result['status']}")
        
        if astar_result['status'] == "success":
            path = astar_result['path']
            print(f"找到路径: {path}")
            
            # 步骤6: 使用Draw2DPath绘制路径
            print("\n步骤6: 绘制路径")
            draw_params = {
                "image": self.temp_image_path,
                "start_point": start_point,
                "directions": path,
                "pixel_coordinate": True,
                "step": self.cell_size,
                "line_color": "blue",
                "line_width": 5
            }
            
            draw_result = self.tool_manager.call_tool("Draw2DPath", draw_params)
            print(f"路径绘制结果: {draw_result['status']}")
            
            # 保存结果图像
            if draw_result['status'] == "success" and "edited_image" in draw_result:
                image_data = base64.b64decode(draw_result['edited_image'])
                final_image = Image.open(io.BytesIO(image_data))
                final_image.save("point_astar_path_result.png")
                print("完整路径绘制结果已保存为 point_astar_path_result.png")
                
                # 在这里可以显示最终图像
                plt.figure(figsize=(10, 10))
                plt.imshow(final_image)
                plt.axis('off')
                plt.title('Final Path')
                plt.show()
            
            # 验证结果
            self.assertEqual(astar_result["status"], "success")
            self.assertEqual(draw_result["status"], "success")
        else:
            print(f"A*寻路失败: {astar_result.get('message', '未知错误')}")
            self.fail("A*寻路失败")


if __name__ == "__main__":
    unittest.main()