# test_all.py
import unittest
import os
import numpy as np
from PIL import Image
import gymnasium as gym
import base64
import io
import json
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from tool_server.tool_workers.tool_manager.base_manager import ToolManager


class TestOfflineTools(unittest.TestCase):
    """测试所有离线工具的功能"""
    
    def setUp(self):
        """初始化测试环境"""
        self.tool_manager = ToolManager()
        # 生成一个FrozenLake环境用于测试
        self.env_size = 8
        self.text_map = generate_random_map(size=self.env_size, p=0.8)  # p=0.8表示20%的概率生成障碍物
        self.env = gym.make('FrozenLake-v1', desc=self.text_map, render_mode="rgb_array", is_slippery=False)
        self.env.reset()
        self.env_image = Image.fromarray(self.env.render())
        self.cell_size = 64  # FrozenLake每个格子的像素大小
        
        # 提取起点、终点和障碍物的坐标
        self.coords = self.extract_coordinates(self.text_map, self.cell_size)
        
        # 保存临时文件
        self.temp_image_path = "temp_frozen_lake.png"
        self.env_image.save(self.temp_image_path)
    
    def tearDown(self):
        """清理测试环境"""
        if os.path.exists(self.temp_image_path):
            os.remove(self.temp_image_path)
        self.env.close()
    
    def extract_coordinates(self, text_map, cell_size=64):
        """
        从文本地图中提取起点、终点和障碍物的坐标
        
        Args:
            text_map (list): 2D地图，S=起点, G=终点, H=障碍物
            cell_size (int): 每个格子的像素大小
            
        Returns:
            dict: 包含起点、终点和障碍物的像素坐标
        """
        half_cell = cell_size / 2
        start_point = None
        holes = []
        goal_point = None
        
        for i, row in enumerate(text_map):
            for j, cell in enumerate(row):
                # 计算像素坐标（格子中心）
                pixel_x = j * cell_size + half_cell
                pixel_y = i * cell_size + half_cell
                
                cell_str = cell.decode('utf-8') if isinstance(cell, bytes) else cell
                
                if cell_str == 'S':
                    start_point = (pixel_x, pixel_y)
                elif cell_str == 'H':
                    holes.append((pixel_x, pixel_y))
                elif cell_str == 'G':
                    goal_point = (pixel_x, pixel_y)
        
        return {
            'start': start_point,
            'obstacles': holes,
            'goal': goal_point
        }
    
    def image_to_base64(self, image):
        """将PIL图像转换为Base64字符串"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def test_astar_pathfinding(self):
        """测试A*寻路算法"""
        print("\n--- 测试 AStarWithPixelCoordinate 工具 ---")
        
        params = {
            "start": list(self.coords['start']),
            "goal": list(self.coords['goal']),
            "obstacles": [list(obs) for obs in self.coords['obstacles']]
        }
        
        result = self.tool_manager.call_tool("AStarWithPixelCoordinate", params)
        
        print(f"起点: {params['start']}")
        print(f"终点: {params['goal']}")
        print(f"障碍物数量: {len(params['obstacles'])}")
        print(f"A*寻路结果: {result}")
        
        # 验证结果
        self.assertEqual(result["error_code"], 0)
        self.assertEqual(result["status"], "success")
        self.assertIsInstance(result["path"], str)
        self.assertTrue(len(result["path"]) > 0)
    
    def test_draw_path(self):
        """测试路径绘制工具"""
        print("\n--- 测试 Draw2DPath 工具 ---")
        
        # 首先使用A*找到一条路径
        astar_params = {
            "start": list(self.coords['start']),
            "goal": list(self.coords['goal']),
            "obstacles": [list(obs) for obs in self.coords['obstacles']]
        }
        
        astar_result = self.tool_manager.call_tool("AStarWithPixelCoordinate", astar_params)
        
        # 使用A*的结果来绘制路径
        draw_params = {
            "image": self.temp_image_path,
            "start_point": list(self.coords['start']),
            "directions": astar_result["path"],
            "pixel_coordinate": True,
            "step": self.cell_size
        }
        
        draw_result = self.tool_manager.call_tool("Draw2DPath", draw_params)
        
        print(f"路径绘制结果: {draw_result['status']}")
        self.assertEqual(draw_result["status"], "success")
        self.assertTrue("edited_image" in draw_result)
        
        # 可选：保存结果图像查看
        if "edited_image" in draw_result:
            image_data = base64.b64decode(draw_result["edited_image"])
            path_image = Image.open(io.BytesIO(image_data))
            path_image.save("path_result.png")
            print("路径绘制结果已保存为 path_result.png")
    
    def test_text_map_generation(self):
        """测试坐标转文本地图工具"""
        print("\n--- 测试 TurnCoordinateIntoTextMap 工具 ---")
        
        params = {
            "start": list(self.coords['start']),
            "goal": list(self.coords['goal']),
            "obstacles": [list(obs) for obs in self.coords['obstacles']],
            "cell_size": self.cell_size
        }
        
        result = self.tool_manager.call_tool("TurnCoordinateIntoTextMap", params)
        
        print(f"文本地图生成结果: {result['status']}")
        print(f"生成的文本地图:\n{result['text_map']}")
        
        # 验证结果
        self.assertEqual(result["status"], "success")
        self.assertIsInstance(result["text_map"], str)
        
        # 检查地图内容是否包含必要元素
        self.assertTrue("@" in result["text_map"])  # 起点标记
        self.assertTrue("*" in result["text_map"])  # 终点标记
        self.assertTrue("#" in result["text_map"])  # 障碍物标记
        
    def test_integrated_workflow(self):
        """测试工具集成工作流"""
        print("\n--- 测试完整工作流 ---")
        
        # 步骤1: 使用TurnCoordinateIntoTextMap生成文本地图
        map_params = {
            "start": list(self.coords['start']),
            "goal": list(self.coords['goal']),
            "obstacles": [list(obs) for obs in self.coords['obstacles']],
            "cell_size": self.cell_size
        }
        
        map_result = self.tool_manager.call_tool("TurnCoordinateIntoTextMap", map_params)
        print("步骤1: 生成文本地图")
        print(map_result["text_map"])
        
        # 步骤2: 使用AStarWithPixelCoordinate找到路径
        astar_params = {
            "start": list(self.coords['start']),
            "goal": list(self.coords['goal']),
            "obstacles": [list(obs) for obs in self.coords['obstacles']]
        }
        
        astar_result = self.tool_manager.call_tool("AStarWithPixelCoordinate", astar_params)
        print("\n步骤2: 使用A*寻找路径")
        print(f"路径: {astar_result['path']}")
        
        # 步骤3: 使用Draw2DPath绘制路径
        draw_params = {
            "image": self.temp_image_path,
            "start_point": list(self.coords['start']),
            "directions": astar_result["path"],
            "pixel_coordinate": True,
            "step": self.cell_size
        }
        
        draw_result = self.tool_manager.call_tool("Draw2DPath", draw_params)
        print("\n步骤3: 绘制路径")
        print(f"绘制结果: {draw_result['status']}")
        
        # 保存最终结果
        if draw_result["status"] == "success" and "edited_image" in draw_result:
            image_data = base64.b64decode(draw_result["edited_image"])
            final_image = Image.open(io.BytesIO(image_data))
            final_image.save("integrated_workflow_result.png")
            print("完整工作流结果已保存为 integrated_workflow_result.png")
        
        # 验证整个流程是否成功
        self.assertEqual(map_result["status"], "success")
        self.assertEqual(astar_result["status"], "success")
        self.assertEqual(draw_result["status"], "success")


if __name__ == "__main__":
    unittest.main()