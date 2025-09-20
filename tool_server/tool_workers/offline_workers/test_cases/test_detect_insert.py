# test_detect_insert.py
import unittest
import os
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
import random
import time
import argparse
import sys
import cv2

from tool_server.tool_workers.tool_manager.base_manager import ToolManager

class TestDetectAndInsert(unittest.TestCase):
    """Test the functionality of DetectBlackArea and InsertImage tools"""
    
    @classmethod
    def setUpClass(cls):
        """Initialize test environment"""
        # Since unittest passes its own parameters when running, we need to get the image path during class initialization
        # Here we use environment variables to pass parameters, avoiding conflicts with unittest parameters
        cls.input_image_path = os.environ.get('TEST_IMAGE_PATH')
    
    def setUp(self):
        """Initialize test environment"""
        self.tool_manager = ToolManager()
        
        # Create test folder
        os.makedirs("test_results", exist_ok=True)
        
        # If an input image is provided, use it
        if self.__class__.input_image_path and os.path.exists(self.__class__.input_image_path):
            print(f"Using specified input image: {self.__class__.input_image_path}")
            self.original_image = Image.open(self.__class__.input_image_path)
            # Ensure the image is of appropriate size, resize if too small
            if min(self.original_image.size) < 300:
                # Maintain aspect ratio when enlarging the image
                ratio = 300 / min(self.original_image.size)
                new_size = (int(self.original_image.size[0] * ratio), 
                            int(self.original_image.size[1] * ratio))
                self.original_image = self.original_image.resize(new_size, Image.LANCZOS)
        else:
            print("Input image not specified or does not exist, creating default test image")
            # Generate default test image
            self.original_image, _ = self.create_grid_image(3, 3)
        
        # Split image into 3×3 grid
        self.sub_images = self.split_into_grid(self.original_image, 3, 3)
        
        # Create base image (bottom right 2x2) and cut out the bottom right corner
        self.base_image, self.correct_part, self.black_area = self.create_base_image_with_black_area()
        
        # 从剩余的5个子图中随机选择一个
        remaining_parts = [img for i, img in enumerate(self.sub_images) 
                         if i not in [4, 5, 7, 8]]  # 排除右下角2x2的索引
        self.random_part = random.choice(remaining_parts)
        
        # 保存各阶段图片用于测试
        self.original_image_path = "test_results/01_original_image.png"
        self.base_image_path = "test_results/02_base_with_black.png"
        self.correct_part_path = "test_results/03_correct_part.png"
        self.random_part_path = "test_results/04_random_part.png"
        
        self.original_image.save(self.original_image_path)
        self.base_image.save(self.base_image_path)
        self.correct_part.save(self.correct_part_path)
        self.random_part.save(self.random_part_path)
        
        # 保存网格图片，显示划分结果
        self.save_grid_visualization()
        
        print(f"图片已保存到test_results目录")
    
    def tearDown(self):
        """清理测试环境 (可选，取决于是否需要保留结果)"""
        pass
    
    # 其他方法保持不变...
    
    def split_into_grid(self, image, rows, cols):
        """
        将图片分割为网格
        
        Args:
            image: PIL图像对象
            rows: 行数
            cols: 列数
            
        Returns:
            list: 子图列表
        """
        width, height = image.size
        cell_width = width // cols
        cell_height = height // rows
        
        sub_images = []
        for i in range(rows):
            for j in range(cols):
                left = j * cell_width
                upper = i * cell_height
                right = left + cell_width
                lower = upper + cell_height
                
                cell = image.crop((left, upper, right, lower))
                sub_images.append(cell)
        
        return sub_images
    
    def save_grid_visualization(self):
        """保存一个可视化的网格图片，显示图片如何被分割"""
        img = self.original_image.copy()
        draw = ImageDraw.Draw(img)
        
        width, height = img.size
        cell_width = width // 3
        cell_height = height // 3
        
        # 绘制网格线
        for i in range(1, 3):
            # 水平线
            draw.line([(0, i * cell_height), (width, i * cell_height)], fill=(255, 0, 0), width=2)
            # 垂直线
            draw.line([(i * cell_width, 0), (i * cell_width, height)], fill=(255, 0, 0), width=2)
        
        # 标记右下角2x2区域
        draw.rectangle(
            [(cell_width, cell_height), (width, height)],
            outline=(0, 255, 0),
            width=3
        )
        
        # 特别标记右下角单元格（将被替换为黑色）
        draw.rectangle(
            [(cell_width * 2, cell_height * 2), (width, height)],
            outline=(0, 0, 255),
            width=3
        )
        
        # 添加标签说明
        draw.text((10, 10), "红线: 3x3网格分割", fill=(255, 0, 0))
        draw.text((10, 30), "绿线: 右下角2x2区域", fill=(0, 255, 0))
        draw.text((10, 50), "蓝线: 将被涂黑的区域", fill=(0, 0, 255))
        
        grid_viz_path = "test_results/00_grid_visualization.png"
        img.save(grid_viz_path)
        print(f"网格可视化图片已保存为: {grid_viz_path}")
    
    def create_grid_image(self, rows, cols):
        """
        创建一个网格图像，每个网格单元有不同的颜色
        
        Args:
            rows: 行数
            cols: 列数
            
        Returns:
            tuple: (完整图像, 子图列表)
        """
        cell_size = 100  # 每个网格单元的大小
        full_width = cols * cell_size
        full_height = rows * cell_size
        
        # 创建完整图像
        full_image = Image.new('RGB', (full_width, full_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(full_image)
        
        # 创建子图列表
        sub_images = []
        
        # 为每个网格单元生成唯一颜色并绘制
        for i in range(rows):
            for j in range(cols):
                # 生成唯一颜色（避免黑色）
                color = (
                    random.randint(50, 250),
                    random.randint(50, 250),
                    random.randint(50, 250)
                )
                
                # 计算当前单元格的位置
                x0 = j * cell_size
                y0 = i * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                
                # 绘制彩色方块
                draw.rectangle([x0, y0, x1, y1], fill=color)
                
                # 添加一些文字以便识别
                cell_id = i * cols + j + 1
                draw.text((x0 + 10, y0 + 10), f"Cell {cell_id}", fill=(0, 0, 0))
                
                # 添加一些随机图形
                shape_type = random.choice(["circle", "rectangle", "line"])
                if shape_type == "circle":
                    draw.ellipse([x0+20, y0+40, x0+80, y0+80], 
                                 fill=(255-color[0], 255-color[1], 255-color[2]))
                elif shape_type == "rectangle":
                    draw.rectangle([x0+30, y0+40, x0+70, y0+80], 
                                   fill=(255-color[0], 255-color[1], 255-color[2]))
                else:
                    draw.line([x0+20, y0+40, x0+80, y0+80], 
                              fill=(255-color[0], 255-color[1], 255-color[2]), width=5)
                
                # 创建并存储子图
                sub_image = full_image.crop((x0, y0, x1, y1))
                sub_images.append(sub_image)
        
        return full_image, sub_images
    
    def create_base_image_with_black_area(self):
        """
        创建一个右下角2x2的base图像，并挖空右下角的单元格
        
        Returns:
            tuple: (base图像, 被挖空的部分, 黑色区域坐标)
        """
        # 计算网格单元的大小
        width, height = self.original_image.size
        cell_width = width // 3
        cell_height = height // 3
        
        # 创建2x2大小的base图像
        base_width = 2 * cell_width
        base_height = 2 * cell_height
        base_image = Image.new('RGB', (base_width, base_height))
        
        # 获取右下角2x2区域的单元格
        right_bottom_cells = [
            self.sub_images[4],  # 右下2x2区域的左上
            self.sub_images[5],  # 右下2x2区域的右上
            self.sub_images[7],  # 右下2x2区域的左下
            self.sub_images[8]   # 右下2x2区域的右下
        ]
        
        # 复制前3个单元格
        base_image.paste(right_bottom_cells[0], (0, 0))
        base_image.paste(right_bottom_cells[1], (cell_width, 0))
        base_image.paste(right_bottom_cells[2], (0, cell_height))
        
        # 保存右下角单元格
        correct_part = right_bottom_cells[3].copy()
        
        # 将右下角涂黑
        draw = ImageDraw.Draw(base_image)
        black_area = [cell_width, cell_height, 2*cell_width, 2*cell_height]
        draw.rectangle(black_area, fill=(0, 0, 0))
        
        return base_image, correct_part, black_area
    
    def image_to_base64(self, image):
        """将PIL图像转换为Base64字符串"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def base64_to_image(self, base64_string):
        """将Base64字符串转换为PIL图像"""
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))
    
    def test_detect_black_area(self):
        """测试黑色区域检测功能"""
        print("\n--- 测试 DetectBlackArea 工具 ---")
        
        params = {
            "image": self.base_image_path,
            "min_area": 100
        }
        
        # 调用工具检测黑色区域
        result = self.tool_manager.call_tool("DetectBlackArea", params)
        
        print(f"检测结果: {result}")
        
        # 保存带有检测框的可视化结果
        self.save_detection_visualization(result)
        
        # 验证结果
        self.assertEqual(result["status"], "success")
        self.assertGreaterEqual(len(result["bounding_boxes"]), 1)
        
        # 检查检测到的边界框是否与预期区域接近
        detected_box = None
        
        box_str = result["bounding_boxes"][0]
        
        box = eval(box_str)
        if (box[0] >= self.black_area[0] - 10 and 
            box[1] >= self.black_area[1] - 10 and 
            box[2] <= self.black_area[2] + 10 and 
            box[3] <= self.black_area[3] + 10):
            detected_box = box_str

        
        self.assertIsNotNone(detected_box, "未检测到黑色区域或位置不正确")
        print(f"检测到的黑色区域: {detected_box}")
        
        return detected_box
    
    def save_detection_visualization(self, detection_result):
        """将检测结果可视化保存为图片"""
        # 加载基础图片
        base_img = Image.open(self.base_image_path)
        draw = ImageDraw.Draw(base_img)
        
        # 在图片上绘制所有检测到的边界框
        for box_str in detection_result["bounding_boxes"]:
            box = eval(box_str)
            draw.rectangle([box[0], box[1], box[2], box[3]], outline=(255, 0, 0), width=2)
            draw.text((box[0], box[1]-20), f"Detected: {box}", fill=(255, 0, 0))
        
        # 在图片上标记期望的黑色区域
        draw.rectangle(self.black_area, outline=(0, 255, 0), width=2)
        draw.text((self.black_area[0], self.black_area[1]-20), "Expected black area", fill=(0, 255, 0))
        
        # 保存可视化结果
        viz_path = "test_results/05_detection_visualization.png"
        base_img.save(viz_path)
        print(f"检测可视化图片已保存为: {viz_path}")
    
    def test_insert_images(self):
        """测试使用正确和随机子图进行图片插入的功能"""
        print("\n--- 测试图片插入功能 ---")
        
        # 首先使用DetectBlackArea找到黑色区域
        bbox = self.test_detect_black_area()
        
        base_image = Image.open(self.base_image_path).convert("RGB")
        base_image_base64 = self.image_to_base64(base_image)
        
        
        # 使用正确的子图插入
        correct_params = {
            "base_image": base_image_base64,
            "image_to_insert": self.correct_part_path,
            "coordinates": bbox,
            "resize": True
        }
        
        # 使用随机选择的子图插入
        random_params = {
            "base_image": base_image_base64,
            "image_to_insert": self.random_part_path,
            "coordinates": bbox,
            "resize": True
        }
        
        # 调用工具插入图片
        correct_result = self.tool_manager.call_tool("InsertImage", correct_params)
        random_result = self.tool_manager.call_tool("InsertImage", random_params)
        
        
        
        print(f"正确子图插入结果: {correct_result['status']}")
        print(f"随机子图插入结果: {random_result['status']}")
        
        # 验证结果
        self.assertEqual(correct_result["status"], "success")
        self.assertEqual(random_result["status"], "success")
        
        # 将结果图像保存到文件
        if "edited_image" in correct_result:
            correct_image = self.base64_to_image(correct_result["edited_image"])
            correct_image_path = "test_results/06_result_correct.png"
            correct_image.save(correct_image_path)
            print(f"正确子图插入后的结果已保存到: {correct_image_path}")
            correct_result.pop("edited_image")  # 删除大数据以便打印
        
        if "edited_image" in random_result:
            random_image = self.base64_to_image(random_result["edited_image"])
            random_image_path = "test_results/07_result_random.png"
            random_image.save(random_image_path)
            print(f"随机子图插入后的结果已保存到: {random_image_path}")
            random_result.pop("edited_image")  # 删除大数据以便打印
            
        print(f"正确子图InsertImage结果:{correct_result}")
        print(f"随机子图InsertImage结果:{random_result}")
        # 保存对比图，显示正确结果和原始图像的右下角2x2部分
        self.save_comparison_visualization(correct_result, random_result)
    
    def save_comparison_visualization(self, correct_result, random_result):
        """保存一个包含原始图像、base图像和两种插入结果的对比图"""
        # 创建一个足够大的画布来展示所有图像
        width, height = self.original_image.size
        canvas = Image.new('RGB', (width * 2, height * 2), color=(240, 240, 240))
        
        # 放置原始图像
        canvas.paste(self.original_image, (0, 0))
        
        # 放置base图像（带黑色区域）
        canvas.paste(self.base_image, (width, 0))
        
        # 放置正确插入结果
        if "edited_image" in correct_result:
            correct_image = self.base64_to_image(correct_result["edited_image"])
            canvas.paste(correct_image, (0, height))
        
        # 放置随机插入结果
        if "edited_image" in random_result:
            random_image = self.base64_to_image(random_result["edited_image"])
            canvas.paste(random_image, (width, height))
        
        # 添加标签
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), "Original Image", fill=(0, 0, 0))
        draw.text((width + 10, 10), "Image with black area", fill=(0, 0, 0))
        draw.text((10, height + 10), "correct res", fill=(0, 0, 0))
        draw.text((width + 10, height + 10), "random res", fill=(0, 0, 0))
        
        # 保存对比图
        comparison_path = "test_results/08_comparison.png"
        canvas.save(comparison_path)
        print(f"对比可视化图片已保存为: {comparison_path}")
    
    def test_integrated_workflow(self):
        """测试完整的工作流程"""
        print("\n--- 测试完整工作流 ---")
        
        # 步骤1: 使用DetectBlackArea检测黑色区域
        detect_params = {
            "image": self.base_image_path,
            "threshold": 10,
            "min_area": 100
        }
        
        detect_result = self.tool_manager.call_tool("DetectBlackArea", detect_params)
        print("步骤1: 检测黑色区域")
        print(f"检测到 {len(detect_result['bounding_boxes'])} 个黑色区域: {detect_result['bounding_boxes']}")
        
        # 确保至少检测到一个区域
        self.assertGreaterEqual(len(detect_result["bounding_boxes"]), 1)
        
        # 使用第一个检测到的区域
        bbox = detect_result["bounding_boxes"][0]
        
        # 步骤2a: 使用正确的子图插入到黑色区域
        correct_params = {
            "base_image": self.base_image_path,
            "image_to_insert": self.correct_part_path,
            "coordinates": bbox,
            "resize": True
        }
        
        # 步骤2b: 使用随机的子图插入到黑色区域
        random_params = {
            "base_image": self.base_image_path,
            "image_to_insert": self.random_part_path,
            "coordinates": bbox,
            "resize": True
        }
        
        # 执行插入操作
        correct_result = self.tool_manager.call_tool("InsertImage", correct_params)
        random_result = self.tool_manager.call_tool("InsertImage", random_params)
        
        print("\n步骤2: 插入图片到黑色区域")
        print(f"正确子图插入结果: {correct_result['status']}")
        print(f"随机子图插入结果: {random_result['status']}")
        
        # 验证插入是否成功
        self.assertEqual(correct_result["status"], "success")
        self.assertEqual(random_result["status"], "success")
        
        # 保存最终结果
        if correct_result["status"] == "success" and "edited_image" in correct_result:
            result_image = self.base64_to_image(correct_result["edited_image"])
            final_path = "test_results/09_integrated_result_correct.png"
            result_image.save(final_path)
            print(f"正确子图插入结果已保存为 {final_path}")
        
        if random_result["status"] == "success" and "edited_image" in random_result:
            result_image = self.base64_to_image(random_result["edited_image"])
            final_path = "test_results/10_integrated_result_random.png"
            result_image.save(final_path)
            print(f"随机子图插入结果已保存为 {final_path}")
        
        # 验证整个流程是否成功
        self.assertEqual(detect_result["status"], "success")
        self.assertEqual(correct_result["status"], "success")
        self.assertEqual(random_result["status"], "success")
        
        # 创建一个最终的总结图像，显示整个流程的结果
        self.save_workflow_summary()
    
    def save_workflow_summary(self):
        """保存一个总结工作流程的图像"""
        # 创建足够大的画布
        canvas_width = 1200
        canvas_height = 800
        canvas = Image.new('RGB', (canvas_width, canvas_height), color=(240, 240, 240))
        draw = ImageDraw.Draw(canvas)
        
        # 添加标题
        draw.text((canvas_width//2-150, 20), "DetectBlackArea + InsertImage 工作流程", fill=(0, 0, 0))
        
        # 加载需要的图像
        original = Image.open(self.original_image_path).resize((300, 300))
        base = Image.open(self.base_image_path).resize((300, 300))
        correct_part = Image.open(self.correct_part_path).resize((100, 100))
        random_part = Image.open(self.random_part_path).resize((100, 100))
        
        result_correct_path = "test_results/06_result_correct.png"
        result_random_path = "test_results/07_result_random.png"
        
        if os.path.exists(result_correct_path) and os.path.exists(result_random_path):
            result_correct = Image.open(result_correct_path).resize((300, 300))
            result_random = Image.open(result_random_path).resize((300, 300))
            
            # 放置图像
            canvas.paste(original, (50, 100))
            canvas.paste(base, (450, 100))
            canvas.paste(correct_part, (350, 450))
            canvas.paste(random_part, (350, 580))
            canvas.paste(result_correct, (850, 100))
            canvas.paste(result_random, (850, 450))
            
            # 添加说明文字
            draw.text((50, 70), "1. 原始图像", fill=(0, 0, 0))
            draw.text((450, 70), "2. 带黑色区域的base图像", fill=(0, 0, 0))
            draw.text((350, 420), "3a. 正确子图", fill=(0, 0, 0))
            draw.text((350, 550), "3b. 随机子图", fill=(0, 0, 0))
            draw.text((850, 70), "4a. 正确子图插入结果", fill=(0, 0, 0))
            draw.text((850, 420), "4b. 随机子图插入结果", fill=(0, 0, 0))
            
            # 绘制流程箭头
            draw.line([(350, 250), (450, 250)], fill=(0, 0, 0), width=2)  # 1 -> 2
            draw.line([(750, 250), (850, 250)], fill=(0, 0, 0), width=2)  # 2 -> 4a
            draw.line([(600, 250), (600, 500), (850, 500)], fill=(0, 0, 0), width=2)  # 2 -> 4b
            
            # 保存总结图像
            summary_path = "test_results/11_workflow_summary.png"
            canvas.save(summary_path)
            print(f"工作流程总结图片已保存为: {summary_path}")


# 创建一个独立的运行脚本，而不是直接在测试类中添加运行代码
def main():
    """主函数，用于处理命令行参数并运行测试"""
    parser = argparse.ArgumentParser(description='测试DetectBlackArea和InsertImage工具')
    parser.add_argument('--image', type=str, required=True, help='输入图片路径')
    
    args = parser.parse_args()
    
    # 验证图片路径
    if not os.path.exists(args.image):
        print(f"错误：图片路径不存在: {args.image}")
        return
    
    # 通过环境变量传递图片路径
    os.environ['TEST_IMAGE_PATH'] = args.image
    
    # 运行测试
    unittest.main(argv=['first-arg-is-ignored'])


if __name__ == "__main__":
    # 判断是通过unittest运行还是直接脚本运行
    if len(sys.argv) > 1 and '--image' in sys.argv:
        # 直接脚本运行，处理命令行参数
        main()
    else:
        # unittest方式运行
        unittest.main()