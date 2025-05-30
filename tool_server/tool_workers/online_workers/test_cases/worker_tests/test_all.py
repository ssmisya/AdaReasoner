"""一键运行所有工具测试脚本"""
import argparse
import json
import time
from io import BytesIO
import cv2
import sys
import numpy as np
import os
import requests
import base64
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw

# 尝试导入可能需要的工具函数
try:
    from tool_server.utils.utils import *
    from tool_server.utils.server_utils import *
    from tool_server.tool_workers.online_workers.utils import annotate_xyxy
except ImportError:
    print("警告: 部分工具模块导入失败，某些功能可能受限")

AVAILABLE_TOOLS = [
    "point", 
    "draw_vertical_line",
    "ocr",
    "segment_region",
    "grounding_dino"
]

def load_image(image_path):
    """加载并适当调整图像大小"""
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    if max(h, w) > 800:
        if h > w:
            new_h = 800
            new_w = int(w * 800 / h)
        else:
            new_w = 800
            new_h = int(h * 800 / w)
        img = F.resize(img, (new_h, new_w))
    return img

def encode(image: Image):
    """将图像编码为base64字符串"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
    return img_b64_str

def base64_to_pil(image_b64):
    """将base64字符串转换为PIL图像"""
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_bytes))
    return image

def get_worker_address(controller_addr, model_name):
    """获取指定模型的工作节点地址"""
    print(f"尝试获取{model_name}的工作节点地址...")
    
    try:
        # 列出所有可用模型
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        print(f"可用模型: {models}")
        
        # 获取指定模型的地址
        ret = requests.post(
            controller_addr + "/get_worker_address", 
            json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        
        if not worker_addr:
            print(f"没有找到{model_name}的可用工作节点")
            return None
            
        print(f"工作节点地址: {worker_addr}")
        return worker_addr
        
    except Exception as e:
        print(f"获取工作节点地址时出错: {str(e)}")
        return None

def test_point(args):
    """测试Point工具"""
    print("\n====== 测试Point工具 ======")
    model_name = 'Point'
    
    worker_addr =  get_worker_address(args.controller_address, model_name)
    if not worker_addr:
        return
        
    img = load_image(args.image_path)
    img_arg = encode(img)
    
    datas = {
        "model": model_name,
        "param": "Point E",
        "image": img_arg,
    }
    
    tic = time.time()
    response = requests.post(
        worker_addr + "/worker_generate",
        headers={"User-Agent": "FastChat Client"},
        json=datas,
    )
    toc = time.time()
    
    print(f"耗时: {toc - tic:.3f}s")
    print(f"返回状态: {response.status_code}")
    
    if response.status_code == 200:
        res = response.json()
        print(f"Point工具响应: {res['text']}")
        
        if "edited_image" in res:
            image = base64_to_pil(res["edited_image"])
            output_path = os.path.join(args.output_dir, "point_result.jpg")
            image.save(output_path)
            print(f"结果图像已保存至: {output_path}")
    else:
        print(f"请求失败: {response.text}")

def test_draw_vertical_line(args):
    """测试DrawVerticalLineByX工具"""
    print("\n====== 测试DrawVerticalLineByX工具 ======")
    model_name = 'DrawVerticalLineByX'
    
    worker_addr =  get_worker_address(args.controller_address, model_name)
    if not worker_addr:
        return
    
    img = load_image(args.image_path)
    img_arg = encode(img)
    
    datas = {
        "model": model_name,
        "param": "<point x=\"51.5\" y=\"82.0\" alt=\"x = 2017\">x = 2017</point>",
        "image": img_arg,
    }
    
    tic = time.time()
    response = requests.post(
        worker_addr + "/worker_generate",
        headers={"User-Agent": "FastChat Client"},
        json=datas,
    )
    toc = time.time()
    
    print(f"耗时: {toc - tic:.3f}s")
    print(f"返回状态: {response.status_code}")
    
    if response.status_code == 200:
        res = response.json()
        print(f"绘制垂直线工具响应: {res['text']}")
        
        if "edited_image" in res:
            image = base64_to_pil(res["edited_image"])
            output_path = os.path.join(args.output_dir, "vertical_line_result.jpg")
            image.save(output_path)
            print(f"结果图像已保存至: {output_path}")
    else:
        print(f"请求失败: {response.text}")

def test_ocr(args):
    """测试OCR工具"""
    print("\n====== 测试OCR工具 ======")
    model_name = 'OCR'
    
    worker_addr =  get_worker_address(args.controller_address, model_name)
    if not worker_addr:
        return
        
    img = load_image(args.image_path)
    img_arg = encode(img)
    
    datas = {
        "model": model_name,
        "image": img_arg,
    }
    
    tic = time.time()
    response = requests.post(
        worker_addr + "/worker_generate",
        headers={"User-Agent": "FastChat Client"},
        json=datas,
    )
    toc = time.time()
    
    print(f"耗时: {toc - tic:.3f}s")
    print(f"返回状态: {response.status_code}")
    
    if response.status_code == 200:
        res = response.json()
        print(f"OCR工具响应: {res['text']}")
    else:
        print(f"请求失败: {response.text}")

def test_segment_region(args):
    """测试SegmentRegionAroundPoint工具"""
    print("\n====== 测试SegmentRegionAroundPoint工具 ======")
    model_name = 'SegmentRegionAroundPoint'
    
    worker_addr =  get_worker_address(args.controller_address, model_name)
    if not worker_addr:
        return
    
    img = load_image(args.image_path)
    img_arg = encode(img)
    
    datas = {
        "model": model_name,
        "param": "<point x=\"63.0\" y=\"46.5\" alt=\"truck in the scene\">truck in the scene</point>",
        "image": img_arg,
    }
    
    tic = time.time()
    response = requests.post(
        worker_addr + "/worker_generate",
        headers={"User-Agent": "FastChat Client"},
        json=datas,
    )
    toc = time.time()
    
    print(f"耗时: {toc - tic:.3f}s")
    print(f"返回状态: {response.status_code}")
    
    if response.status_code == 200:
        res = response.json()
        print(f"分割区域工具响应: {res['text']}")
        
        if "edited_image" in res:
            image = base64_to_pil(res["edited_image"])
            output_path = os.path.join(args.output_dir, "segment_result.jpg")
            image.save(output_path)
            print(f"结果图像已保存至: {output_path}")
    else:
        print(f"请求失败: {response.text}")

def test_grounding_dino(args):
    """测试Grounding DINO工具"""
    print("\n====== 测试Grounding DINO工具 ======")
    model_name = 'grounding_dino'
    
    
    
    worker_addr =  get_worker_address(args.controller_address, model_name)
    if not worker_addr:
        return
    
    img = load_image(args.image_path)
    img_arg = encode(img)
    
    datas = {
        "model": model_name,
        "caption": "car",
        "image": img_arg,
        "box_threshold": 0.3,
        "text_threshold": 0.25,
    }
    
    tic = time.time()
    response = requests.post(
        worker_addr + "/worker_generate",
        headers={"User-Agent": "FastChat Client"},
        json=datas,
    )
    toc = time.time()
    
    print(f"耗时: {toc - tic:.3f}s")
    print(f"返回状态: {response.status_code}")
    
    if response.status_code == 200:
        res = response.json()
        print("Grounding DINO 探测结果:")
        print(json.dumps(res["text"], indent=2))
        
        try:
            # 尝试可视化结果
            boxes = torch.Tensor(res["text"]["boxes"])
            logits = torch.Tensor(res["text"]["logits"])
            phrases = res["text"]["phrases"]
            
            image_source = np.array(Image.open(args.image_path).convert("RGB"))
            
            try:
                annotated_frame = annotate_xyxy(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
                output_path = os.path.join(args.output_dir, "grounding_dino_result.jpg")
                cv2.imwrite(output_path, annotated_frame)
                print(f"已保存标注图像至: {output_path}")
            except NameError:
                print("无法标注框，annotate_xyxy函数不可用")
        except Exception as e:
            print(f"可视化结果时出错: {str(e)}")
    else:
        print(f"请求失败: {response.text}")

def main():
    parser = argparse.ArgumentParser(description="一键测试多种视觉工具")
    
    # 通用参数
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:20001",
        help="控制器地址"
    )
    parser.add_argument(
        "--image-path", type=str, default="./input_cases/subplot_0.png",
        help="测试图像路径"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./test_results",
        help="输出目录"
    )
    parser.add_argument(
        "--tools", type=str, nargs="+", choices=AVAILABLE_TOOLS + ["all"],
        default=["all"], help="要测试的工具列表，可选: " + ", ".join(AVAILABLE_TOOLS)
    )
    
    args = parser.parse_args()
    
    # 检查并创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确定要测试的工具
    tools_to_test = AVAILABLE_TOOLS if "all" in args.tools else args.tools
    
    image_dict = {
        "default":"./input_cases/subplot_0.png",
        "grounding_dino": "./input_cases/truck.jpg"
    }
    
    # 运行测试
    for tool in tools_to_test:
        if tool == "point":
            test_point(args)
        elif tool == "draw_vertical_line":
            test_draw_vertical_line(args)
        elif tool == "ocr":
            test_ocr(args)
        elif tool == "segment_region":
            test_segment_region(args)
        elif tool == "grounding_dino":
            args.image_path = image_dict.get(tool,image_dict["default"])
            test_grounding_dino(args)
    
    print("\n====== 所有测试完成 ======")

if __name__ == "__main__":
    main()