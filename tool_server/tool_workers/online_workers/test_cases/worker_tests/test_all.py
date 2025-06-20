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
    "Point", 
    # "draw_vertical_line",
    # "draw_horizontal_line",
    "draw_line",  # 新增的组合工具
    "OCR",
    "SegmentRegionAroundPoint",
    "GroundingDINO",
    "Crop",
    # "ZoomInSubfigure"
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
    
    worker_addr =  get_worker_address(args.controller_addr, model_name)
    if not worker_addr:
        return
        
    img = load_image(args.image_path)
    img_arg = encode(img)
    
    # 修改参数名称，使用description而非param
    datas = {
        "image": img_arg,
        "description": "Point E"
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
        # 检查正确的响应字段
        if "status" in res:
            print(f"Point工具状态: {res['status']}")
            
        if "points" in res:
            print(f"检测到的点: {res['points']}")
            
        if "raw_response" in res:
            print(f"原始响应: {res['raw_response']}")
            
        if "image_with_points" in res:
            image = base64_to_pil(res["image_with_points"])
            # 修改为PNG格式保存，以支持RGBA模式
            output_path = os.path.join(args.output_dir, "point_result.png")
            image.save(output_path, format="PNG")
            print(f"结果图像已保存至: {output_path}")
    else:
        print(f"请求失败: {response.text}")

# 注释掉旧的垂直线和水平线测试函数
"""
def test_draw_vertical_line(args):
    # ...

def test_draw_horizontal_line(args):
    # ...
"""

def test_draw_line(args):
    """测试DrawLine工具（支持水平线和垂直线）"""
    print("\n====== 测试DrawLine工具 ======")
    model_name = 'DrawLine'
    
    worker_addr = get_worker_address(args.controller_addr, model_name)
    if not worker_addr:
        return
    
    img = load_image(args.image_path)
    img_arg = encode(img)
    
    # 测试两种线型
    test_cases = [
        {
            "name": "水平线",
            "line_type": "horizontal",
            "param": "<point x=\"50.0\" y=\"45.2\" alt=\"y = 2017\">y = 2017</point>",
            "output_file": "horizontal_line_result.jpg"
        },
        {
            "name": "垂直线",
            "line_type": "vertical",
            "param": "<point x=\"51.5\" y=\"82.0\" alt=\"x = 2017\">x = 2017</point>",
            "output_file": "vertical_line_result.jpg"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n------ 测试{test_case['name']} ------")
        
        datas = {
            "model": model_name,
            "line_type": test_case["line_type"],
            "param": test_case["param"],
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
            print(f"绘制{test_case['name']}工具响应: {res.get('message', res.get('text', '无消息'))}")
            
            if "edited_image" in res:
                image = base64_to_pil(res["edited_image"])
                output_path = os.path.join(args.output_dir, test_case["output_file"])
                image.save(output_path)
                print(f"结果图像已保存至: {output_path}")
        else:
            print(f"请求失败: {response.text}")

def test_ocr(args):
    """测试OCR工具"""
    print("\n====== 测试OCR工具 ======")
    model_name = 'OCR'
    
    print(f"控制器地址: {args.controller_addr}")

    worker_addr = get_worker_address(args.controller_addr, model_name)
    print(f"OCR工作节点地址: {worker_addr}")
    if not worker_addr:
        return
        
    img = load_image(args.image_path)
    img_arg = encode(img)
    
    # OCR只需要image参数
    datas = {
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
        
        print(f"OCR工具响应状态: {res.get('status', 'unknown')}")
        
        if "detections" in res:
            detection_count = len(res["detections"])
            print(f"检测到{detection_count}个文本区域:")
            
            # 最多显示前10个文本
            for i, det in enumerate(res["detections"][:10]):
                print(f"  {i+1}. '{det['label']}' (置信度: {det['confidence']:.2f})")
                
            if detection_count > 10:
                print(f"  ... 还有{detection_count-10}个文本未显示")
        else:
            print(f"OCR工具完整响应: {json.dumps(res, indent=2)}")
    else:
        print(f"请求失败: {response.text}")

def test_segment_region(args):
    """测试SegmentRegionAroundPoint工具"""
    print("\n====== 测试SegmentRegionAroundPoint工具 ======")
    model_name = 'SegmentRegionAroundPoint'
    
    worker_addr =  get_worker_address(args.controller_addr, model_name)
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
        print(f"分割区域工具状态: {res['status']}")
        
        if "edited_image" in res and res["status"] == "success":
            image = base64_to_pil(res["edited_image"])
            output_path = os.path.join(args.output_dir, "segment_result.jpg")
            image.save(output_path)
            print(f"结果图像已保存至: {output_path}")
            
            if "image_dimensions_pixels" in res:
                width = res["image_dimensions_pixels"]["width"]
                height = res["image_dimensions_pixels"]["height"]
                print(f"图像尺寸: {width}x{height}")
        else:
            if "message" in res:
                print(f"处理失败: {res['message']}")
            elif "error" in res:
                print(f"处理错误: {res['error']}")
    else:
        print(f"请求失败: {response.text}")

def test_GroundingDINO(args):
    """测试Grounding DINO工具"""
    print("\n====== 测试Grounding DINO工具 ======")
    model_name = 'GroundingDINO'
    
    worker_addr = get_worker_address(args.controller_addr, model_name)
    if not worker_addr:
        return
    
    img = load_image(args.image_path)
    img_arg = encode(img)
    
    datas = {
        "model": model_name,
        "caption": "car",  # 使用caption或description都可以
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
        print(f"Grounding DINO工具状态: {res.get('status', '未知')}")
        
        if "message" in res:
            print(f"消息: {res['message']}")
        
        if "detections" in res:
            detection_count = len(res["detections"])
            print(f"检测到{detection_count}个目标:")
            
            # 最多显示前5个检测结果
            for i, det in enumerate(res["detections"][:5]):
                print(f"  {i+1}. '{det['label']}' (置信度: {det['confidence']:.2f})")
                
            if detection_count > 5:
                print(f"  ... 还有{detection_count-5}个检测结果未显示")
        
        # 保存带有边界框的图像
        if "edited_image" in res:
            try:
                image = base64_to_pil(res["edited_image"])
                output_path = os.path.join(args.output_dir, "GroundingDINO_result.png")
                image.save(output_path)
                print(f"✅ 已成功保存标注图像至: {output_path}")
            except Exception as e:
                print(f"❌ 保存标注图像时出错: {e}")
        else:
            print("❌ 响应中没有包含标注图像")
    else:
        print(f"请求失败: {response.text}")

def test_crop(args):
    """测试Crop工具"""
    print("\n====== 测试Crop工具 ======")
    model_name = 'Crop'
    
    worker_addr = get_worker_address(args.controller_addr, model_name)
    if not worker_addr:
        return
    
    img = load_image(args.image_path)
    img_arg = encode(img)
    
    # 测试用例：先测试绝对坐标，再测试归一化坐标
    test_cases = [
        {
            "name": "绝对坐标",
            "coords": "[100, 100, 400, 300]",
            "output_file": "crop_result_absolute.jpg"
        },
        {
            "name": "归一化坐标",
            "coords": "[0.2, 0.2, 0.8, 0.8]",
            "output_file": "crop_result_normalized.jpg"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n------ 测试{test_case['name']} ------")
        crop_coordinates = test_case["coords"]
        
        datas = {
            "image": img_arg,
            "param": crop_coordinates
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
            print(f"裁剪工具状态: {res.get('status', 'unknown')}")
            
            if "message" in res:
                print(f"消息: {res['message']}")
                
            if "edited_image" in res:
                image = base64_to_pil(res["edited_image"])
                output_path = os.path.join(args.output_dir, test_case["output_file"])
                image.save(output_path)
                print(f"裁剪后的图像已保存至: {output_path}")
                
                if "image_dimensions_pixels" in res:
                    width = res["image_dimensions_pixels"]["width"]
                    height = res["image_dimensions_pixels"]["height"]
                    print(f"裁剪后图像尺寸: {width}x{height}")
            else:
                if "error" in res:
                    print(f"处理错误: {res['error']}")
        else:
            print(f"请求失败: {response.text}")

def test_select_subplot(args):
    """测试ZoomInSubfigure工具"""
    print("\n====== 测试ZoomInSubfigure工具 ======")
    model_name = 'ZoomInSubfigure'
    
    # 检查环境变量中是否设置了Google API密钥
    if "GOOGLE_API_KEY" not in os.environ:
        print("警告: 未设置GOOGLE_API_KEY环境变量，测试可能会失败")
        print("请使用以下命令设置API密钥: export GOOGLE_API_KEY='your-api-key'")
    
    worker_addr = get_worker_address(args.controller_addr, model_name)
    if not worker_addr:
        return
    
    # 使用多子图图像进行测试
    subplot_image_path = args.subplot_image_path if hasattr(args, 'subplot_image_path') and args.subplot_image_path else args.image_path
    img = load_image(subplot_image_path)
    img_arg = encode(img)
    
    # 测试用例：选择特定子图
    datas = {
        "model": model_name,
        "param": "柱状图部分",  # 描述要选择的子图
        "image": img_arg,
    }
    
    print(f"正在选择子图: '{datas['param']}'")
    
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
        print(f"选择子图工具状态: {res.get('status', '未知')}")
        
        if "message" in res:
            print(f"消息: {res['message']}")
            
        if "subplot_images" in res and res["subplot_images"]:
            print(f"找到 {len(res['subplot_images'])} 个匹配的子图")
            
            # 保存找到的子图
            for i, subplot_base64 in enumerate(res["subplot_images"]):
                try:
                    image = base64_to_pil(subplot_base64)
                    output_path = os.path.join(args.output_dir, f"subplot_result_{i+1}.png")
                    image.save(output_path)
                    print(f"子图 {i+1} 已保存至: {output_path}")
                except Exception as e:
                    print(f"保存子图 {i+1} 时出错: {e}")
        else:
            if "error" in res:
                print(f"处理错误: {res['error']}")
    else:
        print(f"请求失败: {response.text}")

def main():
    parser = argparse.ArgumentParser(description="一键测试多种视觉工具")
    
    # 通用参数
    parser.add_argument(
        "--controller_addr", type=str, default="http://SH-IDC1-10-140-37-6:21112",
        help="控制器地址"
    )
    parser.add_argument(
        "--image-path", type=str, default="/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/online_workers/test_cases/worker_tests/input_cases/subplot_0.png",
        help="测试图像路径"
    )
    parser.add_argument(
        "--subplot-image-path", type=str, default="/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/online_workers/test_cases/worker_tests/input_cases/subplot_0.png",
        help="用于测试ZoomInSubfigure的多子图图像路径"
    )
    parser.add_argument(
        "--output-dir", type=str, default="/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/online_workers/test_cases/worker_tests/test_results",
        help="输出目录"
    )
    parser.add_argument(
        "--tools", type=str, nargs="+", choices=AVAILABLE_TOOLS + ["all"],
        default=["all"], help="要测试的工具列表，可选: " + ", ".join(AVAILABLE_TOOLS)
    )
    parser.add_argument(
        "--api-key", type=str, default="AIzaSyAffVgu6q5RxZWEv7GLPYsTwZrIhbKfZng",
        help="Google API密钥，用于ZoomInSubfigure工具测试"
    )
    
    args = parser.parse_args()
    
    # 检查并创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 如果提供了API密钥，设置环境变量
    if args.api_key:
        os.environ["GOOGLE_API_KEY"] = args.api_key
        print(f"已设置GOOGLE_API_KEY环境变量")
    
    # 确定要测试的工具
    tools_to_test = AVAILABLE_TOOLS if "all" in args.tools else args.tools

    # tools_to_test = ["OCR","GroundingDINO"]  # 默认测试ZoomInSubfigure工具
    
    image_dict = {
        "default":"/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/online_workers/test_cases/worker_tests/input_cases/subplot_0.png",
        "GroundingDINO": "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/online_workers/test_cases/worker_tests/input_cases/truck.jpg"
    }
    
    # 运行测试
    for tool in tools_to_test:
        if tool == "Point":
            test_point(args)
        # 注释掉旧的垂直线和水平线测试工具
        # elif tool == "draw_vertical_line":
        #     test_draw_vertical_line(args)
        # elif tool == "draw_horizontal_line":
        #     test_draw_horizontal_line(args)
        elif tool == "draw_line":  # 添加新的测试函数
            test_draw_line(args)
        elif tool == "OCR":
            test_ocr(args)
        elif tool == "SegmentRegionAroundPoint":
            test_segment_region(args)
        elif tool == "GroundingDINO":
            args.image_path = image_dict.get(tool,image_dict["default"])
            test_GroundingDINO(args)
        elif tool == "Crop":
            test_crop(args)
        elif tool == "ZoomInSubfigure":
            test_select_subplot(args)
    
    print("\n====== 所有测试完成 ======")

if __name__ == "__main__":
    main()