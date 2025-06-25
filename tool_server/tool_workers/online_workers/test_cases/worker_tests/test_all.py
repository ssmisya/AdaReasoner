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
    "DrawLine",  # 更新工具名称
    "OCR",
    "SegmentRegionAroundPoint",
    "GroundingDINO",
    "Crop",
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
    
    worker_addr = get_worker_address(args.controller_addr, model_name)
    if not worker_addr:
        return
        
    img = load_image(args.image_path)
    img_arg = encode(img)
    
    # 根据新的描述创建测试用例
    test_cases = [
        {
            "name": "具体物体",
            "description": "the zebra's nose",
            "output_file": "point_result_zebra_nose.png"
        },
        {
            "name": "中心点",
            "description": "center of the image",
            "output_file": "point_result_image_center.png"
        },
        {
            "name": "相对位置",
            "description": "the rightmost zebra",
            "output_file": "point_result_rightmost_zebra.png"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n------ 测试Point: {test_case['name']} ------")
        print(f"描述: {test_case['description']}")
        
        # 使用新的参数格式
        datas = {
            "image": img_arg,
            "description": test_case["description"]
        }
        
        tic = time.time()
        try:
            response = requests.post(
                worker_addr + "/worker_generate",
                headers={"User-Agent": "FastChat Client"},
                json=datas,
                timeout=60  # 添加超时设置
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
                    points = res["points"]
                    print(f"检测到的点: {points}")
                    # 打印更详细的点信息
                    if isinstance(points, list) and len(points) > 0:
                        for i, point in enumerate(points):
                            print(f"  点 {i+1}: x={point.get('x', 'N/A')}, y={point.get('y', 'N/A')}")
                    
                if "raw_response" in res:
                    print(f"原始响应: {res['raw_response']}")
                
                # 保存返回的图像
                image_saved = False
                
                # 检查不同可能的图像字段名称
                image_field_names = ["image_with_points", "edited_image", "annotated_image"]
                for field_name in image_field_names:
                    if field_name in res and res[field_name]:
                        try:
                            print(f"找到图像字段: {field_name}")
                            image = base64_to_pil(res[field_name])
                            output_path = os.path.join(args.output_dir, test_case["output_file"])
                            image.save(output_path, format="PNG")
                            print(f"✅ 结果图像已成功保存至: {output_path}")
                            image_saved = True
                            break
                        except Exception as e:
                            print(f"❌ 保存图像时出错 ({field_name}): {str(e)}")
                
                if not image_saved:
                    print("❌ 响应中没有找到可用的图像数据")
                    # 将完整响应保存到JSON文件以便调试
                    debug_path = os.path.join(args.output_dir, f"point_debug_{test_case['name'].replace(' ', '_')}.json")
                    with open(debug_path, 'w', encoding='utf-8') as f:
                        json.dump(res, f, indent=2, ensure_ascii=False)
                    print(f"已将完整响应保存至: {debug_path}")
            else:
                print(f"❌ 请求失败: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"❌ 请求超时")
        except requests.exceptions.ConnectionError:
            print(f"❌ 连接错误")
        except Exception as e:
            print(f"❌ 发生异常: {str(e)}")
            import traceback
            traceback.print_exc()

def test_draw_line(args):
    """测试DrawLine工具（支持水平线和垂直线）"""
    print("\n====== 测试DrawLine工具 ======")
    model_name = 'DrawLine'
    
    worker_addr = get_worker_address(args.controller_addr, model_name)
    if not worker_addr:
        return
    
    img = load_image(args.image_path)
    img_arg = encode(img)
    
    # 根据新的描述创建测试用例
    test_cases = [
        {
            "name": "单水平线",
            "line_type": "horizontal",
            "description": "y1=100",
            "output_file": "horizontal_line_single.jpg"
        },
        {
            "name": "多水平线",
            "line_type": "horizontal",
            "description": "y1=100, y2=200, y3=300",
            "output_file": "horizontal_line_multiple.jpg"
        },
        {
            "name": "单垂直线",
            "line_type": "vertical",
            "description": "x1=150",
            "output_file": "vertical_line_single.jpg"
        },
        {
            "name": "多垂直线",
            "line_type": "vertical",
            "description": "x1=100, x2=200, x3=300",
            "output_file": "vertical_line_multiple.jpg"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n------ 测试{test_case['name']} ------")
        
        # 使用新的参数格式
        datas = {
            "image": img_arg,
            "line_type": test_case["line_type"],
            "description": test_case["description"]
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
    
    worker_addr = get_worker_address(args.controller_addr, model_name)
    if not worker_addr:
        return
        
    img = load_image(args.image_path)
    img_arg = encode(img)
    
    # OCR只需要image参数
    datas = {
        "image": img_arg
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
                
            # 保存带有文本框的图像（如果有）
            if "edited_image" in res:
                image = base64_to_pil(res["edited_image"])
                output_path = os.path.join(args.output_dir, "ocr_result.png")
                image.save(output_path)
                print(f"带文本框的图像已保存至: {output_path}")
        else:
            print(f"OCR工具完整响应: {json.dumps(res, indent=2)}")
    else:
        print(f"请求失败: {response.text}")

def test_segment_region(args):
    """测试SegmentRegionAroundPoint工具"""
    print("\n====== 测试SegmentRegionAroundPoint工具 ======")
    model_name = 'SegmentRegionAroundPoint'
    
    worker_addr = get_worker_address(args.controller_addr, model_name)
    if not worker_addr:
        return
    
    img = load_image(args.image_path)
    img_arg = encode(img)
    
    # 根据新的描述创建测试用例
    test_cases = [
        {
            "name": "指定点坐标",
            "description": "x=300, y=100",
            "output_file": "segment_result_with_point.jpg"
        },
        {
            "name": "自动分割全图",
            "description": "",  # 空字符串表示不提供点坐标，自动分割
            "output_file": "segment_result_auto.jpg"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n------ 测试{test_case['name']} ------")
        
        # 使用新的参数格式
        datas = {
            "image": img_arg
        }
        
        # 仅当有描述时添加description参数
        if test_case["description"]:
            datas["description"] = test_case["description"]
        
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
            print(f"分割区域工具状态: {res.get('status', 'unknown')}")
            
            # 显示分割模式信息（如果有）
            if "segmentation_mode" in res:
                print(f"分割模式: {res['segmentation_mode']}")
                if "mask_count" in res:
                    print(f"掩码数量: {res['mask_count']}")
                elif "point_count" in res:
                    print(f"点数量: {res['point_count']}")
            
            if "edited_image" in res and res.get("status") == "success":
                image = base64_to_pil(res["edited_image"])
                output_path = os.path.join(args.output_dir, test_case["output_file"])
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
    
    # 根据新的描述创建测试用例
    test_cases = [
        {
            "name": "简单物体",
            "description": "car",
            "output_file": "grounding_dino_car.png"
        },
        {
            "name": "带属性物体",
            "description": "a red car",
            "output_file": "grounding_dino_red_car.png"
        },
        {
            "name": "复杂场景",
            "description": "a man holding a dog",
            "output_file": "grounding_dino_man_dog.png"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n------ 测试{test_case['name']} ------")
        
        # 使用新的参数格式
        datas = {
            "image": img_arg,
            "description": test_case["description"]
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
                    print(f"  bbox: {det['bbox']}")

                    
                if detection_count > 5:
                    print(f"  ... 还有{detection_count-5}个检测结果未显示")
            
            # 保存带有边界框的图像
            if "edited_image" in res:
                try:
                    image = base64_to_pil(res["edited_image"])
                    output_path = os.path.join(args.output_dir, test_case["output_file"])
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
            "description": crop_coordinates
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
        "--output-dir", type=str, default="/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/online_workers/test_cases/worker_tests/test_results",
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
    
    # 为不同工具设置合适的测试图像
    image_dict = {
        "default":"/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/online_workers/test_cases/worker_tests/input_cases/subplot_0.png",
        "GroundingDINO": "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/online_workers/test_cases/worker_tests/input_cases/truck.jpg",
        "SegmentRegionAroundPoint": "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/online_workers/test_cases/worker_tests/input_cases/zebra.jpg",
        "Point": "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/online_workers/test_cases/worker_tests/input_cases/zebra.jpg"
    }
    
    # 运行测试
    for tool in tools_to_test:
        # 为每个工具设置合适的测试图像
        args.image_path = image_dict.get(tool, image_dict["default"])
        print(f"使用图像: {args.image_path}")
        
        if tool == "Point":
            test_point(args)
        elif tool == "DrawLine":  # 更新工具名称
            test_draw_line(args)
        elif tool == "OCR":
            test_ocr(args)
        elif tool == "SegmentRegionAroundPoint":
            test_segment_region(args)
        elif tool == "GroundingDINO":
            test_GroundingDINO(args)
        elif tool == "Crop":
            test_crop(args)
    
    print("\n====== 所有测试完成 ======")

if __name__ == "__main__":
    main()