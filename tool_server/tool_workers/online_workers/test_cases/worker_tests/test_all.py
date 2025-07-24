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
    "DrawShape",  # 添加新工具
    "HighlightBox",  # 添加新工具
    "MaskBox",  # 添加新工具
    "GetSubplotInfo",  # 添加子图信息提取工具
    "GetBarInfo",  # 添加柱状图信息提取工具
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
            "coordinates": "y1=100",
            "output_file": "horizontal_line_single.jpg"
        },
        {
            "name": "多水平线",
            "line_type": "horizontal",
            "coordinates": "y1=100, y2=200, y3=300",
            "output_file": "horizontal_line_multiple.jpg"
        },
        {
            "name": "单垂直线",
            "line_type": "vertical",
            "coordinates": "x1=150",
            "output_file": "vertical_line_single.jpg"
        },
        {
            "name": "多垂直线",
            "line_type": "vertical",
            "coordinates": "x1=100, x2=200, x3=300",
            "output_file": "vertical_line_multiple.jpg"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n------ 测试{test_case['name']} ------")
        
        # 使用新的参数格式
        datas = {
            "image": img_arg,
            "line_type": test_case["line_type"],
            "coordinates": test_case["coordinates"]
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
            "coordinates": "x=300, y=100",
            "output_file": "segment_result_with_point.jpg"
        },
        {
            "name": "自动分割全图",
            "coordinates": "",  # 空字符串表示不提供点坐标，自动分割
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
        if test_case["coordinates"]:
            datas["coordinates"] = test_case["coordinates"]
        
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
            "coordinates": "[100, 100, 400, 300]",
            "output_file": "crop_result_absolute.jpg"
        },
        # {
        #     "name": "归一化坐标",
        #     "coordinates": "[0.2, 0.2, 0.8, 0.8]",
        #     "output_file": "crop_result_normalized.jpg"
        # }
    ]
    
    for test_case in test_cases:
        print(f"\n------ 测试{test_case['name']} ------")
        crop_coordinates = test_case["coordinates"]
        
        datas = {
            "image": img_arg,
            "bbox": crop_coordinates # 可以是bbox，可以是coordinates
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

def test_draw_shape(args):
    """测试DrawShape工具"""
    print("\n====== 测试DrawShape工具 ======")
    model_name = 'DrawShape'
    
    worker_addr = get_worker_address(args.controller_addr, model_name)
    if not worker_addr:
        return
    
    img = load_image(args.image_path)
    img_arg = encode(img)
    
    # 测试用例：绘制不同形状
    test_cases = [
        {
            "name": "矩形",
            "bboxes": [
                {"shape": "rectangle", "coords": [50, 50, 200, 150]}
            ],
            "output_file": "draw_shape_rectangle.jpg"
        },
        {
            "name": "椭圆",
            "bboxes": [
                {"shape": "ellipse", "coords": [250, 100, 400, 200]}
            ],
            "output_file": "draw_shape_ellipse.jpg"
        },
        {
            "name": "圆形",
            "bboxes": [
                {"shape": "circle", "coords": [150, 250, 250, 350]}
            ],
            "output_file": "draw_shape_circle.jpg"
        },
        {
            "name": "多形状",
            "bboxes": [
                {"shape": "rectangle", "coords": [50, 50, 150, 100]},
                {"shape": "ellipse", "coords": [200, 100, 300, 200]},
                {"shape": "circle", "coords": [350, 250, 450, 350]}
            ],
            "output_file": "draw_shape_multiple.jpg"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n------ 测试DrawShape: {test_case['name']} ------")
        
        datas = {
            "image": img_arg,
            "bboxes": test_case["bboxes"]
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
            print(f"DrawShape工具状态: {res.get('status', 'unknown')}")
            
            if "message" in res:
                print(f"消息: {res['message']}")
                
            if "edited_image" in res:
                image = base64_to_pil(res["edited_image"])
                output_path = os.path.join(args.output_dir, test_case["output_file"])
                image.save(output_path)
                print(f"结果图像已保存至: {output_path}")
                
                if "image_dimensions_pixels" in res:
                    width = res["image_dimensions_pixels"]["width"]
                    height = res["image_dimensions_pixels"]["height"]
                    print(f"图像尺寸: {width}x{height}")
            else:
                if "error" in res:
                    print(f"处理错误: {res['error']}")
        else:
            print(f"请求失败: {response.text}")

def test_highlight_box(args):
    """测试HighlightBox工具"""
    print("\n====== 测试HighlightBox工具 ======")
    model_name = 'HighlightBox'
    
    worker_addr = get_worker_address(args.controller_addr, model_name)
    if not worker_addr:
        return
    
    img = load_image(args.image_path)
    img_arg = encode(img)
    
    # 测试用例
    test_cases = [
        {
            "name": "单个区域",
            "bboxes": [[50, 50, 200, 150]],
            "output_file": "highlight_box_single.jpg"
        },
        {
            "name": "多个区域",
            "bboxes": [[50, 50, 150, 100], [200, 150, 300, 250], [350, 200, 450, 300]],
            "output_file": "highlight_box_multiple.jpg"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n------ 测试HighlightBox: {test_case['name']} ------")
        
        datas = {
            "image": img_arg,
            "bboxes": test_case["bboxes"]
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
            print(f"HighlightBox工具状态: {res.get('status', 'unknown')}")
            
            if "message" in res:
                print(f"消息: {res['message']}")
                
            if "edited_image" in res:
                image = base64_to_pil(res["edited_image"])
                output_path = os.path.join(args.output_dir, test_case["output_file"])
                image.save(output_path)
                print(f"结果图像已保存至: {output_path}")
                
                if "image_dimensions_pixels" in res:
                    width = res["image_dimensions_pixels"]["width"]
                    height = res["image_dimensions_pixels"]["height"]
                    print(f"图像尺寸: {width}x{height}")
            else:
                if "error" in res:
                    print(f"处理错误: {res['error']}")
        else:
            print(f"请求失败: {response.text}")

def test_mask_box(args):
    """测试MaskBox工具"""
    print("\n====== 测试MaskBox工具 ======")
    model_name = 'MaskBox'
    
    worker_addr = get_worker_address(args.controller_addr, model_name)
    if not worker_addr:
        return
    
    img = load_image(args.image_path)
    img_arg = encode(img)
    
    # 测试用例
    test_cases = [
        {
            "name": "单个区域",
            "bboxes": [[100, 100, 300, 200]],
            "output_file": "mask_box_single.jpg"
        },
        {
            "name": "多个区域",
            "bboxes": [[50, 50, 150, 150], [200, 100, 350, 200], [400, 200, 500, 300]],
            "output_file": "mask_box_multiple.jpg"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n------ 测试MaskBox: {test_case['name']} ------")
        
        datas = {
            "image": img_arg,
            "bboxes": test_case["bboxes"]
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
            print(f"MaskBox工具状态: {res.get('status', 'unknown')}")
            
            if "message" in res:
                print(f"消息: {res['message']}")
                
            if "edited_image" in res:
                image = base64_to_pil(res["edited_image"])
                output_path = os.path.join(args.output_dir, test_case["output_file"])
                image.save(output_path)
                print(f"结果图像已保存至: {output_path}")
                
                if "image_dimensions_pixels" in res:
                    width = res["image_dimensions_pixels"]["width"]
                    height = res["image_dimensions_pixels"]["height"]
                    print(f"图像尺寸: {width}x{height}")
            else:
                if "error" in res:
                    print(f"处理错误: {res['error']}")
        else:
            print(f"请求失败: {response.text}")

def test_language_model(args):
    """测试LanguageModel工具"""
    print("\n====== 测试LanguageModel工具 ======")
    model_name = 'LanguageModel'
    
    worker_addr = get_worker_address(args.controller_addr, model_name)
    if not worker_addr:
        return
    
    img = load_image(args.image_path)
    img_arg = encode(img)
    
    prompt1 = '''
You are an expert in scientific image analysis. Your task is to carefully observe the provided image and identify all subplots.
For each subplot, you must extract its full title and determine the precise pixel coordinates of its bounding box.
**Key features to identify a subplot:**
1.  **Title Pattern:** A subplot almost always has a title. Look for labels that start with an enumeration like `(a)`, `a)`, `(b)`, `b)`, etc., often followed by a descriptive text. The entire string (e.g., "(a) Temperature over time") should be treated as the title.
2.  **Graphical Area:** The coordinates should represent the bounding box that tightly encloses the main graphical content of the subplot (e.g., the plot area with axes, data points, lines, etc.). This bounding box should generally *exclude* the title itself, which is typically located just above or beside the plot.
**Output Format:**
Return ONLY a valid JSON dictionary where each key is the full title of a subplot and its value is a list of four integers representing the bounding box coordinates: `[x_min, y_min, x_max, y_max]`. The origin `(0, 0)` is the top-left corner of the image.
Example format:
`{{"(a) Title of first plot": [x1, y1, x2, y2], "(b) Title of second plot": [x3, y3, x4, y4]}}`
'''

    # 测试用例
    test_cases = [
        {
            "name": "中文prompt",
            "prompt": "这张图片中一共有几个子图，各个子图的标题和坐标是什么,坐标是指在图片中的坐标，格式为(x1,y1,x2,y2)",
            "output_file": "language_model_description.txt"
        },
        {
            "name": "英文prompt",
            "prompt": prompt1,
            "output_file": "language_model_analysis.txt"
        },
        # {
        #     "name": "问答",
        #     "prompt": "这张图片中有什么物体？它们在做什么？",
        #     "output_file": "language_model_qa.txt"
        # }
    ]
    
    for test_case in test_cases:
        print(f"\n------ 测试LanguageModel: {test_case['name']} ------")
        print(f"提示词: {test_case['prompt']}")

        # print("testall中的language model 的image: ", img_arg)
        
        datas = {
            "image": img_arg,
            "prompt": test_case["prompt"]
        }
        
        tic = time.time()
        try:
            response = requests.post(
                worker_addr + "/worker_generate",
                headers={"User-Agent": "FastChat Client"},
                json=datas,
                timeout=120  # 语言模型可能需要更长时间
            )
            toc = time.time()
            
            print(f"耗时: {toc - tic:.3f}s")
            print(f"返回状态: {response.status_code}")
            
            if response.status_code == 200:
                res = response.json()
                print(f"LanguageModel工具状态: {res.get('status', 'unknown')}")
                
                if "message" in res:
                    print(f"消息: {res['message']}")
                
                if "response" in res:
                    # 保存文本响应
                    # output_path = os.path.join(args.output_dir, test_case["output_file"])
                    # with open(output_path, 'w', encoding='utf-8') as f:
                    #     f.write(res["response"])
                    # print(f"✅ 模型响应已保存至: {output_path}")
                    
                    # 显示响应的前100个字符
                    # preview = res["response"][:100] + "..." if len(res["response"]) > 100 else res["response"]
                    preview = res["response"]
                    print(f"响应预览: {preview}")
                else:
                    print("❌ 响应中没有找到文本内容")
                    # 保存完整响应以便调试
                    # debug_path = os.path.join(args.output_dir, f"language_model_debug_{test_case['name'].replace(' ', '_')}.json")
                    # with open(debug_path, 'w', encoding='utf-8') as f:
                    #     json.dump(res, f, indent=2, ensure_ascii=False)
                    # print(f"已将完整响应保存至: {debug_path}")
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

def test_get_subplot_info(args):
    """测试GetSubplotInfo工具"""
    print("\n====== 测试GetSubplotInfo工具 ======")
    model_name = 'GetSubplotInfo'
    
    worker_addr = get_worker_address(args.controller_addr, model_name)
    if not worker_addr:
        return
    
    # 使用带有子图的图像进行测试
    img = load_image(args.image_path)
    img_arg = encode(img)
    
    # GetSubplotInfo只需要image参数
    datas = {
        "image": img_arg
    }

    # print("testall中的getsubplotinfo 的image: ", img_arg)
    
    tic = time.time()
    response = requests.post(
        worker_addr + "/worker_generate",
        headers={"User-Agent": "FastChat Client"},
        json=datas,
        timeout=60  # 设置超时时间
    )
    toc = time.time()
    
    print(f"耗时: {toc - tic:.3f}s")
    print(f"返回状态: {response.status_code}")
    
    if response.status_code == 200:
        res = response.json()
        
        print(f"GetSubplotInfo工具响应状态: {res.get('status', 'unknown')}")
        
        if "subplots" in res:
            subplots = res["subplots"]
            subplot_count = len(subplots)
            print(f"这是系统文本----检测到{subplot_count}个子图:")
            
            # 显示子图信息
            for title, bbox in list(subplots.items())[:5]:  # 最多显示5个
                print(f"  '{title}': {bbox}")
                
            if subplot_count > 5:
                print(f"  ... 还有{subplot_count-5}个子图未显示")
            
            # # 将结果保存为JSON文件
            # output_json_path = os.path.join(args.output_dir, "subplot_info_result.json")
            # with open(output_json_path, 'w', encoding='utf-8') as f:
            #     json.dump(subplots, f, indent=2, ensure_ascii=False)
            # print(f"子图信息已保存至: {output_json_path}")
            
            # 可视化子图边界框
            try:
                # 创建一个副本用于绘制
                img_with_boxes = img.copy()
                draw = ImageDraw.Draw(img_with_boxes)
                
                # 为不同的子图使用不同颜色
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                          (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                
                for i, (title, bbox) in enumerate(subplots.items()):
                    color = colors[i % len(colors)]
                    # 绘制边界框
                    draw.rectangle(bbox, outline=color, width=3)
                    # 绘制标题
                    draw.text((bbox[0], bbox[1]-20), title, fill=color)
                
                # 保存可视化结果
                output_image_path = os.path.join(args.output_dir, "subplot_info_visualization.png")
                img_with_boxes.save(output_image_path)
                print(f"子图可视化结果已保存至: {output_image_path}")
            except Exception as e:
                print(f"可视化子图时出错: {str(e)}")
        else:
            print(f"GetSubplotInfo工具完整响应: {json.dumps(res, indent=2)}")
    else:
        print(f"请求失败: {response.text}")

def test_get_bar_info(args):
    """测试GetBarInfo工具"""
    print("\n====== 测试GetBarInfo工具 ======")
    model_name = 'GetBarInfo'
    
    worker_addr = get_worker_address(args.controller_addr, model_name)
    if not worker_addr:
        return
    
    # 使用带有柱状图的图像进行测试
    img = load_image(args.image_path)
    img_arg = encode(img)
    
    # GetBarInfo只需要image参数
    datas = {
        "image": img_arg
    }
    
    tic = time.time()
    response = requests.post(
        worker_addr + "/worker_generate",
        headers={"User-Agent": "FastChat Client"},
        json=datas,
        timeout=60  # 设置超时时间
    )
    toc = time.time()
    
    print(f"耗时: {toc - tic:.3f}s")
    print(f"返回状态: {response.status_code}")
    
    if response.status_code == 200:
        res = response.json()
        
        print(f"GetBarInfo工具响应状态: {res.get('status', 'unknown')}")
        
        if "bars" in res:
            bars = res["bars"]
            bar_count = len(bars)
            print(f"检测到{bar_count}个柱子:")
            
            # 显示柱子信息
            for label, bbox in list(bars.items())[:5]:  # 最多显示5个
                print(f"  '{label}': {bbox}")
                
            if bar_count > 5:
                print(f"  ... 还有{bar_count-5}个柱子未显示")
            
            # 将结果保存为JSON文件
            output_json_path = os.path.join(args.output_dir, "bar_info_result.json")
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(bars, f, indent=2, ensure_ascii=False)
            print(f"柱状图信息已保存至: {output_json_path}")
            
            # 可视化柱子边界框
            try:
                # 创建一个副本用于绘制
                img_with_boxes = img.copy()
                draw = ImageDraw.Draw(img_with_boxes)
                
                # 为不同的柱子使用不同颜色
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                          (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                
                for i, (label, bbox) in enumerate(bars.items()):
                    color = colors[i % len(colors)]
                    # 绘制边界框
                    draw.rectangle(bbox, outline=color, width=3)
                    # 绘制标签
                    draw.text((bbox[0], bbox[1]-20), label, fill=color)
                
                # 保存可视化结果
                output_image_path = os.path.join(args.output_dir, "bar_info_visualization.png")
                img_with_boxes.save(output_image_path)
                print(f"柱状图可视化结果已保存至: {output_image_path}")
            except Exception as e:
                print(f"可视化柱状图时出错: {str(e)}")
        else:
            print(f"GetBarInfo工具完整响应: {json.dumps(res, indent=2)}")
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

    # tools_to_test = ["OCR", "DrawShape","Crop","Point","GroundingDINO","SegmentRegionAroundPoint","DrawLine","HighlightBox","MaskBox"]
    tools_to_test = ["Crop"]
    
    # 为不同工具设置合适的测试图像
    image_dict = {
        "default":"./input_cases/subplot_0.png",
        "GroundingDINO": "./input_cases/truck.jpg",
        "SegmentRegionAroundPoint": "./input_cases/zebra.jpg",
        "Point": "./input_cases/zebra.jpg",
        "LanguageModel": "./input_cases/subplots1.jpg",
        "GetSubplotInfo": "./input_cases/subplots2.jpg",
        "GetBarInfo": "./input_cases/bars1.jpg"
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
        elif tool == "DrawShape":  # 添加新工具
            test_draw_shape(args)
        elif tool == "HighlightBox":  # 添加新工具
            test_highlight_box(args)
        elif tool == "MaskBox":  # 添加新工具
            test_mask_box(args)
        elif tool == "LanguageModel":  # 添加语言模型工具
            test_language_model(args)
        elif tool == "GetSubplotInfo":  # 添加子图信息提取工具
            test_get_subplot_info(args)
        elif tool == "GetBarInfo":  # 添加柱状图信息提取工具
            test_get_bar_info(args)
    
    print("\n====== 所有测试完成 ======")

if __name__ == "__main__":
    main()