import argparse
import json
import time
import os
import sys
import requests
import base64
import asyncio
import concurrent.futures
from io import BytesIO
from PIL import Image, ImageDraw
import torchvision.transforms.functional as F
import numpy as np
import cv2

# 尝试导入可能需要的工具函数 - 保留原始文件的兼容性
try:
    from tool_server.utils.utils import *
    from tool_server.utils.server_utils import *
    from tool_server.tool_workers.online_workers.utils import annotate_xyxy
except ImportError:
    print("警告: 部分工具模块导入失败，某些功能可能受限")

# 定义所有可用的工具
AVAILABLE_TOOLS = [
    "Point", 
    "DrawLine", 
    "OCR",
    "SegmentRegionAroundPoint",
    "GroundingDINO",
    "Crop",
    "DrawShape", 
    "HighlightBox", 
    "MaskBox", 
    "LanguageModel", 
    "GetSubplotInfo", 
    "GetBarInfo", 
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
        img = img.resize((new_w, new_h))
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
    print(f"尝试获取 {model_name} 的工作节点地址...")
    
    try:
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        # print(f"可用模型: {models}") # 打印所有模型可能过于冗长
        
        ret = requests.post(
            controller_addr + "/get_worker_address", 
            json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        
        if not worker_addr:
            print(f"没有找到 {model_name} 的可用工作节点")
            return None
            
        print(f"工作节点地址: {worker_addr}")
        return worker_addr
        
    except Exception as e:
        print(f"获取工作节点地址时出错: {str(e)}")
        return None

def send_single_request(worker_addr, data, request_id, tool_name):
    """发送单个请求并处理响应"""
    print(f"\n------ [{tool_name}] 发送请求 {request_id+1}/5 ------")
    
    start_time = time.time()
    result = {
        "id": request_id + 1,
        "tool_name": tool_name,
        "start_time": start_time,
        "status": "未知",
        "duration": None,
        "response_summary": {} # 用于存储关键响应信息
    }
    
    try:
        response = requests.post(
            worker_addr + "/worker_generate",
            headers={"User-Agent": "FastChat Client"},
            json=data,
            timeout=120 # 设置超时时间，对于LLM等可能需要更长
        )
        end_time = time.time()
        duration = end_time - start_time
        
        result["duration"] = duration
        result["end_time"] = end_time
        
        print(f"请求 {request_id+1} 已发送，等待响应...")
        
        if response.status_code == 200:
            res = response.json()
            result["status"] = "成功"
            result["response_status"] = res.get("status", "unknown")
            
            # 根据工具类型提取关键信息
            if tool_name == "OCR" and "detections" in res:
                result["response_summary"]["detection_count"] = len(res["detections"])
                result["response_summary"]["first_detection"] = res["detections"][0]["label"] if res["detections"] else None
            elif tool_name == "GetBarInfo" and "bars" in res:
                result["response_summary"]["bars_count"] = len(res["bars"])
                result["response_summary"]["first_bar_label"] = list(res["bars"].keys())[0] if res["bars"] else None
            elif tool_name == "GetSubplotInfo" and "subplots" in res:
                result["response_summary"]["subplot_count"] = len(res["subplots"])
                result["response_summary"]["first_subplot_title"] = list(res["subplots"].keys())[0] if res["subplots"] else None
            elif tool_name == "LanguageModel" and "response" in res:
                result["response_summary"]["response_length"] = len(res["response"])
                result["response_summary"]["response_preview"] = res["response"][:50] + "..." if len(res["response"]) > 50 else res["response"]
            elif tool_name in ["Point", "GroundingDINO"] and "points" in res: # For Point
                result["response_summary"]["point_count"] = len(res["points"])
            elif tool_name in ["GroundingDINO"] and "detections" in res: # For GroundingDINO
                result["response_summary"]["detection_count"] = len(res["detections"])
            elif tool_name in ["Crop", "DrawLine", "DrawShape", "HighlightBox", "MaskBox"] and "edited_image" in res:
                result["response_summary"]["image_edited"] = True
                result["response_summary"]["image_dimensions"] = res.get("image_dimensions_pixels")

        else:
            result["status"] = "失败"
            result["error"] = response.text
    
    except requests.exceptions.Timeout:
        result["status"] = "超时"
    except Exception as e:
        result["status"] = "错误"
        result["error"] = str(e)
    
    return result

def run_batch_test(args, tool_name, datas_template, num_requests=5, output_filename_prefix="batch_test_results"):
    """
    通用批量测试函数
    :param args: argparse的参数对象
    :param tool_name: 要测试的工具名称
    :param datas_template: 用于生成请求数据的字典模板。如果需要变化的输入，可以修改此函数。
                           对于本例，我们假设所有请求的输入数据是相同的，除了image参数
                           会在每次请求前encode。
    :param num_requests: 请求次数
    :param output_filename_prefix: 输出文件名的前缀
    """
    print(f"\n====== 批量测试 {tool_name} 工具 ======")
    
    worker_addr = get_worker_address(args.controller_addr, tool_name)
    if not worker_addr:
        print(f"无法获取 {tool_name} 的工作节点地址，跳过测试。")
        return

    # 加载图像并在每次请求前编码，以确保数据新鲜（虽然这里每次都是同一张图）
    img = load_image(args.image_path)
    img_arg = encode(img)

    # 准备每个请求的数据，如果数据是动态的，这里可以是一个列表或生成器
    # 对于本例，我们假设图像和参数是固定的
    datas = datas_template.copy()
    datas["image"] = img_arg

    # 开始批量请求的计时
    batch_start_time = time.time()
    print(f"开始发送批量请求，共 {num_requests} 个请求，每个请求间隔1秒发送...")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        future_to_id = {}
        for i in range(num_requests):
            if i > 0:
                time.sleep(1) # 每个请求间隔1秒发送
            future = executor.submit(send_single_request, worker_addr, datas, i, tool_name)
            future_to_id[future] = i
            print(f"已提交请求 {i+1} for {tool_name}")
        
        # 收集所有结果
        for future in concurrent.futures.as_completed(future_to_id):
            request_id = future_to_id[future]
            try:
                result = future.result()
                results.append(result)
                print(f"收到请求 {request_id+1} 的响应 for {tool_name}")
            except Exception as e:
                print(f"处理请求 {request_id+1} for {tool_name} 时出错: {str(e)}")
                results.append({
                    "id": request_id + 1,
                    "tool_name": tool_name,
                    "status": "错误",
                    "error": str(e)
                })
    
    # 计算总时间
    batch_end_time = time.time()
    total_time = batch_end_time - batch_start_time
    
    # 对结果进行排序，按请求ID
    results.sort(key=lambda x: x["id"])
    
    # 统计和展示结果
    print(f"\n====== {tool_name} 批量请求统计 ======")
    print(f"总耗时: {total_time:.3f}秒")
    print(f"请求次数: {num_requests}")
    
    success_results = [r for r in results if r["status"] == "成功" and r["duration"] is not None]
    if success_results:
        success_times = [r["duration"] for r in success_results]
        print(f"成功请求数: {len(success_results)}")
        print(f"平均请求时间: {sum(success_times)/len(success_times):.3f}秒")
        print(f"最长请求时间: {max(success_times):.3f}秒")
        print(f"最短请求时间: {min(success_times):.3f}秒")
    else:
        print("所有请求均失败")
    
    print("\n各请求详细结果:")
    for result in results:
        req_id = result["id"]
        status = result["status"]
        tool_name = result["tool_name"] # 确保打印的工具名称正确
        
        if status == "成功":
            duration = result["duration"]
            summary = result["response_summary"]
            print(f"  请求 {req_id} ({tool_name}): {status}, 耗时: {duration:.3f}秒, 摘要: {summary}")
        else:
            error = result.get("error", "未知错误")
            print(f"  请求 {req_id} ({tool_name}): {status}, 错误: {error}")
    
    # 保存结果到JSON文件
    output_json_path = os.path.join(args.output_dir, f"{output_filename_prefix}_{tool_name.lower()}_results.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {output_json_path}")

## 各工具的批量测试函数
# 这些函数主要负责为 run_batch_test 准备 datas_template 和 tool_name

def test_point_batch(args):
    datas_template = {"description": "the zebra's nose"}
    run_batch_test(args, "Point", datas_template, output_filename_prefix="batch_point")

def test_draw_line_batch(args):
    datas_template = {"line_type": "horizontal", "coordinates": "y1=100"}
    run_batch_test(args, "DrawLine", datas_template, output_filename_prefix="batch_draw_line")

def test_ocr_batch(args):
    datas_template = {} # OCR只需要image
    run_batch_test(args, "OCR", datas_template, output_filename_prefix="batch_ocr")

def test_segment_region_around_point_batch(args):
    datas_template = {"coordinates": "x=300, y=100"}
    run_batch_test(args, "SegmentRegionAroundPoint", datas_template, output_filename_prefix="batch_segment_region")

def test_grounding_dino_batch(args):
    datas_template = {"description": "a red car"}
    run_batch_test(args, "GroundingDINO", datas_template, output_filename_prefix="batch_grounding_dino")

def test_crop_batch(args):
    datas_template = {"coordinates": "[100, 100, 400, 300]"}
    run_batch_test(args, "Crop", datas_template, output_filename_prefix="batch_crop")

def test_draw_shape_batch(args):
    datas_template = {"bboxes": [{"shape": "rectangle", "coords": [50, 50, 200, 150]}]}
    run_batch_test(args, "DrawShape", datas_template, output_filename_prefix="batch_draw_shape")

def test_highlight_box_batch(args):
    datas_template = {"bboxes": [[50, 50, 200, 150]]}
    run_batch_test(args, "HighlightBox", datas_template, output_filename_prefix="batch_highlight_box")

def test_mask_box_batch(args):
    datas_template = {"bboxes": [[100, 100, 300, 200]]}
    run_batch_test(args, "MaskBox", datas_template, output_filename_prefix="batch_mask_box")

def test_language_model_batch(args):
    # 使用一个中文prompt作为示例
    prompt_cn = "这张图片中一共有几个子图，各个子图的标题和坐标是什么,坐标是指在图片中的坐标，格式为(x1,y1,x2,y2)"
    datas_template = {"prompt": prompt_cn}
    run_batch_test(args, "LanguageModel", datas_template, output_filename_prefix="batch_language_model")

def test_get_subplot_info_batch(args):
    datas_template = {} # GetSubplotInfo只需要image
    run_batch_test(args, "GetSubplotInfo", datas_template, output_filename_prefix="batch_get_subplot_info")

def test_get_bar_info_batch(args):
    datas_template = {} # GetBarInfo只需要image
    run_batch_test(args, "GetBarInfo", datas_template, output_filename_prefix="batch_get_bar_info")


def main():
    parser = argparse.ArgumentParser(description="批量测试多种视觉工具")
    
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
        "--output-dir", type=str, default="/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/online_workers/test_cases/worker_tests/batch_test_results",
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
        "Point": "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/online_workers/test_cases/worker_tests/input_cases/zebra.jpg",
        "LanguageModel": "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/online_workers/test_cases/worker_tests/input_cases/subplots4.jpg",
        "GetSubplotInfo": "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/online_workers/test_cases/worker_tests/input_cases/subplots2.jpg",
        "GetBarInfo": "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/online_workers/test_cases/worker_tests/input_cases/bars3.jpg",
        "OCR": "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/online_workers/test_cases/worker_tests/input_cases/subplot_0.png", # OCR 建议使用包含文本的图像
        "DrawLine": "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/online_workers/test_cases/worker_tests/input_cases/subplot_0.png",
        "Crop": "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/online_workers/test_cases/worker_tests/input_cases/subplot_0.png",
        "DrawShape": "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/online_workers/test_cases/worker_tests/input_cases/subplot_0.png",
        "HighlightBox": "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/online_workers/test_cases/worker_tests/input_cases/subplot_0.png",
        "MaskBox": "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/online_workers/test_cases/worker_tests/input_cases/subplot_0.png",
    }
    
    # 运行测试
    for tool in tools_to_test:
        # 为每个工具设置合适的测试图像
        args.image_path = image_dict.get(tool, image_dict["default"])
        print(f"\n--- 准备对工具 '{tool}' 进行批量测试，使用图像: {args.image_path} ---")
        
        # 调用相应的批量测试函数
        if tool == "Point":
            test_point_batch(args)
        elif tool == "DrawLine": 
            test_draw_line_batch(args)
        elif tool == "OCR":
            test_ocr_batch(args)
        elif tool == "SegmentRegionAroundPoint":
            test_segment_region_around_point_batch(args)
        elif tool == "GroundingDINO":
            test_grounding_dino_batch(args)
        elif tool == "Crop":
            test_crop_batch(args)
        elif tool == "DrawShape": 
            test_draw_shape_batch(args)
        elif tool == "HighlightBox": 
            test_highlight_box_batch(args)
        elif tool == "MaskBox": 
            test_mask_box_batch(args)
        elif tool == "LanguageModel": 
            test_language_model_batch(args)
        elif tool == "GetSubplotInfo": 
            test_get_subplot_info_batch(args)
        elif tool == "GetBarInfo": 
            test_get_bar_info_batch(args)
        else:
            print(f"❌ 未知或未实现的工具批量测试: {tool}")
    
    print("\n====== 所有批量测试完成 ======")

if __name__ == "__main__":
    main()

# python test_batch_tools.py --tools OCR GetBarInfo --controller_addr http://SH-IDC1-10-140-37-6:21112