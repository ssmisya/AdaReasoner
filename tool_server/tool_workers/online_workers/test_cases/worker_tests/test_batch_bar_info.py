"""批量测试GetBarInfo工具，异步发送5个请求，每个请求之间间隔1秒"""
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
from PIL import Image

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

def send_request(worker_addr, data, request_id):
    """发送单个请求并处理响应"""
    print(f"\n------ 发送请求 {request_id+1}/5 ------")
    
    start_time = time.time()
    result = {
        "id": request_id + 1,
        "start_time": start_time,
        "status": "未知",
        "duration": None,
        "bars_count": 0,
        "bars": {}
    }
    
    try:
        response = requests.post(
            worker_addr + "/worker_generate",
            headers={"User-Agent": "FastChat Client"},
            json=data,
            timeout=6000  # 设置超时时间
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
            
            if "bars" in res:
                bars = res["bars"]
                result["bars_count"] = len(bars)
                result["bars"] = bars
        else:
            result["status"] = "失败"
            result["error"] = response.text
    
    except requests.exceptions.Timeout:
        result["status"] = "超时"
    except Exception as e:
        result["status"] = "错误"
        result["error"] = str(e)
    
    return result

def test_get_bar_info_batch(args):
    """批量测试GetBarInfo工具，异步发送5个请求，每个请求之间间隔1秒"""
    print("\n====== 批量测试GetBarInfo工具 ======")
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

    # 开始批量请求的计时
    batch_start_time = time.time()
    print(f"开始发送批量请求，共5个请求，每个请求间隔1秒发送...")
    
    # 使用线程池异步处理请求
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # 提交所有请求，每个间隔1秒
        future_to_id = {}
        for i in range(5):
            if i > 0:
                time.sleep(1)  # 每个请求间隔1秒发送
            future = executor.submit(send_request, worker_addr, datas, i)
            future_to_id[future] = i
            print(f"已提交请求 {i+1}")
        
        # 收集所有结果（这里不会阻塞，请求都已经发出去了）
        for future in concurrent.futures.as_completed(future_to_id):
            request_id = future_to_id[future]
            try:
                result = future.result()
                results.append(result)
                print(f"收到请求 {request_id+1} 的响应")
            except Exception as e:
                print(f"处理请求 {request_id+1} 时出错: {str(e)}")
                results.append({
                    "id": request_id + 1,
                    "status": "错误",
                    "error": str(e)
                })
    
    # 计算总时间
    batch_end_time = time.time()
    total_time = batch_end_time - batch_start_time
    
    # 对结果进行排序，按请求ID
    results.sort(key=lambda x: x["id"])
    
    # 统计和展示结果
    print("\n====== 批量请求统计 ======")
    print(f"总耗时: {total_time:.3f}秒")
    print(f"请求次数: 5")
    
    # 统计成功的请求
    success_results = [r for r in results if r["status"] == "成功" and r["duration"] is not None]
    if success_results:
        success_times = [r["duration"] for r in success_results]
        print(f"成功请求数: {len(success_results)}")
        print(f"平均请求时间: {sum(success_times)/len(success_times):.3f}秒")
        print(f"最长请求时间: {max(success_times):.3f}秒")
        print(f"最短请求时间: {min(success_times):.3f}秒")
    else:
        print("所有请求均失败")
    
    # 打印每个请求的详细结果
    print("\n各请求详细结果:")
    for result in results:
        req_id = result["id"]
        status = result["status"]
        
        if status == "成功":
            duration = result["duration"]
            bars_count = result["bars_count"]
            print(f"  请求 {req_id}: {status}, 耗时: {duration:.3f}秒, 检测到柱子数: {bars_count}")
            
            # 显示前3个柱子的信息
            bars = result["bars"]
            for j, (label, bbox) in enumerate(list(bars.items())[:3]):
                print(f"    '{label}': {bbox}")
            
            if bars_count > 3:
                print(f"    ... 还有{bars_count-3}个柱子未显示")
        else:
            error = result.get("error", "未知错误")
            print(f"  请求 {req_id}: {status}, 错误: {error}")
    
    # 保存结果到JSON文件
    output_json_path = os.path.join(args.output_dir, "batch_bar_info_results.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {output_json_path}")

def main():
    parser = argparse.ArgumentParser(description="批量测试GetBarInfo工具")
    
    # 通用参数
    parser.add_argument(
        "--controller_addr", type=str, default="http://SH-IDC1-10-140-37-6:21112",
        help="控制器地址"
    )
    parser.add_argument(
        "--image-path", type=str, 
        default="tool_server/tool_workers/online_workers/test_cases/worker_tests/input_cases/bars3.jpg",
        help="测试图像路径"
    )
    parser.add_argument(
        "--output-dir", type=str, 
        default="tool_server/tool_workers/online_workers/test_cases/worker_tests/test_results_bar",
        help="输出目录"
    )
    
    args = parser.parse_args()
    
    # 检查并创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行批量测试
    test_get_bar_info_batch(args)
    
    print("\n====== 批量测试完成 ======")

if __name__ == "__main__":
    main() 