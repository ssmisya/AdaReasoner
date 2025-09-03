# TEST_POINT_PARALLEL.PY
import argparse
import asyncio
import base64
import json
import os
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import aiohttp
import numpy as np
import requests
from PIL import Image, ImageDraw
from tqdm import tqdm

# 设置日志格式
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("point_parallel_test")

# 测试场景描述列表 - 提供多种对象和位置描述
DESCRIPTIONS = [
    "the person's face", "the center of the image", "the dog's nose",
    "the leftmost object", "the car's headlight", "the traffic light",
    "the rightmost person", "the tallest building", "the center of the table",
    "the clock's hour hand", "the cat's eyes", "the tree on the right",
    "the laptop screen", "the chair in the foreground", "the book on the shelf",
    "the bird's beak", "the top-left corner of the sign", "the doorknob",
    "the center of the flower", "the steering wheel"
]

def load_image(image_path):
    """加载图像并调整大小"""
    try:
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        
        # 如果图像太大，调整到合理大小
        if max(h, w) > 1200:
            scale = 1200 / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            img = img.resize((new_w, new_h))
            
        return img
    except Exception as e:
        logger.error(f"无法加载图像 {image_path}: {e}")
        raise

def encode_image(image):
    """将PIL图像转换为base64编码"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode()

def get_worker_address(controller_addr, model_name):
    """获取指定模型的工作器地址"""
    logger.info(f"正在获取'{model_name}'工作器地址...")
    
    try:
        # 获取可用模型列表
        response = requests.post(controller_addr + "/list_models")
        if response.status_code != 200:
            logger.error(f"获取模型列表失败: {response.text}")
            return None
            
        models = response.json()["models"]
        logger.info(f"可用模型: {models}")
        
        if model_name not in models:
            logger.error(f"模型'{model_name}'不在可用模型列表中")
            return None
        
        # 获取工作器地址
        response = requests.post(
            controller_addr + "/get_worker_address", 
            json={"model": model_name}
        )
        
        if response.status_code != 200:
            logger.error(f"获取工作器地址失败: {response.text}")
            return None
            
        worker_addr = response.json()["address"]
        
        if not worker_addr:
            logger.error(f"未找到'{model_name}'的工作器地址")
            return None
            
        logger.info(f"工作器地址: {worker_addr}")
        return worker_addr
        
    except Exception as e:
        logger.error(f"获取工作器地址时出错: {str(e)}")
        traceback.print_exc()
        return None

class PointTester:
    """点检测测试器"""
    
    def __init__(self, controller_addr, worker_addr=None, output_dir="./test_results"):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化结果统计
        self.results = {
            "single_call": {},
            "batch_call": {},
            "concurrent_call": {}
        }
    
    async def single_request(self, image, description, session=None):
        """发送单个点检测请求"""
        if not self.worker_addr:
            return None
            
        data = {
            "image": encode_image(image),
            "description": description
        }
        
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True
            
        try:
            start_time = time.time()
            async with session.post(
                f"{self.worker_addr}/worker_generate",
                json=data,
                timeout=600
            ) as response:
                duration = time.time() - start_time
                if response.status == 200:
                    result = await response.json()
                    result["request_time"] = duration
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"请求失败 ({response.status}): {error_text}")
                    return {
                        "status": "failed", 
                        "error": f"HTTP {response.status}", 
                        "request_time": duration
                    }
        except asyncio.TimeoutError:
            logger.error("请求超时")
            return {"status": "failed", "error": "Timeout", "request_time": time.time() - start_time}
        except Exception as e:
            logger.error(f"请求异常: {str(e)}")
            return {"status": "failed", "error": str(e), "request_time": time.time() - start_time}
        finally:
            if close_session:
                await session.close()
    
    async def batch_request(self, image, descriptions, batch_size=None):
        """发送批量点检测请求"""
        if not self.worker_addr:
            return None
            
        # 为每个描述准备参数
        batch_params = []
        for desc in descriptions:
            batch_params.append({
                "image": encode_image(image),
                "description": desc
            })
            
        # 如果指定了batch_size，分批处理
        if batch_size and batch_size < len(batch_params):
            logger.info(f"将{len(batch_params)}个请求分成{len(batch_params) // batch_size + 1}批处理")
            all_results = []
            
            for i in range(0, len(batch_params), batch_size):
                batch_chunk = batch_params[i:i+batch_size]
                chunk_results = await self._execute_batch(batch_chunk)
                all_results.extend(chunk_results if isinstance(chunk_results, list) else [chunk_results])
                
            return all_results
        else:
            return await self._execute_batch(batch_params)
    
    async def _execute_batch(self, batch_params):
        """执行批量请求"""
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            try:
                async with session.post(
                    f"{self.worker_addr}/worker_generate_batch",
                    json=batch_params,
                    timeout=1200
                ) as response:
                    duration = time.time() - start_time
                    
                    if response.status == 200:
                        results = await response.json()
                        # 添加请求时间信息
                        if isinstance(results, list):
                            for result in results:
                                result["batch_request_time"] = duration
                        else:
                            results["batch_request_time"] = duration
                        return results
                    else:
                        error_text = await response.text()
                        logger.error(f"批量请求失败 ({response.status}): {error_text}")
                        return {
                            "status": "failed", 
                            "error": f"HTTP {response.status}", 
                            "batch_request_time": duration
                        }
            except asyncio.TimeoutError:
                logger.error("批量请求超时")
                return {"status": "failed", "error": "Timeout", "batch_request_time": time.time() - start_time}
            except Exception as e:
                logger.error(f"批量请求异常: {str(e)}")
                traceback.print_exc()
                return {"status": "failed", "error": str(e), "batch_request_time": time.time() - start_time}
    
    async def concurrent_requests(self, image, descriptions, concurrency):
        """并发发送多个单点检测请求"""
        if not self.worker_addr:
            return None
            
        async with aiohttp.ClientSession() as session:
            tasks = []
            for desc in descriptions:
                tasks.append(self.single_request(image, desc, session))
                
            # 使用信号量控制并发
            semaphore = asyncio.Semaphore(concurrency)
            
            async def controlled_task(task):
                async with semaphore:
                    return await task
                    
            controlled_tasks = [controlled_task(task) for task in tasks]
            
            start_time = time.time()
            results = await asyncio.gather(*controlled_tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # 处理可能的异常
            processed_results = []
            for res in results:
                if isinstance(res, Exception):
                    processed_results.append({"status": "failed", "error": str(res)})
                else:
                    processed_results.append(res)
            
            return {
                "results": processed_results,
                "total_time": total_time,
                "average_time": total_time / len(descriptions) if descriptions else 0
            }
    
    async def test_single_call(self, image_path, num_calls=5):
        """测试单个调用性能"""
        logger.info(f"测试单个调用 ({num_calls}次)...")
        
        image = load_image(image_path)
        
        times = []
        success_count = 0
        
        async with aiohttp.ClientSession() as session:
            for i in range(num_calls):
                description = random.choice(DESCRIPTIONS)
                logger.info(f"请求 {i+1}/{num_calls}: '{description}'")
                
                result = await self.single_request(image, description, session)
                
                if result and result.get("status") == "success":
                    success_count += 1
                    times.append(result["request_time"])
                    
                    # 保存第一个成功结果的图片示例
                    if success_count == 1 and "edited_image" in result:
                        try:
                            img_data = base64.b64decode(result["edited_image"])
                            with open(os.path.join(self.output_dir, "single_call_example.jpg"), "wb") as f:
                                f.write(img_data)
                                logger.info("已保存单次调用示例图像")
                        except Exception as e:
                            logger.error(f"保存示例图像失败: {e}")
                
                # 短暂暂停避免请求过快
                await asyncio.sleep(0.5)
        
        # 计算统计信息
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            p95_time = sorted(times)[int(len(times) * 0.95)]
            
            self.results["single_call"] = {
                "success_rate": success_count / num_calls,
                "average_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "p95_time": p95_time
            }
            
            logger.info(f"单次调用结果:")
            logger.info(f"成功率: {success_count}/{num_calls} ({success_count/num_calls*100:.1f}%)")
            logger.info(f"平均时间: {avg_time:.3f}秒")
            logger.info(f"最小时间: {min_time:.3f}秒")
            logger.info(f"最大时间: {max_time:.3f}秒")
            logger.info(f"P95时间: {p95_time:.3f}秒")
        else:
            logger.error("所有单次调用均失败")
            self.results["single_call"] = {"success_rate": 0}
    
    async def test_batch_call(self, image_path, batch_sizes=[10, 50, 100]):
        """测试批量调用性能"""
        logger.info(f"测试批量调用...")
        
        image = load_image(image_path)
        self.results["batch_call"] = {}
        
        for batch_size in batch_sizes:
            logger.info(f"测试批量大小: {batch_size}")
            
            # 随机选择不同描述
            descriptions = random.choices(DESCRIPTIONS, k=batch_size)
            
            # 发送批量请求
            start_time = time.time()
            batch_results = await self.batch_request(image, descriptions)
            total_time = time.time() - start_time
            
            if isinstance(batch_results, list):
                success_count = sum(1 for r in batch_results if r.get("status") == "success")
                
                self.results["batch_call"][batch_size] = {
                    "total_time": total_time,
                    "average_time": total_time / batch_size,
                    "success_rate": success_count / batch_size,
                    "throughput": batch_size / total_time
                }
                
                logger.info(f"批量大小 {batch_size} 结果:")
                logger.info(f"总时间: {total_time:.3f}秒")
                logger.info(f"平均时间: {total_time/batch_size:.3f}秒")
                logger.info(f"成功率: {success_count}/{batch_size} ({success_count/batch_size*100:.1f}%)")
                logger.info(f"吞吐量: {batch_size/total_time:.1f}请求/秒")
                
                # 保存第一个批量调用的示例图像
                if success_count > 0 and "edited_image" in batch_results[0]:
                    try:
                        img_data = base64.b64decode(batch_results[0]["edited_image"])
                        with open(os.path.join(self.output_dir, f"batch_{batch_size}_example.jpg"), "wb") as f:
                            f.write(img_data)
                            logger.info(f"已保存批量大小{batch_size}的示例图像")
                    except Exception as e:
                        logger.error(f"保存示例图像失败: {e}")
            else:
                logger.error(f"批量调用失败: {batch_results}")
                self.results["batch_call"][batch_size] = {
                    "error": batch_results.get("error", "Unknown error"),
                    "success_rate": 0
                }
            
            # 在大型批次之间暂停
            await asyncio.sleep(2)
    
    async def test_concurrent_call(self, image_path, concurrency_levels=[10, 20, 50]):
        """测试并发调用性能"""
        logger.info(f"测试并发调用...")
        
        image = load_image(image_path)
        self.results["concurrent_call"] = {}
        
        for concurrency in concurrency_levels:
            logger.info(f"测试并发级别: {concurrency}")
            
            # 为每个并发请求随机选择描述
            descriptions = random.choices(DESCRIPTIONS, k=concurrency)
            
            # 发送并发请求
            start_time = time.time()
            concurrent_results = await self.concurrent_requests(image, descriptions, concurrency)
            total_time = time.time() - start_time
            
            if concurrent_results and "results" in concurrent_results:
                success_count = sum(1 for r in concurrent_results["results"] if r.get("status") == "success")
                
                self.results["concurrent_call"][concurrency] = {
                    "total_time": total_time,
                    "average_time": total_time / concurrency,
                    "success_rate": success_count / concurrency,
                    "throughput": concurrency / total_time
                }
                
                logger.info(f"并发级别 {concurrency} 结果:")
                logger.info(f"总时间: {total_time:.3f}秒")
                logger.info(f"平均时间: {total_time/concurrency:.3f}秒")
                logger.info(f"成功率: {success_count}/{concurrency} ({success_count/concurrency*100:.1f}%)")
                logger.info(f"吞吐量: {concurrency/total_time:.1f}请求/秒")
                
                # 保存示例图像
                for result in concurrent_results["results"]:
                    if result.get("status") == "success" and "edited_image" in result:
                        try:
                            img_data = base64.b64decode(result["edited_image"])
                            with open(os.path.join(self.output_dir, f"concurrent_{concurrency}_example.jpg"), "wb") as f:
                                f.write(img_data)
                                logger.info(f"已保存并发级别{concurrency}的示例图像")
                            break  # 只保存第一个成功的示例
                        except Exception as e:
                            logger.error(f"保存示例图像失败: {e}")
            else:
                logger.error(f"并发调用失败: {concurrent_results}")
                self.results["concurrent_call"][concurrency] = {
                    "error": "Failed to get valid results",
                    "success_rate": 0
                }
            
            # 在高并发测试之间暂停
            await asyncio.sleep(3)
    
    async def test_scalability(self, image_path, request_counts=[100, 500]):
        """测试可扩展性 - 处理大量请求的能力"""
        logger.info(f"测试系统可扩展性...")

        image = load_image(image_path)
        self.results["scalability"] = {}

        for req_count in request_counts:
            logger.info(f"测试 {req_count} 个请求...")

            # 随机选择描述
            descriptions = random.choices(DESCRIPTIONS, k=req_count)

            # 单个请求逐一发送
            start_time = time.time()
            all_results = []
            success_count = 0

            async with aiohttp.ClientSession() as session:
                for i, description in enumerate(tqdm(descriptions, desc=f"处理{req_count}个请求")):
                    result = await self.single_request(image, description, session)
                    all_results.append(result)

                    if result and result.get("status") == "success":
                        success_count += 1

            total_time = time.time() - start_time

            self.results["scalability"][req_count] = {
                "total_time": total_time,
                "average_time": total_time / req_count,
                "success_rate": success_count / req_count,
                "throughput": req_count / total_time
            }

            logger.info(f"请求数 {req_count} 结果:")
            logger.info(f"总时间: {total_time:.3f}秒")
            logger.info(f"平均时间: {total_time/req_count:.3f}秒/请求")
            logger.info(f"成功率: {success_count}/{req_count} ({success_count/req_count*100:.1f}%)")
            logger.info(f"吞吐量: {req_count/total_time:.1f}请求/秒")

            # 大量请求之间暂停
            await asyncio.sleep(5)
    
    async def run_all_tests(self, image_path, 
                           single_calls=5, 
                           batch_sizes=[8, 32, 64], 
                           concurrency_levels=[8, 16, 32, 64, 128, 256], 
                           request_counts=[100, 500]):
        """运行所有测试"""
        # 如果没有提供工作器地址，尝试从控制器获取
        if not self.worker_addr:
            self.worker_addr = get_worker_address(self.controller_addr, "Point")
            if not self.worker_addr:
                logger.error("无法获取Point工作器地址，测试将失败")
                return False
        
        logger.info("====== 开始点检测工作器并行性能测试 ======")
        
        # 运行所有测试
        await self.test_single_call(image_path, num_calls=single_calls)
        # await self.test_batch_call(image_path, batch_sizes=batch_sizes)
        await self.test_concurrent_call(image_path, concurrency_levels=concurrency_levels)
        # await self.test_scalability(image_path, request_counts=request_counts)
        
        # 保存测试结果
        self.save_results()
        
        logger.info("====== 测试完成 ======")
        return True
    
    def save_results(self):
        """保存测试结果到JSON文件"""
        result_file = os.path.join(self.output_dir, "point_parallel_test_results.json")
        with open(result_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"结果已保存至: {result_file}")
        
        # 生成简单的结果摘要
        summary = self.generate_summary()
        summary_file = os.path.join(self.output_dir, "point_parallel_test_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(summary)
        logger.info(f"结果摘要已保存至: {summary_file}")
    
    def generate_summary(self):
        """生成测试结果摘要"""
        summary = []
        summary.append("=== 点检测工作器并行性能测试摘要 ===\n")
        
        # 单次调用摘要
        summary.append("--- 单次调用性能 ---")
        single_call = self.results.get("single_call", {})
        if single_call:
            summary.append(f"成功率: {single_call.get('success_rate', 0)*100:.1f}%")
            if "average_time" in single_call:
                summary.append(f"平均响应时间: {single_call['average_time']:.3f}秒")
                summary.append(f"最小/最大响应时间: {single_call['min_time']:.3f}/{single_call['max_time']:.3f}秒")
                summary.append(f"P95响应时间: {single_call['p95_time']:.3f}秒")
        
        # 批量调用摘要
        summary.append("\n--- 批量调用性能 ---")
        batch_call = self.results.get("batch_call", {})
        if batch_call:
            for batch_size, metrics in sorted(batch_call.items()):
                summary.append(f"批量大小 {batch_size}:")
                if "error" in metrics:
                    summary.append(f"  错误: {metrics['error']}")
                else:
                    summary.append(f"  成功率: {metrics.get('success_rate', 0)*100:.1f}%")
                    summary.append(f"  总时间: {metrics.get('total_time', 0):.3f}秒")
                    summary.append(f"  吞吐量: {metrics.get('throughput', 0):.1f}请求/秒")
        
        # 并发调用摘要
        summary.append("\n--- 并发调用性能 ---")
        concurrent_call = self.results.get("concurrent_call", {})
        if concurrent_call:
            for concurrency, metrics in sorted(concurrent_call.items()):
                summary.append(f"并发级别 {concurrency}:")
                if "error" in metrics:
                    summary.append(f"  错误: {metrics['error']}")
                else:
                    summary.append(f"  成功率: {metrics.get('success_rate', 0)*100:.1f}%")
                    summary.append(f"  总时间: {metrics.get('total_time', 0):.3f}秒")
                    summary.append(f"  吞吐量: {metrics.get('throughput', 0):.1f}请求/秒")
        
        # 可扩展性测试摘要
        summary.append("\n--- 可扩展性测试 ---")
        scalability = self.results.get("scalability", {})
        if scalability:
            for req_count, metrics in sorted(scalability.items()):
                summary.append(f"请求数 {req_count}:")
                if "error" in metrics:
                    summary.append(f"  错误: {metrics['error']}")
                else:
                    summary.append(f"  成功率: {metrics.get('success_rate', 0)*100:.1f}%")
                    summary.append(f"  总时间: {metrics.get('total_time', 0):.3f}秒")
                    summary.append(f"  吞吐量: {metrics.get('throughput', 0):.1f}请求/秒")
        
        return "\n".join(summary)

async def main():
    parser = argparse.ArgumentParser(description="点检测工作器并行性能测试")
    
    parser.add_argument(
        "--controller_addr", 
        type=str, 
        default="http://SH-IDC1-10-140-37-82:50001",
        help="控制器地址"
    )
    parser.add_argument(
        "--worker_addr", 
        type=str, 
        default=None,
        help="点检测工作器地址，若不提供则从控制器获取"
    )
    parser.add_argument(
        "--image-path", 
        type=str, 
        default="./input_cases/zebra.jpg",
        help="测试图像路径"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./point_parallel_results",
        help="输出目录"
    )
    parser.add_argument(
        "--single-calls", 
        type=int, 
        default=5,
        help="单次调用测试次数"
    )
    parser.add_argument(
        "--batch-sizes", 
        type=int, 
        nargs="+", 
        default=[8, 32, 64],
        help="批量调用测试的批量大小"
    )
    parser.add_argument(
        "--concurrency-levels", 
        type=int, 
        nargs="+", 
        default=[64, 256],
        help="并发测试的并发级别"
    )
    parser.add_argument(
        "--request-counts", 
        type=int, 
        nargs="+", 
        default=[],
        help="可扩展性测试的请求数"
    )
    parser.add_argument(
        "--test-type",
        type=str,
        choices=["all", "single", "batch", "concurrent", "scalability"],
        default="all",
        help="要运行的测试类型"
    )
    
    args = parser.parse_args()
    
    # 初始化测试器
    tester = PointTester(
        controller_addr=args.controller_addr,
        worker_addr=args.worker_addr,
        output_dir=args.output_dir
    )
    
    # 根据测试类型运行对应测试
    if args.test_type == "all":
        await tester.run_all_tests(
            args.image_path,
            single_calls=args.single_calls,
            batch_sizes=args.batch_sizes,
            concurrency_levels=args.concurrency_levels,
            request_counts=args.request_counts
        )
    elif args.test_type == "single":
        await tester.test_single_call(args.image_path, num_calls=args.single_calls)
        tester.save_results()
    elif args.test_type == "batch":
        await tester.test_batch_call(args.image_path, batch_sizes=args.batch_sizes)
        tester.save_results()
    elif args.test_type == "concurrent":
        await tester.test_concurrent_call(args.image_path, concurrency_levels=args.concurrency_levels)
        tester.save_results()
    elif args.test_type == "scalability":
        await tester.test_scalability(args.image_path, request_counts=args.request_counts)
        tester.save_results()

if __name__ == "__main__":
    # 设置更高的uvloop限制（如果可用）
    try:
        import uvloop
        uvloop.install()
        logger.info("已启用uvloop加速")
    except ImportError:
        logger.info("未安装uvloop，使用标准事件循环")
    
    # 调整默认连接池限制
    asyncio.get_event_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=32)
    )
    
    # 运行主函数
    asyncio.run(main())