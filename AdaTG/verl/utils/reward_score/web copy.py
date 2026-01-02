# coding: utf-8
from openai import OpenAI
import requests
import random
import re
import os
import ast
import gymnasium as gym
import numpy as np
import contextlib
# 导入并发处理和进度条所需的库
import concurrent.futures
from tqdm import tqdm

# ==============================================================================
# 变量和客户端初始化 (您的原始代码)
# ==============================================================================

openai_api_key = "not-needed"
# 确保 openai_api_url 是一个字符串
_openai_api_url_tuple = os.environ.get("LLM_AS_A_JUDGE_BASE", "http://10.39.3.123:18901/v1"),
openai_api_url = _openai_api_url_tuple[0] if isinstance(_openai_api_url_tuple, tuple) else _openai_api_url_tuple
openai_model_name = os.environ.get("LLM_AS_A_JUDGE_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")

# 设置并发请求的最大线程数
MAX_WORKERS = 64 # 您可以根据服务器的承载能力和网络情况调整此数值

# 添加代理禁用上下文管理器
@contextlib.contextmanager
def no_proxy():
    """一个上下文管理器，可以在其作用域内临时禁用代理环境变量。"""
    proxy_keys = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']
    original_proxies = {key: os.environ.get(key) for key in proxy_keys}
    
    for key in proxy_keys:
        if key in os.environ:
            del os.environ[key]
            
    try:
        yield
    finally:
        for key, value in original_proxies.items():
            if value is not None:
                os.environ[key] = value

try:
    CLIENT = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_url
    )
    print("OpenAI client initialized successfully.")
except Exception as e:
    CLIENT = None
    print(f"Failed to initialize OpenAI client: {e}")

# ==============================================================================
# Prompt 构建函数 (您的原始代码)
# ==============================================================================

def get_chat_template():
    """返回用于 LLM 判断的基础指令模板。"""
    return """
Below are two answers to a question. [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question. Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judgement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""

def get_prompt_examples():
    """返回用于 Few-shot 学习的示例。"""
    example_1 = """[Question]: Is the countertop tan or blue?\n[Standard Answer]: The countertop is tan.\n[Model_answer] : tan\nJudgement: 1"""
    example_2 = """[Question]: Who is wearing pants?\n[Standard Answer]: The boy is wearing pants.\n[Model_answer] : The girl in the picture is wearing pants.\nJudgement: 0"""
    example_3 = """[Question]: What color is the towel in the center of the picture?\n[Standard Answer]: A. The towel in the center of the picture is blue.\n[Model_answer] : The towel in the center of the picture is pink.\nJudgement: 0"""
    return [example_1, example_2, example_3]

def get_full_prompt(predict_str, ground_truth_str, question=""):
    """构建最终发送给 LLM 的完整 Prompt。"""
    chat_template = get_chat_template()
    examples = get_prompt_examples()
    demo_prompt = chat_template + "\n\n".join(examples) + "\n\n"
    test_prompt = f"""[Question]: {question}\n[Standard Answer]: {ground_truth_str}\n[Model_answer] : {predict_str}\nJudgement:"""
    return f'{demo_prompt}{test_prompt}'

# ==============================================================================
# 核心评分函数 (您的原始代码)
# `llm_judge_web` 保持不变，它将作为每个线程的工作单元
# ==============================================================================

def llm_judge_web(solution_str, ground_truth, question=""):
    """
    使用LLM对模型回答进行评分 (此函数现在是并发执行的单元)
    
    Args:
        solution_str: 模型的回答字符串
        ground_truth: 标准答案
        question: 问题（可选）
        
    Returns:
        float: 评分结果，1.0表示正确，0.0表示错误
    """
    response_pattern = r'<response>(.*?)</response>'
    response_match = re.search(response_pattern, solution_str, re.DOTALL)
    if not response_match:
        # 如果格式不匹配，直接返回 0.0，避免不必要的API调用
        return 0.0
    
    response_content = response_match.group(1).strip()

    try:
        pred_str = str(response_content)
        gold_str = str(ground_truth)
        full_prompt = get_full_prompt(pred_str, gold_str, question)
        
        with no_proxy():
            chat_response = CLIENT.chat.completions.create(
                model=openai_model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that evaluates answers."},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.0,
                max_tokens=10
            )
        
        response = chat_response.choices[0].message.content.strip()
        
        if 'Judgement:' in response:
            response = response.split('Judgement:')[-1].strip()
        
        if '1' in response:
            return 1.0
        elif '0' in response:
            return 0.0
        else:
            print(f"警告: 无法解析的 LLM 响应 '{response}', 计为 0 分。")
            return 0.0
            
    except Exception as e:
        print(f"错误: 调用 LLM API 时发生错误: {e}")
        return 0.0

# ==============================================================================
# 辅助函数 (您的原始代码)
# ==============================================================================

def average_before_first_negative_one(numbers):
    # ... (您的原始代码，无需修改)
    try:
        if not isinstance(numbers, list):
            numbers = list(numbers)
        try:
            first_negative_one_index = numbers.index(-1)
            valid_numbers = numbers[:first_negative_one_index]
        except ValueError:
            valid_numbers = numbers
        if not valid_numbers:
            return 0.0
        end_index = len(valid_numbers)
        for i in range(len(valid_numbers) - 1, -1, -1):
            if valid_numbers[i] != 0:
                break
            end_index = i
        numbers_for_average = valid_numbers[:end_index]
        if not numbers_for_average:
            return 0.0
        return sum(numbers_for_average) / len(numbers_for_average)
    except Exception as e:
        print(f"Error in average_before_first_negative_one: {e}")
        return 0.0

def calculate_format_reward(solution_str):
    # ... (您的原始代码，无需修改)
    try:
        think_open_count = solution_str.count('<think>')
        think_close_count = solution_str.count('</think>')
        response_open_count = solution_str.count('<response>')
        response_close_count = solution_str.count('</response>')
        tool_call_open_count = solution_str.count('<tool_call>')
        tool_call_close_count = solution_str.count('</tool_call>')
        if think_open_count == 0 or response_open_count == 0:
            return 0
        if response_open_count != 1:
            return 0
        if think_open_count != think_close_count:
            return 0
        if response_open_count != response_close_count:
            return 0
        if tool_call_open_count != tool_call_close_count:
            return 0
        if think_open_count != (response_open_count + tool_call_open_count):
            return 0
        return 1
    except Exception as e:
        print(f"Error in format calculation: {e}")
        return 0

# ==============================================================================
# 【核心修改】实现并发处理的 `compute_score` 函数
# ==============================================================================

def compute_score(data_sources, solution_strs, ground_truths, extra_infos=None, tool_rewards=None, **kwargs):
    """
    使用并发方式计算所有样本的分数。
    """
    num_samples = len(data_sources)
    # 1. 初始化一个与输入大小相同的结果列表，用于存储最终的详细信息
    reward_details = [None] * num_samples
    # 2. 创建一个列表，用于存放需要LLM评判的任务
    tasks_to_judge = []

    print("开始预处理和准备评判任务...")
    # 3. 第一次遍历：预处理，分离需要LLM评判的任务
    for i in range(num_samples):
        solution_str = solution_strs[i]
        format_reward = calculate_format_reward(solution_str)
        
        if format_reward > 0:
            # 格式正确，创建一个任务并添加到待办列表
            # 任务中包含调用llm_judge_web所需参数以及原始索引，以便后续结果重组
            task = {
                'index': i,
                'solution_str': solution_str,
                'ground_truth': ground_truths[i],
                'question': extra_infos[i]['question'] if extra_infos and extra_infos[i] and 'question' in extra_infos[i] else ""
            }
            tasks_to_judge.append(task)
        else:
            # 格式错误，直接计算最终结果并填充
            reward_details[i] = {
                "score": 0.0,
                "format_reward": 0,
                "accuracy_reward": 0,
                "tool_reward": 0.0, # tool_reward在这种情况下也计为0
                "data_source": data_sources[i]
            }

    print(f"预处理完成。总共有 {len(tasks_to_judge)} / {num_samples} 个样本需要LLM评判。")

    # 4. 第二步：使用线程池并发执行所有LLM评判任务
    if tasks_to_judge:
        future_to_task = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务到线程池
            for task in tasks_to_judge:
                future = executor.submit(llm_judge_web, task['solution_str'], task['ground_truth'])
                future_to_task[future] = task

            # 使用tqdm显示进度条，并处理已完成的任务
            print(f"开始使用 {MAX_WORKERS} 个线程进行并发评判...")
            for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks_to_judge), desc="LLM Judging"):
                task = future_to_task[future]
                original_index = task['index']
                
                try:
                    # 获取LLM返回的准确度分数
                    accuracy_reward = future.result()
                    
                    # ---- 在这里完成该样本的剩余计分逻辑 ----
                    tool_reward = tool_rewards[original_index]
                    
                    if tool_reward is not None:
                        if isinstance(tool_reward, np.ndarray):
                            tool_reward_list = tool_reward.tolist()
                        else:
                            tool_reward_list = tool_reward
                        processed_tool_reward = average_before_first_negative_one(tool_reward_list)
                    else:
                        processed_tool_reward = 0.0
                    
                    if processed_tool_reward is None:
                        processed_tool_reward = 0.0
                    
                    total_score = accuracy_reward + processed_tool_reward
                    
                    # 将完整的奖励信息存入结果列表的正确位置
                    reward_details[original_index] = {
                        "score": total_score,
                        "format_reward": 1, # 格式在此处必为1
                        "accuracy_reward": accuracy_reward,
                        "tool_reward": processed_tool_reward,
                        "data_source": data_sources[original_index]
                    }
                except Exception as exc:
                    print(f"任务 {original_index} 执行时产生错误: {exc}")
                    # 即使任务出错，也要填充一个错误结果，避免列表出现None
                    reward_details[original_index] = {
                        "score": 0.0,
                        "format_reward": 1,
                        "accuracy_reward": 0.0,
                        "tool_reward": 0.0,
                        "data_source": data_sources[original_index],
                        "error": str(exc)
                    }

    print("所有评判任务完成。")
    return reward_details

# ==============================================================================
# 示例用法
# ==============================================================================
if __name__ == '__main__':
    # 构造一些模拟数据来测试
    # 在实际使用中，您会从您的数据源加载这些列表
    mock_data_sources = [
        {"id": 1, "question": "What color is the sky?"},
        {"id": 2, "question": "Is the cat on the mat?"},
        {"id": 3, "question": "What is 2+2?"},
        {"id": 4, "question": "Invalid format test"}
    ]
    
    mock_solution_strs = [
        "<think>The sky is blue.</think><response>blue</response>", # 格式正确，答案正确
        "<think>The cat is not on the mat, it is under the table.</think><response>The cat is under the table.</response>", # 格式正确，答案错误
        "<think>2+2 is 4</think><response>4</response>", # 格式正确，答案正确
        "<response>missing think tag</response>" # 格式错误
    ]
    
    mock_ground_truths = [
        "blue",
        "The cat is on the mat.",
        "4",
        "N/A"
    ]
    
    mock_tool_rewards = [
        [1.0, 1.0, -1], # 平均值为 1.0
        [0.5, 0.5, 0.0, -1], # 平均值为 0.5
        [], # 平均值为 0.0
        [1.0, 1.0, 1.0] # 平均值为 1.0 (但因为格式错误，不会被计算)
    ]
    
    # 运行并发评分函数
    if CLIENT:
        final_rewards = compute_score(mock_data_sources, mock_solution_strs, mock_ground_truths, mock_tool_rewards)
        
        # 打印结果
        import json
        print("\n--- 最终评分结果 ---")
        print(json.dumps(final_rewards, indent=2))
    else:
        print("无法运行示例，因为OpenAI客户端初始化失败。")