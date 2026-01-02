# jigsaw.py
from openai import OpenAI
import requests
import random
import re
import os
import ast
import numpy as np


def compute_score(data_sources, solution_strs, ground_truths, tool_rewards,extra_infos=None):
    # 每个的形状都是batch_size*n，包括模型的回答solution_strs
    scores = [0] * len(data_sources)
    reward_details = []  # 存储详细的奖励信息
    
    for i in range(len(data_sources)):
        solution_str = solution_strs[i]
        ground_truth = ground_truths[i]
        data_source = data_sources[i]
        tool_reward = tool_rewards[i]
        
        # 计算format奖励
        format_reward = calculate_format_reward(solution_str)
        accuracy_reward = 0
        
        if format_reward:
            accuracy_reward = calculate_jigsaw_accuracy(solution_str, ground_truth)
            
            if accuracy_reward:
                accuracy_reward = 4  # 如同FrozenLake任务，答对给4分
            
            # 处理tool_reward - 确保转换为列表
            if tool_reward is not None:
                # 如果是numpy数组，转换为列表
                if isinstance(tool_reward, np.ndarray):
                    tool_reward_list = tool_reward.tolist()
                else:
                    tool_reward_list = tool_reward
                processed_tool_reward = average_before_first_negative_one(tool_reward_list)
            else:
                processed_tool_reward = 0.0
                
            if processed_tool_reward is None:
                processed_tool_reward = 0.0
            
            # 总分计算
            if accuracy_reward:
                # 只要答对了，就给8分
                total_score = 8
            else:
                # 答错了，按照工具使用情况给分
                total_score = accuracy_reward + processed_tool_reward
            scores[i] = total_score
        
        else:
            total_score = 0.0
            scores[i] = 0.0
            processed_tool_reward = 0.0
        
        # 存储详细的奖励信息
        reward_details.append({
            "score": total_score,
            "format_reward": format_reward,
            "accuracy_reward": accuracy_reward,
            "tool_reward": processed_tool_reward,
            "data_source": data_source
        })
    
    return reward_details


def average_before_first_negative_one(numbers):
    """
    计算列表中第一个-1前所有数字的平均值，但排除第一个-1前的连续末尾0
    
    Args:
        numbers: 数字列表
        
    Returns:
        float: 平均值，如果没有有效数字则返回0.0
    """
    try:
        # 确保numbers是列表类型
        if not isinstance(numbers, list):
            numbers = list(numbers)
            
        # 找到第一个-1的索引，如果不存在则使用整个列表
        try:
            first_negative_one_index = numbers.index(-1)
            valid_numbers = numbers[:first_negative_one_index]
        except ValueError:
            # 如果列表中没有-1，使用整个列表
            valid_numbers = numbers
        
        # 如果没有有效数字，返回0.0
        if not valid_numbers:
            return 0.0
        
        # 从末尾开始，找到最后一个非0数字的位置
        # 排除第一个-1前的连续末尾0
        end_index = len(valid_numbers)
        for i in range(len(valid_numbers) - 1, -1, -1):
            if valid_numbers[i] != 0:
                break
            end_index = i
        
        # 获取用于计算平均值的数字
        numbers_for_average = valid_numbers[:end_index]
        
        # 如果没有有效数字，返回0.0
        if not numbers_for_average:
            return 0.0
        
        # 计算平均值
        return sum(numbers_for_average) / len(numbers_for_average)
        
    except Exception as e:
        print(f"Error in average_before_first_negative_one: {e}")
        return 0.0


def calculate_format_reward(solution_str):
    """
    计算格式奖励
    
    条件：
    1. 必须有<think>content</think>和<response>content</response>这两种标签
    2. <think>和</think>的数量要相同，<response>和</response>的数量要相同，<tool_call>和</tool_call>数量要相同
    3. <think>的数量要 = <response> + <tool_call>
    
    参数:
        solution_str (str): 模型的回复字符串
        
    返回:
        int: 1表示格式正确，0表示格式错误
    """
    try:
        # 统计各种标签的数量
        think_open_count = solution_str.count('<think>')
        think_close_count = solution_str.count('</think>')
        
        response_open_count = solution_str.count('<response>')
        response_close_count = solution_str.count('</response>')
        
        tool_call_open_count = solution_str.count('<tool_call>')
        tool_call_close_count = solution_str.count('</tool_call>')
        
        # 条件1：必须有<think>和<response>标签
        if think_open_count == 0 or response_open_count == 0:
            return 0
        
        # 条件2：<response>标签数量只能为1
        if response_open_count != 1:
            return 0
        
        # 条件3：各标签的开闭数量要相同
        if think_open_count != think_close_count:
            return 0
        if response_open_count != response_close_count:
            return 0
        if tool_call_open_count != tool_call_close_count:
            return 0
        
        # 条件4：<think>的数量 = <response> + <tool_call>
        if think_open_count != (response_open_count + tool_call_open_count):
            return 0
        
        # 所有条件都满足，返回1
        return 1
        
    except Exception as e:
        # 如果出现任何异常，返回0
        print(f"Error in format calculation: {e}")
        return 0


def calculate_jigsaw_accuracy(solution_str, ground_truth):
    """
    计算jigsaw拼图任务的准确性奖励
    
    参数:
        solution_str (str): 模型的回复字符串
        ground_truth (str): 真实答案，形式为"a", "b", "c"等
        
    返回:
        int: 1表示回答正确，0表示回答错误
    """
    try:
        # 先提取<response>标签中的内容
        response_pattern = r'<response>(.*?)</response>'
        response_match = re.search(response_pattern, solution_str, re.DOTALL)
        
        if response_match:
            response_content = response_match.group(1).strip()
        else:
            # 如果没有找到response标签，返回0
            return 0
        
        # 方法1：尝试从response内容中提取\\boxed{}
        model_answer = extract_boxed_answer(response_content)
        
        # 方法2：如果没有找到boxed，则直接查找选项字母
        if model_answer is None:
            model_answer = extract_letter_answer(response_content)
        
        # 如果仍然没有找到有效答案，返回0
        if model_answer is None:
            return 0
        
        # 比较模型答案和真实答案（不区分大小写）
        if model_answer.lower() == ground_truth.lower():
            return 1
        else:
            return 0
            
    except Exception as e:
        print(f"Error in jigsaw accuracy calculation: {e}")
        return 0


def extract_boxed_answer(text):
    """
    从文本中提取\\boxed{}中的答案
    
    参数:
        text (str): 要提取的文本
        
    返回:
        str or None: 提取的字母选项或None
    """
    # 匹配boxed{...}模式
    boxed_pattern = r'boxed\{([^}]+)\}'
    boxed_match = re.search(boxed_pattern, text, re.IGNORECASE)
    
    if boxed_match:
        boxed_content = boxed_match.group(1).strip()
        # 提取单个字母答案
        letter_pattern = r'([A-Za-z])'
        letter_match = re.search(letter_pattern, boxed_content)
        
        if letter_match:
            return letter_match.group(1)
    
    return None


def extract_letter_answer(text):
    """
    从文本中直接查找选项字母
    
    参数:
        text (str): 要查找的文本
        
    返回:
        str or None: 找到的字母选项或None
    """
    # 常见的选项标记模式
    patterns = [
        r'(?:my answer is|I choose|the answer is|选择|答案是)[^A-Za-z]*\(?([A-Za-z])\)?',
        r'option\s*\(?([A-Za-z])\)?',
        r'选项\s*\(?([A-Za-z])\)?',
        r'答案\s*\(?([A-Za-z])\)?',
        r'选择\s*\(?([A-Za-z])\)?',
        r'^\s*\(?([A-Za-z])\)?\s*$',  # 单独的字母
        r'(?:^|\s)([A-Za-z])(?:\s|$)'  # 被空格包围的单个字母
    ]
    
    # 尝试各种模式
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        potential_answers = [match.group(1) for match in matches]
        
        if potential_answers:
            # 统计各个字母出现的频率
            counts = {}
            for ans in potential_answers:
                counts[ans.upper()] = counts.get(ans.upper(), 0) + 1
            
            # 找出出现最多的字母
            most_common = max(counts.items(), key=lambda x: x[1])
            return most_common[0]
    
    # 如果上面都没找到，尝试最简单的方法：找出所有单字母
    all_letters = re.findall(r'(?:^|\s)([A-Za-z])(?:\s|$)', text)
    
    if all_letters:
        # 只考虑出现在前几个选项的字母 (A-E)
        valid_options = [l for l in all_letters if l.upper() in 'ABCDE']
        if valid_options:
            # 按出现顺序取第一个有效选项
            return valid_options[0]
    
    return None


def test_calculate_jigsaw_accuracy():
    """
    测试jigsaw准确度计算函数
    """
    # 测试用例集
    test_cases = [
        {
            "solution": "<think>分析图片中缺失的部分，对比选项...(略)</think><response>根据图片分析，答案是 \\boxed{A}</response>",
            "ground_truth": "a",
            "expected": 1
        },
        {
            "solution": "<think>分析图片中缺失的部分，对比选项...(略)</think><response>缺失部分应该是选项B，\\boxed{B}</response>",
            "ground_truth": "b",
            "expected": 1
        },
        {
            "solution": "<think>分析图片中缺失的部分，对比选项...(略)</think><response>缺失部分应该是选项B，\\boxed{B}</response>",
            "ground_truth": "a",
            "expected": 0
        },
        {
            "solution": "<think>分析图片中缺失的部分，对比选项...(略)</think><response>我认为答案是选项A</response>",
            "ground_truth": "a",
            "expected": 1
        },
        {
            "solution": "<think>分析图片中缺失的部分，对比选项...(略)</think><response>答案: B</response>",
            "ground_truth": "b",
            "expected": 1
        },
        {
            "solution": "<think>分析图片中缺失的部分，对比选项...(略)</think><response>选项C最符合缺失部分</response>",
            "ground_truth": "c",
            "expected": 1
        },
        {
            "solution": "<think>分析图片中缺失的部分，对比选项...(略)</think><response>我选择 (A)</response>",
            "ground_truth": "a",
            "expected": 1
        }
    ]
    
    # 运行测试用例
    for i, case in enumerate(test_cases):
        result = calculate_jigsaw_accuracy(case["solution"], case["ground_truth"])
        success = result == case["expected"]
        print(f"测试用例 {i+1}: {'通过' if success else '失败'}")
        if not success:
            print(f"  输入: {case['solution']}")
            print(f"  期望: {case['expected']}")
            print(f"  实际: {result}")


if __name__ == "__main__":
    # 运行测试函数
    test_calculate_jigsaw_accuracy()