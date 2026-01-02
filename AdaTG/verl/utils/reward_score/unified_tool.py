from openai import OpenAI
import requests
import random
import re
import os
import ast
import gymnasium as gym
import numpy as np
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# OpenAI 客户端设置

openai_api_key = os.environ.get("JUDGE_API_KEY", "not-needed")
openai_api_url = os.environ.get("JUDGE_BASE_URL", "http://SH-IDC1-10-140-37-111:16113/v1"),
openai_model_name = os.environ.get("JUDGE_MODEL_NAME", "/mnt/petrelfs/share_data/ai4good_shared/models/Qwen/Qwen2.5-72B-Instruct")
assert openai_api_key, "JUDGE_API_KEY is not set"
assert openai_api_url, "JUDGE_BASE_URL is not set"
assert openai_model_name, "JUDGE_MODEL_NAME is not set"

if isinstance(openai_api_url, tuple):
    openai_api_url = openai_api_url[0]
assert isinstance(openai_api_url, str)

# 设置并发请求的最大线程数
MAX_WORKERS = 32

# 用于禁用代理的上下文管理器
@contextlib.contextmanager
def no_proxy():
    """临时禁用代理环境变量的上下文管理器"""
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

# 初始化 OpenAI 客户端
CLIENT = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_url,
    timeout=300
)


def compute_score(data_sources, solution_strs, ground_truths, extra_infos=None, tool_rewards=None, **kwargs):
    """
    计算各种任务的综合奖励
    
    Args:
        data_sources: 任务类型列表
        solution_strs: 模型回答列表
        ground_truths: 标准答案列表
        extra_infos: 额外信息列表
        tool_rewards: 工具使用奖励列表
        
    Returns:
        list: 详细奖励信息列表
    """
    reward_details = []
    
    # 收集需要 Web 评判的任务
    web_tasks = []
    vs_tasks = []
    
    for i in range(len(data_sources)):
        solution_str = solution_strs[i]
        ground_truth = ground_truths[i]
        data_source = data_sources[i]
        tool_reward = tool_rewards[i] if tool_rewards is not None else None
        
        # 计算格式奖励
        format_reward = calculate_format_reward(solution_str)
        accuracy_reward = 0
        processed_tool_reward = 0.0
        
        if format_reward:
            # 根据任务类型选择不同的评分方法
            if data_source == "path_nav":
                accuracy_reward = calculate_path_nav_accuracy(solution_str, ground_truth)
                if accuracy_reward:
                    accuracy_reward = 4  # 正确路径给4分
            
            elif data_source == "path_ver":
                accuracy_reward = calculate_path_ver_accuracy(solution_str, ground_truth)
                if accuracy_reward:
                    accuracy_reward = 4  # 正确验证给4分
            
            elif data_source == "jigsaw_coco":
                accuracy_reward = calculate_jigsaw_accuracy(solution_str, ground_truth)
                if accuracy_reward:
                    accuracy_reward = 4  # 拼图正确给4分
            
            elif data_source == "visual_cot":
                # accuracy_reward = calculate_jigsaw_accuracy(solution_str, ground_truth)
                
                # if accuracy_reward:
                #     accuracy_reward = 4
                question = extra_infos[i]['question'] if extra_infos and extra_infos[i] and 'question' in extra_infos[i] else ""
                vs_tasks.append({
                    'index': i,
                    'solution_str': solution_str, 
                    'ground_truth': ground_truth,
                    'question': question
                })
                
                accuracy_reward = 0  # 先设置为0，稍后更新
                
            elif data_source == "web_guichat":
                # 收集 web 任务，稍后并行评估
                # breakpoint()
                question = extra_infos[i]['question'] if extra_infos and extra_infos[i] and 'question' in extra_infos[i] else ""
                web_tasks.append({
                    'index': i,
                    'solution_str': solution_str, 
                    'ground_truth': ground_truth,
                    'question': question
                })
                # 先设置为0，稍后更新
                accuracy_reward = 0
            
            else:
                print(f"警告: 未知数据源类型 {data_source}")
                accuracy_reward = 0
            
            # 处理工具奖励 - 对所有任务通用
            if tool_reward is not None:
                if isinstance(tool_reward, np.ndarray):
                    tool_reward_list = tool_reward.tolist()
                else:
                    tool_reward_list = tool_reward
                processed_tool_reward = average_before_first_negative_one(tool_reward_list)
                if processed_tool_reward is None:
                    processed_tool_reward = 0.0
            
            # 计算总分
            if accuracy_reward:
                total_score = 8  # 答对了给8分
            else:
                total_score = accuracy_reward + processed_tool_reward  # 答错了，按照工具使用情况给分
        
        else:
            # 格式不正确，所有奖励为0
            total_score = 0.0
            processed_tool_reward = 0.0
        
        # 存储详细的奖励信息
        reward_details.append({
            "score": total_score,
            "format_reward": format_reward,
            "accuracy_reward": accuracy_reward,
            "tool_reward": processed_tool_reward,
            "data_source": data_source
        })
    
    # 如果有 web 任务，并行处理它们
    if web_tasks and CLIENT:
        print(f"开始评估 {len(web_tasks)} 个 web 任务...")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_task = {
                executor.submit(llm_judge_web, task['solution_str'], task['ground_truth'], task['question']): task 
                for task in web_tasks
            }
            
            for future in tqdm(as_completed(future_to_task), total=len(web_tasks), desc="Web 任务评估"):
                task = future_to_task[future]
                task_index = task['index']
                
                try:
                    # 获取 LLM 判断的结果
                    accuracy = future.result()
                    accuracy_reward = 4 if accuracy else 0
                    
                    # 更新该任务的奖励信息
                    reward_details[task_index]["accuracy_reward"] = accuracy_reward
                    
                    # 重新计算总分
                    if accuracy_reward:
                        reward_details[task_index]["score"] = 8
                    else:
                        reward_details[task_index]["score"] = accuracy_reward + reward_details[task_index]["tool_reward"]
                        
                except Exception as e:
                    print(f"Web 任务 {task_index} 评估失败: {e}")
    
    if vs_tasks and CLIENT:
        print(f"开始评估 {len(vs_tasks)} 个 visual_search 任务...")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_task = {
                executor.submit(llm_judge_web, task['solution_str'], task['ground_truth'], task['question']): task 
                for task in vs_tasks
            }
            
            for future in tqdm(as_completed(future_to_task), total=len(vs_tasks), desc="visual_search 任务评估"):
                task = future_to_task[future]
                task_index = task['index']
                
                try:
                    # 获取 LLM 判断的结果
                    accuracy = future.result()
                    accuracy_reward = 4 if accuracy else 0
                    
                    # 更新该任务的奖励信息
                    reward_details[task_index]["accuracy_reward"] = accuracy_reward
                    
                    # 重新计算总分
                    if accuracy_reward:
                        reward_details[task_index]["score"] = 8
                    else:
                        reward_details[task_index]["score"] = accuracy_reward + reward_details[task_index]["tool_reward"]
                        
                except Exception as e:
                    print(f"visual_search 任务 {task_index} 评估失败: {e}")
    
    return reward_details


def llm_judge_web(solution_str, ground_truth, question="", timeout=300):
    """
    使用 LLM 判断 web 任务的答案是否正确，超时返回0
    
    Args:
        solution_str: 模型的回答
        ground_truth: 标准答案
        question: 问题
        timeout: 超时时间(秒)
        
    Returns:
        float: 1.0 表示正确，0.0 表示错误或超时
    """
    if not CLIENT:
        raise ValueError("OpenAI 客户端未初始化，无法评估 web 任务")
        
    response_pattern = r'<response>(.*?)</response>'
    response_match = re.search(response_pattern, solution_str, re.DOTALL)
    if not response_match:
        return 0.0
    
    response_content = response_match.group(1).strip()

    try:
        # 构建提示
        full_prompt = get_full_prompt(response_content, ground_truth, question)
        
        with no_proxy():
            chat_response = CLIENT.chat.completions.create(
                model=openai_model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that evaluates answers."},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.0,
                max_tokens=10,
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
            
    except requests.exceptions.Timeout:
        print(f"LLM 评估请求超时，计为 0 分。")
        return 0.0
    except Exception as e:
        print(f"调用 LLM API 时发生错误: {e}")
        return 0.0


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


def calculate_path_ver_accuracy(solution_str, ground_truth):
    """
    计算path_ver任务的准确性奖励
    
    参数:
        solution_str (str): 模型的回复字符串
        ground_truth (bool): 真实答案，True表示安全，False表示不安全
        
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
        model_answer = extract_boxed_yes_or_no(response_content)
        
        # 方法2：如果没有找到boxed，则直接查找yes/no
        if model_answer is None:
            model_answer = extract_yes_no_answer(response_content)
        
        # 如果仍然没有找到有效答案，返回0
        if model_answer is None:
            return 0
        
        # 比较模型答案和真实答案
        if ground_truth == "yes" and model_answer == "yes":
            return 1
        elif ground_truth == "no" and model_answer == "no":
            return 1
        else:
            return 0
            
    except Exception as e:
        print(f"Error in path_ver accuracy calculation: {e}")
        return 0


def extract_boxed_yes_or_no(text):
    """
    从文本中提取\\boxed{}中的答案
    
    参数:
        text (str): 要提取的文本
        
    返回:
        str or None: "yes"、"no" 或 None
    """
    # 匹配boxed{...}模式
    boxed_pattern = r'boxed\{([^}]+)\}'
    boxed_match = re.search(boxed_pattern, text, re.IGNORECASE)
    
    if boxed_match:
        boxed_content = boxed_match.group(1).strip().lower()
        if "yes" in boxed_content:
            return "yes"
        elif "no" in boxed_content:
            return "no"
    
    return None


def extract_yes_no_answer(text):
    """
    从文本中直接查找yes/no
    
    参数:
        text (str): 要查找的文本
        
    返回:
        str or None: "yes"、"no" 或 None
    """
    text_lower = text.lower()
    
    # 统计yes和no的出现次数
    yes_count = text_lower.count("yes")
    no_count = text_lower.count("no")
    
    # 如果只有yes，没有no
    if yes_count > 0 and no_count == 0:
        return "yes"
    # 如果只有no，没有yes
    elif no_count > 0 and yes_count == 0:
        return "no"
    # 如果都有或者都没有，返回None
    else:
        return None


def calculate_path_nav_accuracy(solution_str, ground_truth):
    """
    计算path_nav任务的准确性奖励
    
    参数:
        solution_str (str): 模型的回复字符串
        ground_truth (list): gym地图格式，如[["F", "H", "S", "H", "F"], ...]
        
    返回:
        int: 1表示路径有效且到达目标，0表示路径无效或未到达目标
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
        
        # 提取路径序列
        action_sequence = extract_path_sequence(response_content)
        
        if not action_sequence:
            return 0
        
        # 在gym环境中验证路径
        return validate_path_in_gym(action_sequence, ground_truth)
            
    except Exception as e:
        print(f"Error in path_nav accuracy calculation: {e}")
        return 0


def extract_path_sequence(text):
    """
    从文本中提取路径序列
    
    参数:
        text (str): 要提取的文本
        
    返回:
        list: 动作序列，如['L', 'R', 'U', 'D']，如果提取失败返回空列表
    """
    # 方法1：尝试从\\boxed{}中提取
    boxed_pattern = r'boxed\{([^}]+)\}'
    boxed_match = re.search(boxed_pattern, text, re.IGNORECASE)
    
    if boxed_match:
        boxed_content = boxed_match.group(1).strip()
        # 从boxed内容中提取路径
        actions = extract_actions_from_string(boxed_content)
        if actions:
            return actions
    
    # 方法2：尝试从整个文本中提取路径模式
    # 匹配类似 "L,R,U,D" 或 "L, R, U, D" 的模式
    path_pattern = r'([UDLR](?:\s*,\s*[UDLR])*)'
    path_matches = re.findall(path_pattern, text, re.IGNORECASE)
    
    for match in path_matches:
        actions = extract_actions_from_string(match)
        if actions:
            return actions
    
    # 方法3：提取所有UDLR字符作为连续序列
    actions = []
    for char in text.upper():
        if char in ['U', 'D', 'L', 'R']:
            actions.append(char)
    
    # 如果找到的动作太少，可能不是有效路径
    if len(actions) >= 1:
        return actions
    
    return []


def extract_actions_from_string(action_str):
    """
    从字符串中提取动作列表
    
    参数:
        action_str (str): 包含动作的字符串
        
    返回:
        list: 动作列表
    """
    actions = []
    
    # 按逗号分割
    if ',' in action_str:
        for action in action_str.split(','):
            action = action.strip().upper()
            if action in ['U', 'D', 'L', 'R']:
                actions.append(action)
    else:
        # 没有逗号，提取所有UDLR字符
        for char in action_str.upper():
            if char in ['U', 'D', 'L', 'R']:
                actions.append(char)
    
    return actions


def validate_path_in_gym(actions, gym_map):
    """
    在gym环境中验证路径的有效性
    
    参数:
        actions (list): 动作序列，如['L', 'R', 'U', 'D']
        gym_map (list): gym地图格式
        
    返回:
        int: 1表示路径有效且到达目标，0表示路径无效或未到达目标
    """
    try:
        # 创建FrozenLake环境
        gym_map = ast.literal_eval(gym_map) # 将字符串转换为列表
        env = gym.make('FrozenLake-v1', desc=gym_map, is_slippery=False)
        env.reset()
        
        # 动作映射
        action_mapping = {'L': 0, 'D': 1, 'R': 2, 'U': 3}
        
        # 执行动作序列
        for action in actions:
            if action not in action_mapping:
                env.close()
                return 0  # 无效动作
            
            observation, reward, terminated, truncated, info = env.step(action_mapping[action])
            
            if terminated:
                env.close()
                # 如果到达目标 (reward > 0)
                if reward > 0:
                    return 1
                # 如果掉入洞中 (reward = 0)
                else:
                    return 0
        
        env.close()
        # 如果完成所有动作但没有terminated，说明没有到达目标
        return 0
    
    except Exception as e:
        print(f"Error validating path in gym: {e}")
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