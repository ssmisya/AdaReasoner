import os
import json
import argparse
import openai
import contextlib
from pathlib import Path
from tqdm import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime

from tool_server.utils.utils import process_jsonl, append_jsonl

# ==============================================================================
# 步骤 1: 定义一个用于临时禁用代理的上下文管理器
# ==============================================================================
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

def get_chat_template():
    """返回用于 LLM 判断的基础指令模板。"""
    return """
You are an expert evaluator. Your goal is to determine if a [Model Answer] correctly and factually answers a [Question] when compared against a [Standard Answer].

**Core Evaluation Principle:**
The [Model Answer] is considered consistent if it contains the **essential key information** present in the [Standard Answer]. The [Model Answer] is allowed to be much more verbose, conversational, and include additional correct context or explanations. Your primary task is to **verify the presence of the core facts**, not to penalize extra information. **If a question asks for specific formatting like coordinates or tables, but the model identifies the correct core element textually, it should still be considered consistent.**

- **Consistent (Judgement: 1):** The [Model Answer] successfully identifies the main point or action from the [Standard Answer]. For example, if the standard answer is to "click button A", the model answer is consistent if it mentions clicking or interacting with "button A", even if it's surrounded by other text.
- **Inconsistent (Judgement: 0):** The [Model Answer] fails to mention the key information, provides contradictory information, or hallucinates a different solution.

**Output Format:**
Just output `Judgement: 1` or `Judgement: 0`. Do not output anything else.
"""

def get_full_prompt(predict_str, ground_truth_str, question):
    """构建最终发送给 LLM 的完整 Prompt。"""
    question = question
    # prefix = "Analyze the website screenshot and provide a detailed answer to the question. If the question involves locating or interacting with specific elements on the screen, include the bounding box coordinates [x_min, y_min, x_max, y_max] in your response.\n"
    # question = question.replace(prefix,'',1)
    chat_template = get_chat_template()
    # demo_prompt = chat_template + "\n\n".join(examples) + "\n\n"
    demo_prompt = chat_template + "\n\n"
    test_prompt = f"""[Question]: {question}\n[Standard Answer]: {ground_truth_str}\n[Model_answer] : {predict_str}\nJudgement:"""
    return f'{demo_prompt}{test_prompt}'

# ==============================================================================
# 步骤 3: 核心功能函数
# ==============================================================================
def read_jsonl(file_path):
    """读取JSONL文件并解析其内容"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}", file=sys.stderr)
        return None
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                data.append(record)
            except json.JSONDecodeError:
                print(f"警告: 跳过无法解析的JSON行: {line.strip()}", file=sys.stderr)
                continue
                
    print(f"从 {Path(file_path).name} 中共读取了 {len(data)} 行记录。")
    return data

def get_llm_score(client, model_name, pred, gold, question):
    """向 vLLM 服务发送请求并解析判断分数。"""
    # 将 None 转换为字符串 "None" 以便处理
    pred_str = str(pred)
    gold_str = str(gold)

    full_prompt = get_full_prompt(pred_str, gold_str, question)
    
    try:
        with no_proxy():
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    # {"role": "system", "content": "You are a helpful assistant that evaluates answers."},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.0,
                max_tokens=10
            )
        response = chat_response.choices[0].message.content.strip()

        if 'Judgement:' in response:
            response = response.split('Judgement:')[-1].strip()
        
        if '1' in response:
            return 1
        elif '0' in response:
            return 0
        else:
            print(f"警告: 无法解析的 LLM 响应 '{response}', 计为 0 分。", file=sys.stderr)
            return 0
            
    except Exception as e:
        print(f"错误: 调用 API 时发生错误: {e}", file=sys.stderr)
        return 0

def _calculate_average(scores_list):
    """辅助函数，用于安全地计算平均分。"""
    if not scores_list:
        return 0.0
    return sum(scores_list) / len(scores_list)

def generate_timestamp_filename(prefix="test_results", suffix=".json"):
    """生成基于当前时间的文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{prefix}_{timestamp}{suffix}"

def extract_image_name_from_path(jsonl_path):
    """从jsonl文件路径中提取图片名称或模型名称"""
    path = Path(jsonl_path)
    # 移除_llm后缀（如果存在）
    stem = path.stem.replace('_llm', '')
    return stem

def append_result_to_summary(file_result, summary_path):
    """将单个文件的结果追加到汇总文件中"""
    try:
        # 如果文件不存在，创建初始结构
        if not summary_path.exists():
            initial_summary = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_files_processed': 0,
                'files': []
            }
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(initial_summary, f, ensure_ascii=False, indent=2)
        
        # 读取现有内容
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        # 添加新的文件结果
        summary_data['files'].append(file_result)
        summary_data['total_files_processed'] += 1
        summary_data['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 写回文件
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
        print(f"结果已追加到汇总文件: {summary_path}")
        
    except Exception as e:
        print(f"错误: 追加结果到汇总文件失败: {e}", file=sys.stderr)

def process_and_save_jsonl(jsonl_path, client, model_name):
    """
    处理单个 jsonl 文件：读取、评分、计算平均分并保存到新文件。
    如果路径包含 'charxiv'，则额外计算并保存分类平均分。
    如果路径包含 'vstar'，则额外计算并保存分类平均分。
    """
    print(f"--- 开始处理文件: {jsonl_path} ---")
    
    data = read_jsonl(jsonl_path)
    if not data:
        print(f"错误: 未能从 {jsonl_path} 读取到任何数据。", file=sys.stderr)
        return

    # 根据您的代码，核心数据在 'compare_logs' 键下
    try:
        compare_data = data[0]['compare_logs']
    except (KeyError, IndexError):
        print(f"错误: 在 {jsonl_path} 的第一行中未找到 'compare_logs' 键。", file=sys.stderr)
        return
    
    is_charxiv = 'charxiv' in jsonl_path
    is_vstar = 'vstar' in jsonl_path
    is_webmmu = 'webmmu' in jsonl_path

    # 初始化分数追踪器
    updated_data = []
    all_scores = []
    descriptive_scores = [] if is_charxiv else None
    reasoning_scores = [] if is_charxiv else None
    direct_attributes_scores = [] if is_vstar else None
    relative_position_scores = [] if is_vstar else None
    functional_scores = [] if is_webmmu else None
    general_image_understanding_scores = [] if is_webmmu else None
    complex_reasoning_scores = [] if is_webmmu else None
    
    # 使用 tqdm 创建进度条
    for item in tqdm(compare_data, desc=f"评分 {Path(jsonl_path).name}"):
        
        gold_answer = item.get('gold')
        pred_answer = item.get('pred') # pred_answer可能是None
        question = item.get('question')
        if gold_answer is None:
            print(f"警告: 记录缺少 'gold' 键，计为 0 分: {item}", file=sys.stderr)
            score = 0
        else:
            score = get_llm_score(client, model_name, pred_answer, gold_answer, question)
        
        item['llm_score'] = score
        all_scores.append(score)
        
        if is_charxiv:
            item_type = item.get('Type')
            if item_type == 'descriptive':
                descriptive_scores.append(score)
            elif item_type == 'reasoning':
                reasoning_scores.append(score)
        if is_vstar:
            item_type = item.get('Type')
            if item_type == 'direct_attributes':
                direct_attributes_scores.append(score)
            elif item_type == 'relative_position':
                relative_position_scores.append(score)
        if is_webmmu:
            item_type = item.get('category')
            if item_type == 'Functional':
                functional_scores.append(score)
            elif item_type == 'General Image Understanding':
                general_image_understanding_scores.append(score)
            elif item_type == 'Complex Reasoning':
                complex_reasoning_scores.append(score)
        updated_data.append(item)

    # 计算平均分
    avg_score_summary = {}
    average_score = _calculate_average(all_scores)
    avg_score_summary["average_llm_score"] = average_score

    if is_charxiv:
        avg_descriptive = _calculate_average(descriptive_scores)
        avg_reasoning = _calculate_average(reasoning_scores)
        avg_score_summary["average_descriptive_score"] = avg_descriptive
        avg_score_summary["average_reasoning_score"] = avg_reasoning
        print(f"文件处理完成。总平均分: {average_score:.4f}, Descriptive 平均分: {avg_descriptive:.4f}, Reasoning 平均分: {avg_reasoning:.4f}")
    elif is_vstar:
        avg_direct_attributes = _calculate_average(direct_attributes_scores)
        avg_relative_position = _calculate_average(relative_position_scores)
        avg_score_summary["average_direct_attributes_score"] = avg_direct_attributes
        avg_score_summary["average_relative_position_score"] = avg_relative_position
        print(f"文件处理完成。总平均分: {average_score:.4f}, Direct Attributes 平均分: {avg_direct_attributes:.4f}, Relative Position 平均分: {avg_relative_position:.4f}")
    elif is_webmmu:
        avg_functional = _calculate_average(functional_scores)
        avg_general_image_understanding = _calculate_average(general_image_understanding_scores)
        avg_complex_reasoning = _calculate_average(complex_reasoning_scores)
        avg_score_summary["average_functional_score"] = avg_functional
        avg_score_summary["average_general_image_understanding_score"] = avg_general_image_understanding
        avg_score_summary["average_complex_reasoning_score"] = avg_complex_reasoning
        print(f"文件处理完成。总平均分: {average_score:.4f}, Functional 平均分: {avg_functional:.4f}, General Image Understanding 平均分: {avg_general_image_understanding:.4f}, Complex Reasoning 平均分: {avg_complex_reasoning:.4f}")
    else:
        print(f"文件处理完成。平均 LLM 分数: {average_score:.4f}")

    # 构建新文件名并写入
    p = Path(jsonl_path)
    output_path = p.parent / f"{p.stem}_llm.jsonl"

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # 第一行写入包含所有平均分的摘要
            f.write(json.dumps(avg_score_summary) + '\n')
            
            # 写入更新后的数据
            # 注意：此处我们写入更新后的 compare_logs 内容，而不是原始的 data 结构
            for item in updated_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"结果已保存至: {output_path}\n")
    except IOError as e:
        print(f"错误: 写入文件失败 {output_path}: {e}", file=sys.stderr)


# ==============================================================================
# 步骤 3a: 并行处理函数
# ==============================================================================
import concurrent.futures

def process_item(args):
    """处理单个项目并返回其分数和更新后的项目"""
    item, client, model_name, question_key = args
    gold_answer = item.get('gold')
    pred_answer = item.get('pred')  # pred_answer可能是None
    question = item.get(question_key)
    
    if gold_answer is None:
        print(f"警告: 记录缺少 'gold' 键，计为 0 分: {item}", file=sys.stderr)
        score = 0
    else:
        score = get_llm_score(client, model_name, pred_answer, gold_answer, question)
    
    item['llm_score'] = score
    return score, item

def process_and_save_jsonl_parallel(jsonl_path, client, model_name, num_threads=64, question_key='question'):
    """
    并行处理单个 jsonl 文件：读取、评分、计算平均分并保存到新文件。
    
    参数:
        jsonl_path: 输入的JSONL文件路径
        client: OpenAI API客户端
        model_name: 要使用的模型名称
        num_threads: 使用的线程数量，默认为64
        question_key: 问题字段的键名，默认为'question'
    
    返回:
        dict: 包含文件处理结果的字典
    """
    print(f"--- 开始并行处理文件 (使用 {num_threads} 线程): {jsonl_path} ---")
    
    resfile_data = process_jsonl(jsonl_path)
    if not resfile_data:
        print(f"错误: 未能从 {jsonl_path} 读取到任何数据。", file=sys.stderr)
        return None

    result_dict = {}
    for item in resfile_data:
        result_dict[item['task_name']] = item
    
    resfile_data = [v for k,v in result_dict.items()]
    
    # 初始化汇总结果
    file_summary = {
        'file_path': str(jsonl_path),
        'file_name': extract_image_name_from_path(jsonl_path),
        'tasks': []
    }
    
    for data_item_idx,data_item in enumerate(resfile_data):
        task_name = data_item.get('task_name', 'unknown_task')
        print(f"Processing the {data_item_idx}/{len(resfile_data)} data item in {jsonl_path}, Task Name: {task_name}")
        try:
            compare_data = data_item['compare_logs']
        except (KeyError, IndexError):
            print(f"错误: 在 {jsonl_path} 的第一行中未找到 'compare_logs' 键。", file=sys.stderr)
            continue
        
        is_charxiv = task_name == "charxiv"
        is_vstar = task_name == "vstar"
        is_webmmu = task_name == "webmmu"
        # 初始化线程安全的数据结构
        lock = threading.Lock()
        updated_data = []
        all_scores = []
        descriptive_scores = [] if is_charxiv else None
        reasoning_scores = [] if is_charxiv else None
        direct_attributes_scores = [] if is_vstar else None
        relative_position_scores = [] if is_vstar else None
        functional_scores = [] if is_webmmu else None
        general_image_understanding_scores = [] if is_webmmu else None
        complex_reasoning_scores = [] if is_webmmu else None
        # 准备任务参数
        tasks = [(item, client, model_name, question_key) for item in compare_data]
        
        # 创建进度条
        progress_bar = tqdm(total=len(tasks), desc=f"并行评分 {Path(jsonl_path).name}")
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 提交所有任务
            future_to_item = {executor.submit(process_item, task): task[0] for task in tasks}
            
            # 处理完成的任务
            for future in concurrent.futures.as_completed(future_to_item):
                original_item = future_to_item[future]
                try:
                    score, updated_item = future.result()
                    
                    with lock:
                        all_scores.append(score)
                        updated_data.append(updated_item)
                        
                        if is_charxiv:
                            item_type = updated_item.get('Type')
                            if item_type == 'descriptive':
                                descriptive_scores.append(score)
                            elif item_type == 'reasoning':
                                reasoning_scores.append(score)
                        
                        if is_vstar:
                            item_type = updated_item.get('category')
                            if item_type == 'direct_attributes':
                                direct_attributes_scores.append(score)
                            elif item_type == 'relative_position':
                                relative_position_scores.append(score)
                        if is_webmmu:
                            item_type = updated_item.get('category')
                            if item_type == 'Functional':
                                functional_scores.append(score)
                            elif item_type == 'General Image Understanding':
                                general_image_understanding_scores.append(score)
                            elif item_type == 'Complex Reasoning':
                                complex_reasoning_scores.append(score)
                        # 更新进度条
                        progress_bar.update(1)
                        
                except Exception as e:
                    print(f"错误: 处理项目时出错: {e}", file=sys.stderr)
                    progress_bar.update(1)
        
        progress_bar.close()
        
        data_item['compare_logs'] = updated_data
        # 计算平均分
        avg_score_summary = {}
        average_score = _calculate_average(all_scores)
        avg_score_summary["average_llm_score"] = average_score

        if is_charxiv:
            avg_descriptive = _calculate_average(descriptive_scores)
            avg_reasoning = _calculate_average(reasoning_scores)
            avg_score_summary["average_descriptive_score"] = avg_descriptive
            avg_score_summary["average_reasoning_score"] = avg_reasoning
            print(f"文件处理完成。总平均分: {average_score:.4f}, Descriptive 平均分: {avg_descriptive:.4f}, Reasoning 平均分: {avg_reasoning:.4f}")
        elif is_vstar:
            avg_direct_attributes = _calculate_average(direct_attributes_scores)
            avg_relative_position = _calculate_average(relative_position_scores)
            avg_score_summary["average_direct_attributes_score"] = avg_direct_attributes
            avg_score_summary["average_relative_position_score"] = avg_relative_position
            print(f"文件处理完成。总平均分: {average_score:.4f}, Direct Attributes 平均分: {avg_direct_attributes:.4f}, Relative Position 平均分: {avg_relative_position:.4f}")
        elif is_webmmu:
            avg_functional = _calculate_average(functional_scores)
            avg_general_image_understanding = _calculate_average(general_image_understanding_scores)
            avg_complex_reasoning = _calculate_average(complex_reasoning_scores)
            avg_score_summary["average_functional_score"] = avg_functional
            avg_score_summary["average_general_image_understanding_score"] = avg_general_image_understanding
            avg_score_summary["average_complex_reasoning_score"] = avg_complex_reasoning
            print(f"文件处理完成。总平均分: {average_score:.4f}, Functional 平均分: {avg_functional:.4f}, General Image Understanding 平均分: {avg_general_image_understanding:.4f}, Complex Reasoning 平均分: {avg_complex_reasoning:.4f}")
        else:
            print(f"文件处理完成。平均 LLM 分数: {average_score:.4f}")
        
        data_item["llm_eval_summary"] = avg_score_summary
        
        # 添加到汇总结果
        task_summary = {
            'task_name': task_name,
            'average_llm_score': average_score,
            'total_items': len(all_scores)
        }
        if is_charxiv:
            task_summary.update({
                'average_descriptive_score': avg_descriptive,
                'average_reasoning_score': avg_reasoning
            })
        elif is_vstar:
            task_summary.update({
                'average_direct_attributes_score': avg_direct_attributes,
                'average_relative_position_score': avg_relative_position
            })
        elif is_webmmu:
            task_summary.update({
                'average_functional_score': avg_functional,
                'average_general_image_understanding_score': avg_general_image_understanding,
                'average_complex_reasoning_score': avg_complex_reasoning
            })
        
        file_summary['tasks'].append(task_summary)

        # 构建新文件名并写入
        p = Path(jsonl_path)
        output_path = p.parent / f"{p.stem}_llm.jsonl"

        append_jsonl(data_item,output_path)
    
    return file_summary


# ==============================================================================
# 步骤 4: 主程序入口
# ==============================================================================
def main():
    log_file_path = "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tf_eval/scripts/logs/ckpt/eval_logs"

    parser = argparse.ArgumentParser(description="使用 LLM 评估 JSONL 文件中的预测结果。")
    # 让 jsonl_paths 成为可选参数，如果未提供，则使用硬编码的列表
    parser.add_argument('--jsonl_paths', nargs='*', help="一个或多个待处理的 .jsonl 文件的路径。如果未提供，将使用代码中定义的默认路径。")
    parser.add_argument('--api_url', type=str, default="http://SH-IDC1-10-140-37-111:16113/v1", help="vLLM 服务的 API base URL。")
    # parser.add_argument('--api_url', type=str, default="http://SH-IDC1-10-140-37-82:16113/v1", help="vLLM 服务的 API base URL。")
    parser.add_argument('--api_key', type=str, default="not-needed", help="API key")
    parser.add_argument('--model_name', type=str, default="/mnt/petrelfs/share_data/ai4good_shared/models/Qwen/Qwen2.5-72B-Instruct", help="要使用的模型名称或路径。")
    parser.add_argument("--num_threads", type=int, default=64, help="并行处理时使用的线程数。")
    parser.add_argument("--summary_path", type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/logs/lm_as_a_judge/llm_eval_summary", help="汇总结果保存路径。如果未提供，将使用基于时间戳的默认文件名。")
    
    args = parser.parse_args()
    
    # 如果命令行没有提供路径，则使用您预设的列表
    if args.jsonl_paths:
        jsonl_paths = args.jsonl_paths
    else:
        print("未从命令行接收到路径，使用代码中预设的路径列表。")
        # 运行记得关代理
        # jsonl_paths = [
        #     "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tf_eval/scripts/logs/ckpt/web_guichatv1/v1_400_rl_output.jsonl",
        # ]

        jsonl_paths = [
            "/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/logs/results/unified/randomize/unified_jigsaw_randomized_sft_randomized_rl_3tasks_7b_s700",
            "/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/logs/results/unified/randomize/unified_jigsaw_randomized_sft_randomized_rl_3tasks_7b_s500",
            "/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/logs/results/unified/randomize/unified_direct_rl_7b",
            "/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/logs/results/unified/randomize/unified_jigsaw_7b",
            "/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/logs/results/unified/randomize/unified_jigsaw_7b_sft",
            "/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/logs/results/unified/randomize/unified_jigsaw_randomized_7b",
            "/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/logs/results/unified/randomize/unified_jigsaw_randomized_7b_sft",
        ]
        # tgt_dirs = [
        #     "/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/logs/results/frozen_lake_zs/gemini25flash",
        #     "/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/logs/results/unified/unified_all_randomized_sft_randomized_rl_4tasks_7b",
        #     "/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/logs/results/unified/unified_jigsaw_randomized_sft_randomized_rl_3tasks_7b"
        # ]
        jsonl_paths = [os.path.join(p, "web_raw_res.jsonl") for p in jsonl_paths]
        base_dir = "/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/logs/results/unified/sec4_all_3tasks"

        # sub_dirs = os.listdir(base_dir)
        # for sub_dir in tqdm(sub_dirs):
        #     full_sub_dir = os.path.join(base_dir, sub_dir)
        #     if not os.path.isdir(full_sub_dir):
        #         continue
        #     if not "web_raw_res.jsonl" in os.listdir(full_sub_dir):
        #         continue
        #     jsonl_paths.append(os.path.join(full_sub_dir, "web_raw_res.jsonl"))
            
        # base_dir = "/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/logs/results/unified"
        
        # sub_dirs = os.listdir(base_dir)
        # for sub_dir in tqdm(sub_dirs):
        #     full_sub_dir = os.path.join(base_dir, sub_dir)
        #     if not os.path.isdir(full_sub_dir):
        #         continue
        #     if not "web_raw_res.jsonl" in os.listdir(full_sub_dir):
        #         continue
        #     jsonl_paths.append(os.path.join(full_sub_dir, "web_raw_res.jsonl"))
    
    if len(jsonl_paths) == 1 and os.path.isdir(jsonl_paths[0]):
        # 如果只有一个路径且它是目录，则获取该目录下的所有 .jsonl 文件
        dir_path = jsonl_paths[0]
        jsonl_paths = list(Path(dir_path).glob('*.jsonl'))
        if not jsonl_paths:
            print(f"错误: 在目录 {dir_path} 中未找到任何 .jsonl 文件。", file=sys.stderr)
            return

    # 初始化 OpenAI 客户端
    try:
        client = openai.OpenAI(
            api_key=args.api_key,
            base_url=args.api_url
        )
    except Exception as e:
        print(f"错误: 初始化 OpenAI 客户端失败: {e}", file=sys.stderr)
        return

    # 生成汇总文件路径（在开始处理前就确定文件名）
    summary_filename = generate_timestamp_filename("test_results_summary", ".json")
    summary_filename = os.path.join(args.summary_path, summary_filename) if args.summary_path else summary_filename
    
    summary_path = Path(summary_filename)
    print(f"汇总结果将保存到: {summary_path}")
    
    processed_files_count = 0
    
    # 循环处理每个文件
    if args.num_threads > 1:
        print(f"使用 {args.num_threads} 个线程并行处理 JSONL 文件...")
        for path in jsonl_paths:
            result = process_and_save_jsonl_parallel(path, client, args.model_name, num_threads=args.num_threads)
            if result:
                # 立即将结果追加到汇总文件
                append_result_to_summary(result, summary_path)
                processed_files_count += 1
                
                # 打印当前文件的统计信息
                print(f"文件 {processed_files_count}: {result['file_name']}")
                for task in result['tasks']:
                    print(f"  任务: {task['task_name']}, 平均分: {task['average_llm_score']:.4f}, 项目数: {task['total_items']}")
    else:
        for path in jsonl_paths:
            # 注意：这里需要修改原来的单线程函数以返回结果，或者统一使用并行版本
            result = process_and_save_jsonl_parallel(path, client, args.model_name, num_threads=1)
            if result:
                # 立即将结果追加到汇总文件
                append_result_to_summary(result, summary_path)
                processed_files_count += 1
                
                # 打印当前文件的统计信息
                print(f"文件 {processed_files_count}: {result['file_name']}")
                for task in result['tasks']:
                    print(f"  任务: {task['task_name']}, 平均分: {task['average_llm_score']:.4f}, 项目数: {task['total_items']}")
    
    # 最终统计信息
    if processed_files_count > 0:
        print(f"\n=== 处理完成 ===")
        print(f"总共处理了 {processed_files_count} 个文件")
        print(f"汇总结果已保存到: {summary_path}")
    else:
        print("没有成功处理任何文件。", file=sys.stderr)

if __name__ == '__main__':
    main()