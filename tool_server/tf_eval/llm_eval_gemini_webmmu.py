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
import time  # 添加time模块用于延时

from tool_server.utils.utils import process_jsonl, append_jsonl

def get_chat_template():
    """返回用于 LLM 判断的基础指令模板。"""
    return """
Below are two answers to a question. [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question. Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent. Furthermore, if [Model Answer] includes the key information in [Standard Answer], it is also considered consistent.
If they are consistent, Judgement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""

def get_prompt_examples():
    """返回用于 Few-shot 学习的示例。"""
    example_1 = """[Question]: Is the countertop tan or blue?\n[Standard Answer]: A. The countertop is tan.\n[Model_answer] : tan\nJudgement: 1"""
    example_2 = """[Question]: On which side of the picture is the barrier?\n[Standard Answer]: The barrier is on the left side of the picture.\n[Model_answer] : left.\nJudgement: 1"""
    example_3 = """[Question]: Where should a user click to return to the homepage?\n[Standard Answer]: The user can click the "HOME" link located in the top navigation bar, right under the header in {"x1":558.56,"x2":603.99,"y1":236.74,"y2":249.65} coordinates.\n[Model_answer] : HOME.\nJudgement: 1"""
    example_4 = """[Question]: How can the user read the tweets made by Cafe Hound on this given screenshot?\n[Standard Answer]: The user can navigate towards the right side of this webpage, under the "CAFE HOUND" tab, and click the "Tweets by cafehound" located in {"x1":789.87, "x2":917.96, "y1":387.04, "y2":400.26} coordinates.\n[Model_answer] : Tweets by cafehound.\nJudgement: 1"""
    example_5 = """[Question]: Who is wearing pants?\n[Standard Answer]: The boy is wearing pants.\n[Model_answer] : The girl in the picture is wearing pants.\nJudgement: 0"""
    return [example_1, example_2, example_3, example_4, example_5]

def get_full_prompt(predict_str, ground_truth_str, question):
    """构建最终发送给 LLM 的完整 Prompt。"""
    # question = question
    chat_template = get_chat_template()
    examples = get_prompt_examples()
    demo_prompt = chat_template + "\n\n".join(examples) + "\n\n"
    test_prompt = f"""[Question]: {question}\n[Standard Answer]: {ground_truth_str}\n[Model_answer] : {predict_str}\nJudgement:"""
    return f'{demo_prompt}{test_prompt}'

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

def get_llm_score(client, model_name, pred, gold, question, max_retry=3):
    """向 vLLM 服务发送请求并解析判断分数，包含重试机制。"""
    # 将 None 转换为字符串 "None" 以便处理
    pred_str = str(pred)
    gold_str = str(gold)

    full_prompt = get_full_prompt(pred_str, gold_str, question)
    
    for attempt in range(max_retry + 1):  # 总共尝试 max_retry + 1 次
        try:
            # print("开始请求模型:", model_name)
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that evaluates answers."}]},
                    {"role": "user", "content": [{"type": "text", "text": full_prompt}]},
                ],
            )
            
            # 检查响应是否为空
            if (not hasattr(chat_response, 'choices') or 
                len(chat_response.choices) == 0 or 
                not hasattr(chat_response.choices[0], 'message') or 
                not hasattr(chat_response.choices[0].message, 'content') or 
                chat_response.choices[0].message.content is None):
                
                if attempt < max_retry:
                    wait_time = 2 ** attempt
                    print(f"警告: 第 {attempt + 1} 次请求返回空响应，{wait_time} 秒后重试...", file=sys.stderr)
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"错误: 经过 {max_retry + 1} 次尝试后仍然收到空响应", file=sys.stderr)
                    return 0
            
            response = chat_response.choices[0].message.content.strip()
            # print("response:", response)

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
            if attempt < max_retry:
                wait_time = 2 ** attempt  # 指数退避：1秒、2秒、4秒...
                print(f"警告: 第 {attempt + 1} 次请求失败: {e}，{wait_time} 秒后重试...", file=sys.stderr)
                time.sleep(wait_time)
            else:
                print(f"错误: 经过 {max_retry + 1} 次尝试后仍然失败: {e}", file=sys.stderr)
                return 0
    
    return 0

def _calculate_average(scores_list):
    """辅助函数，用于安全地计算平均分。"""
    if not scores_list:
        return 0.0
    return sum(scores_list) / len(scores_list)

def process_and_save_jsonl(jsonl_path, client, model_name, max_retry=3):
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
            score = get_llm_score(client, model_name, pred_answer, gold_answer, question, max_retry)
        
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
    output_path = p.parent / f"{p.stem}_llm_gemini.jsonl"

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


import concurrent.futures

def process_item(args):
    """处理单个项目并返回其分数和更新后的项目"""
    item, client, model_name, question_key, max_retry = args
    gold_answer = item.get('gold')
    pred_answer = item.get('pred')  # pred_answer可能是None
    question = item.get(question_key)
    
    if gold_answer is None:
        print(f"警告: 记录缺少 'gold' 键，计为 0 分: {item}", file=sys.stderr)
        score = 0
    else:
        score = get_llm_score(client, model_name, pred_answer, gold_answer, question, max_retry)
    
    item['llm_score'] = score
    return score, item

def process_and_save_jsonl_parallel(jsonl_path, client, model_name, num_threads=64, question_key='question', max_retry=3):
    """
    并行处理单个 jsonl 文件：读取、评分、计算平均分并保存到新文件。
    如果路径包含 'charxiv'，则额外计算并保存分类平均分。
    如果路径包含 'vstar'，则额外计算并保存分类平均分。
    
    参数:
        jsonl_path: 输入的JSONL文件路径
        client: OpenAI API客户端
        model_name: 要使用的模型名称
        num_threads: 使用的线程数量，默认为64
        question_key: 问题字段的键名，默认为'question'
        max_retry: 最大重试次数，默认为3
    """
    print(f"--- 开始并行处理文件 (使用 {num_threads} 线程): {jsonl_path} ---")
    
    resfile_data = process_jsonl(jsonl_path)
    if not resfile_data:
        print(f"错误: 未能从 {jsonl_path} 读取到任何数据。", file=sys.stderr)
        return

    result_dict = {}
    for item in resfile_data:
        result_dict[item['task_name']] = item
    
    resfile_data = [v for k,v in result_dict.items()]
    
    for data_item_idx,data_item in enumerate(resfile_data):
        task_name = data_item.get('task_name', 'unknown_task')
        print(f"Processing the {data_item_idx}/{len(resfile_data)} data item in {jsonl_path}, Task Name: {task_name}")
        try:
            compare_data = data_item['compare_logs']
        except (KeyError, IndexError):
            print(f"错误: 在 {jsonl_path} 的第一行中未找到 'compare_logs' 键。", file=sys.stderr)
            return

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
        tasks = [(item, client, model_name, question_key, max_retry) for item in compare_data]
        
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

        # 构建新文件名并写入
        p = Path(jsonl_path)
        output_path = p.parent / f"{p.stem}_llm_gemini.jsonl"

        append_jsonl(data_item,output_path)

def main():
    parser = argparse.ArgumentParser(description="使用 LLM 评估 JSONL 文件中的预测结果。")
    # 让 jsonl_paths 成为可选参数，如果未提供，则使用硬编码的列表
    parser.add_argument('--jsonl_paths', nargs='*', help="一个或多个待处理的 .jsonl 文件的路径。如果未提供，将使用代码中定义的默认路径。")
    parser.add_argument('--api_url', type=str, default="http://SH-IDC1-10-140-37-71:16113/v1", help="vLLM 服务的 API base URL。")
    parser.add_argument('--api_key', type=str, default="not-needed", help="API key")
    parser.add_argument('--model_name', type=str, default="/mnt/petrelfs/share_data/ai4good_shared/models/Qwen/Qwen2.5-72B-Instruct", help="要使用的模型名称或路径。")
    parser.add_argument("--num_threads", type=int, default=16, help="并行处理时使用的线程数。")
    parser.add_argument("--max_retry", type=int, default=8, help="API请求失败时的最大重试次数。")
    
    args = parser.parse_args()

    # 如果命令行没有提供路径，则使用您预设的列表
    if args.jsonl_paths:
        jsonl_paths = args.jsonl_paths
    else:
        print("未从命令行接收到路径，使用代码中预设的路径列表。")
        # 运行记得关代理
        jsonl_paths = [
            # # webguichat
            # "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tf_eval/scripts/logs/ckpt/web_guichat/7b_wo_tool_output.jsonl",
            # "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tf_eval/scripts/logs/ckpt/web_guichat/32b_wo_tool_output.jsonl",
            # "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tf_eval/scripts/logs/ckpt/web_guichat/72b_wo_tool_output.jsonl"
            # webmmu
            "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tf_eval/scripts/logs/ckpt/webmmu/7b_wo_tool_output.jsonl",
            "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tf_eval/scripts/logs/ckpt/webmmu/32b_wo_tool_output.jsonl",
            "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tf_eval/scripts/logs/ckpt/webmmu/72b_wo_tool_output.jsonl"
        ]
    
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

    # 循环处理每个文件
    if args.num_threads > 1:
        print(f"使用 {args.num_threads} 个线程并行处理 JSONL 文件，最大重试次数: {args.max_retry}...")
        for path in jsonl_paths:
            process_and_save_jsonl_parallel(path, client, args.model_name, num_threads=args.num_threads, max_retry=args.max_retry)
    else:
        print(f"顺序处理 JSONL 文件，最大重试次数: {args.max_retry}...")
        for path in jsonl_paths:
            process_and_save_jsonl(path, client, args.model_name, max_retry=args.max_retry)

if __name__ == '__main__':
    main()