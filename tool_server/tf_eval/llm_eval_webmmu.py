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
import concurrent.futures

# 假设这些工具函数在你的项目中存在
from tool_server.utils.utils import process_jsonl, append_jsonl

# ==============================================================================
# 步骤 1: 定义一个用于临时禁用代理的上下文管理器 (保持不变)
# ==============================================================================
@contextlib.contextmanager
def no_proxy():
    """一个上下文管理器，可以在其作用域内临时禁用代理环境变量。"""
    proxy_keys = ['http_proxy', 'https-proxy', 'HTTP_PROXY', 'HTTPS_PROXY']
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

# ==============================================================================
# 步骤 2: 全新的 Prompt 构建函数
# ==============================================================================
def build_vqa_judge_prompt(question, model_answer, ground_truth):
    """
    根据用户提供的 VQA 评估格式构建完整的 prompt。
    """
    # 用户提供的 Few-shot 示例
    examples = [
        {
            "INPUT": {
                "question": "What is the capital of France?",
                "model_answer": "Paris",
                "ground_truth": "Paris",
            },
            "OUTPUT": {
                "rating": 1,
                "rationale": "The model’s answer matches the reference answer exactly."
            }
        },
        {
            "INPUT": {
                "question": "What is in the left of the image?",
                "model_answer": "A bus is in the left of the image.",
                "ground_truth": "A dog is in the left of the image.",
            },
            "OUTPUT": {
                "rating": 0,
                "rationale": "The model’s answer is incorrect because the reference answer is ’A dog’."
            }
        },
        {
            "INPUT": {
                "question": "Where is the burger on the table? Tell me the coordinates.",
                "model_answer": "The burger is on the table.",
                "ground_truth": "The burger is on the table at (50, 10, 150, 60).",
            },
            "OUTPUT": {
                "rating": 0,
                "rationale": "The predicted answer is incomplete because it does not provide the coordinates as requested in the question."
            }
        }
    ]

    # 当前需要评估的测试用例
    test_case = {
        "INPUT": {
            "question": question,
            "model_answer": str(model_answer),  # 确保是字符串
            "ground_truth": str(ground_truth)    # 确保是字符串
        }
    }

    # - Provides misleading or irrelevant information.

    # 主指令模板
    prompt_template = """You are evaluating a Visual Question Answering (VQA) system's response. Compare the model’s answer with the ground truth and rate its accuracy.
**Rating Scale (1 or 0)**:
1 - Correct and Complete: - The predicted answer fully matches the ground truth. - No factual errors or missing details. - Addresses the question with the correct level of specificity.
0 - Incorrect or Irrelevant: - Any factual errors or mismatches with the reference answer. - Does not address the question properly. - Provides misleading or irrelevant information.
**Examples for reference**:
{examples_json}
Question, Model Answer, and Ground Truth:
{test_case_json}
You must provide your evaluation in the following JSON format (without any extra text):
{output_format_json}"""

    # 期望的输出格式示例
    output_format_example = {
        "rating": "0 or 1",
        "rationale": "[Brief explanation of why this rating was chosen]"
    }

    # 格式化最终的 prompt 字符串
    full_prompt = prompt_template.format(
        examples_json=json.dumps(examples, indent=4),
        test_case_json=json.dumps(test_case, indent=4),
        output_format_json=json.dumps(output_format_example)
    )

    return full_prompt

# ==============================================================================
# 步骤 3: 核心功能函数 (已更新)
# ==============================================================================
def get_llm_score(client, model_name, pred, gold, question):
    """向 LLM 服务发送请求并解析判断分数和理由。"""
    # 构建新的 prompt
    full_prompt = build_vqa_judge_prompt(question, pred, gold)
    
    try:
        with no_proxy():
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.0,
                max_tokens=1000  # 增加 max_tokens 以容纳 JSON 输出和理由
            )
        response_text = chat_response.choices[0].message.content.strip()

        # 尝试解析 JSON 响应
        eval_result = json.loads(response_text)
        score = eval_result.get("rating", 0)
        rationale = eval_result.get("rationale", "No rationale provided.")

        # 确保分数是 0 或 1
        if score not in [0, 1]:
            print(f"警告: LLM 返回了无效的评分 '{score}'。默认为 0。", file=sys.stderr)
            score = 0

        return score, rationale

    except json.JSONDecodeError:
        print(f"警告: 从 LLM 响应中解码 JSON 失败: '{response_text}'。默认为 0 分。", file=sys.stderr)
        return 0, "Failed to parse LLM response."
    except Exception as e:
        print(f"错误: 调用 API 时发生错误: {e}", file=sys.stderr)
        return 0, str(e)


def _calculate_average(scores_list):
    """辅助函数，用于安全地计算平均分。"""
    if not scores_list:
        return 0.0
    return sum(scores_list) / len(scores_list)

# (process_and_save_jsonl 保持原状，但现在会接收并保存 rationale)
def process_and_save_jsonl(jsonl_path, client, model_name):
    # ... (此函数的实现与并行版本类似，这里省略以保持简洁，主要逻辑在并行版本中)
    pass 

# ==============================================================================
# 步骤 3a: 并行处理函数 (已更新)
# ==============================================================================
def process_item(args):
    """处理单个项目并返回其分数和更新后的项目"""
    item, client, model_name, question_key = args
    gold_answer = item.get('gold')
    pred_answer = item.get('pred')
    question = item.get(question_key)
    
    if gold_answer is None:
        print(f"警告: 记录缺少 'gold' 键，计为 0 分: {item}", file=sys.stderr)
        score = 0
        rationale = "Missing ground truth."
    else:
        score, rationale = get_llm_score(client, model_name, pred_answer, gold_answer, question)
    
    item['llm_score'] = score
    item['llm_rationale'] = rationale  # 添加理由
    return score, item

def process_and_save_jsonl_parallel(jsonl_path, client, model_name, num_threads=64, question_key='question'):
    """
    并行处理单个 jsonl 文件：读取、评分、计算平均分并保存到新文件。
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
            print(f"错误: 在 {jsonl_path} 的数据项中未找到 'compare_logs' 键。", file=sys.stderr)
            continue # 使用 continue 跳过这个数据项

        is_charxiv = task_name == "charxiv"
        is_vstar = task_name == "vstar"
        is_webmmu = task_name == "webmmu"
        
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
        
        tasks = [(item, client, model_name, question_key) for item in compare_data]
        
        progress_bar = tqdm(total=len(tasks), desc=f"并行评分 {Path(jsonl_path).name} - {task_name}")
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_item = {executor.submit(process_item, task): task[0] for task in tasks}
            
            for future in concurrent.futures.as_completed(future_to_item):
                try:
                    score, updated_item = future.result()
                    
                    with lock:
                        all_scores.append(score)
                        updated_data.append(updated_item)
                        
                        # 分类计分逻辑 (保持不变)
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
                        
                        progress_bar.update(1)
                        
                except Exception as e:
                    print(f"错误: 处理项目时出错: {e}", file=sys.stderr)
                    progress_bar.update(1)
        
        progress_bar.close()
        
        data_item['compare_logs'] = updated_data
        
        # 计算平均分 (逻辑保持不变)
        avg_score_summary = {}
        average_score = _calculate_average(all_scores)
        avg_score_summary["average_llm_score"] = average_score

        if is_charxiv:
            avg_descriptive = _calculate_average(descriptive_scores)
            avg_reasoning = _calculate_average(reasoning_scores)
            avg_score_summary["average_descriptive_score"] = avg_descriptive
            avg_score_summary["average_reasoning_score"] = avg_reasoning
            print(f"任务 {task_name} 处理完成。总平均分: {average_score:.4f}, Descriptive 平均分: {avg_descriptive:.4f}, Reasoning 平均分: {avg_reasoning:.4f}")
        elif is_vstar:
            avg_direct_attributes = _calculate_average(direct_attributes_scores)
            avg_relative_position = _calculate_average(relative_position_scores)
            avg_score_summary["average_direct_attributes_score"] = avg_direct_attributes
            avg_score_summary["average_relative_position_score"] = avg_relative_position
            print(f"任务 {task_name} 处理完成。总平均分: {average_score:.4f}, Direct Attributes 平均分: {avg_direct_attributes:.4f}, Relative Position 平均分: {avg_relative_position:.4f}")
        elif is_webmmu:
            avg_functional = _calculate_average(functional_scores)
            avg_general_image_understanding = _calculate_average(general_image_understanding_scores)
            avg_complex_reasoning = _calculate_average(complex_reasoning_scores)
            avg_score_summary["average_functional_score"] = avg_functional
            avg_score_summary["average_general_image_understanding_score"] = avg_general_image_understanding
            avg_score_summary["average_complex_reasoning_score"] = avg_complex_reasoning
            print(f"任务 {task_name} 处理完成。总平均分: {average_score:.4f}, Functional 平均分: {avg_functional:.4f}, General Image Understanding 平均分: {avg_general_image_understanding:.4f}, Complex Reasoning 平均分: {avg_complex_reasoning:.4f}")
        else:
            print(f"任务 {task_name} 处理完成。平均 LLM 分数: {average_score:.4f}")
        
        data_item["llm_eval_summary"] = avg_score_summary

        p = Path(jsonl_path)
        output_path = p.parent / f"{p.stem}_llm.jsonl"
        append_jsonl(data_item, output_path)
        print(f"任务 {task_name} 的结果已追加至: {output_path}\n")

# ==============================================================================
# 步骤 4: 主程序入口 (保持不变)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="使用 LLM 评估 JSONL 文件中的预测结果。")
    parser.add_argument('--jsonl_paths', nargs='*', help="一个或多个待处理的 .jsonl 文件的路径。如果未提供，将使用代码中定义的默认路径。")
    parser.add_argument('--api_url', type=str, default="http://SH-IDC1-10-140-37-71:16113/v1", help="vLLM 服务的 API base URL。")
    parser.add_argument('--api_key', type=str, default="not-needed", help="API key")
    parser.add_argument('--model_name', type=str, default="/mnt/petrelfs/share_data/ai4good_shared/models/Qwen/Qwen2.5-72B-Instruct", help="要使用的模型名称或路径。")
    parser.add_argument("--num_threads", type=int, default=64, help="并行处理时使用的线程数。")
    
    args = parser.parse_args()

    if args.jsonl_paths:
        jsonl_paths = args.jsonl_paths
    else:
        print("未从命令行接收到路径，使用代码中预设的路径列表。")
        jsonl_paths = [
            "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tf_eval/scripts/logs/ckpt/webmmu/v1_200_rl_output.jsonl"
        ]
    
    if len(jsonl_paths) == 1 and os.path.isdir(jsonl_paths[0]):
        dir_path = jsonl_paths[0]
        jsonl_paths = list(Path(dir_path).glob('*.jsonl'))
        if not jsonl_paths:
            print(f"错误: 在目录 {dir_path} 中未找到任何 .jsonl 文件。", file=sys.stderr)
            return

    try:
        client = openai.OpenAI(
            api_key=args.api_key,
            base_url=args.api_url
        )
    except Exception as e:
        print(f"错误: 初始化 OpenAI 客户端失败: {e}", file=sys.stderr)
        return

    # 循环处理每个文件
    # 注意：为了避免重复写入，在开始处理前最好清空或删除旧的输出文件
    for path in jsonl_paths:
        p = Path(path)
        output_path = p.parent / f"{p.stem}_llm.jsonl"
        if os.path.exists(output_path):
            print(f"警告: 输出文件 {output_path} 已存在，将删除后重新生成。")
            os.remove(output_path)
    
    if args.num_threads > 1:
        print(f"使用 {args.num_threads} 个线程并行处理 JSONL 文件...")
        for path in jsonl_paths:
            process_and_save_jsonl_parallel(str(path), client, args.model_name, num_threads=args.num_threads)
    else:
        # 如果需要单线程版本，请确保 process_and_save_jsonl 函数也已相应更新
        print("使用单线程处理 JSONL 文件...")
        for path in jsonl_paths:
            # 你需要实现一个与并行版本逻辑类似的单线程函数
            # 为了简单起见，这里直接调用并行版本，但线程数为1
            process_and_save_jsonl_parallel(str(path), client, args.model_name, num_threads=1)


if __name__ == '__main__':
    main()