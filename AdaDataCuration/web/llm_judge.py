import os
import json
import argparse
import openai
import contextlib
from pathlib import Path
from tqdm import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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

# ==============================================================================
# 步骤 2: Prompt 构建函数
# ==============================================================================
def get_chat_template():
    """返回用于 LLM 判断的基础指令模板。"""
    return """
Below are two answers to a question. [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question. Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judgement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.
Sometimes [Standard Answer] is <no answer>, which indicates that there is no answer to this question. If the model outputs the exact answer, it is 0. If it indicates that there is no answer, it is 1.\n\n
"""

def get_prompt_examples():
    """返回用于 Few-shot 学习的示例。"""
    example_1 = """[Question]: Is the countertop tan or blue?\n[Standard Answer]: The countertop is tan.\n[Model_answer] : tan\nJudgement: 1"""
    example_2 = """[Question]: Who is wearing pants?\n[Standard Answer]: The boy is wearing pants.\n[Model_answer] : The girl in the picture is wearing pants.\nJudgement: 0"""
    example_3 = """[Question]: What color is the towel in the center of the picture?\n[Standard Answer]: A. The towel in the center of the picture is blue.\n[Model_answer] : The towel in the center of the picture is pink.\nJudgement: 0"""
    example_4 = """[Question]: What is the current mode shown on the screen?\n[Standard Answer]:  <no answer>\n[Model_answer] : Airplane mode.\nJudgement: 0"""
    example_5 = """[Question]: What is the current mode shown on the screen?\n[Standard Answer]: <no answer>\n[Model_answer] : I can't answer that because the screenshot doesn't show the current mode.\nJudgement: 1"""
    return [example_1, example_2, example_3, example_4, example_5]

def get_full_prompt(predict_str, ground_truth_str, question):
    """构建最终发送给 LLM 的完整 Prompt。"""
    chat_template = get_chat_template()
    examples = get_prompt_examples()
    demo_prompt = chat_template + "\n\n".join(examples) + "\n\n"
    test_prompt = f"""[Question]: {question}\n[Standard Answer]: {ground_truth_str}\n[Model_answer] : {predict_str}\nJudgement:"""
    return f'{demo_prompt}{test_prompt}'

# ==============================================================================
# 步骤 3: 核心功能函数
# ==============================================================================
def read_jsonl(file_path):
    """读取JSONL文件并解析其内容"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}", file=sys.stderr)
        return []
    
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
    """向 LLM 服务发送请求并解析判断分数。"""
    # 将 None 转换为字符串 "None" 以便处理
    pred_str = str(pred)
    gold_str = str(gold)

    full_prompt = get_full_prompt(pred_str, gold_str, question)
    
    try:
        with no_proxy():
            chat_response = client.chat.completions.create(
                model=model_name,
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

def process_item(item, client, model_name):
    """
    处理单个数据项：提取信息、获取LLM评分并构建最终输出字典。
    """
    question = item.get('question')
    model_response = item.get('model_response')
    
    # 安全地提取 ground_truth 信息
    ground_truth_list = item.get('ground_truth', [])
    if ground_truth_list and isinstance(ground_truth_list, list) and len(ground_truth_list) > 0:
        full_answer = ground_truth_list[0].get('full_answer')
        ui_elements = ground_truth_list[0].get('ui_elements', [])
    else:
        full_answer = None
        ui_elements = []
        print(f"警告: 记录缺少有效的 'ground_truth' 列表，计为 0 分: {item.get('id')}", file=sys.stderr)

    if full_answer is None:
        score = 0
    else:
        score = get_llm_score(client, model_name, model_response, full_answer, question)
    
    # 构建要输出的新记录
    processed_item = {
        'id': item.get('id'),
        'question': question,
        'full_answer': full_answer,
        'model_response': model_response,
        'screen_id': item.get('screen_id'),
        'split': item.get('split'),
        'ui_elements': ui_elements,
        'llm_score': score
    }
    
    return score, processed_item

def process_file_parallel(jsonl_path, client, model_name, num_threads=64):
    """
    并行处理单个 jsonl 文件：读取、评分、计算平均分并保存到新文件。
    """
    print(f"--- 开始并行处理文件 (使用 {num_threads} 线程): {jsonl_path} ---")
    
    data = read_jsonl(jsonl_path)
    if not data:
        print(f"错误: 未能从 {jsonl_path} 读取到任何数据。", file=sys.stderr)
        return

    # 初始化线程安全的数据结构
    lock = threading.Lock()
    updated_data = []
    all_scores = []

    # 准备任务参数
    tasks = [(item, client, model_name) for item in data]
    
    # 创建进度条
    progress_bar = tqdm(total=len(tasks), desc=f"并行评分 {Path(jsonl_path).name}")
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_item = {executor.submit(process_item, *task): task for task in tasks}
        
        for future in as_completed(future_to_item):
            try:
                score, processed_item = future.result()
                
                with lock:
                    all_scores.append(score)
                    updated_data.append(processed_item)
                    
            except Exception as e:
                print(f"错误: 处理项目时出错: {e}", file=sys.stderr)
            finally:
                # 更新进度条
                progress_bar.update(1)
    
    progress_bar.close()
    
    # 计算平均分
    average_score = _calculate_average(all_scores)
    avg_score_summary = {"average_llm_score": average_score}
    print(f"文件处理完成。平均 LLM 分数: {average_score:.4f}")

    # 构建新文件名并写入
    p = Path(jsonl_path)
    output_path = p.parent / f"{p.stem}_llm.jsonl"

    try:
        # 为了保证输出顺序与输入一致（如果需要），可以对结果进行排序
        # 此处我们简单地写入处理完成的数据
        with open(output_path, 'w', encoding='utf-8') as f:
            # 第一行写入包含所有平均分的摘要
            f.write(json.dumps(avg_score_summary) + '\n')
            
            # 写入更新后的数据
            for item in updated_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"结果已保存至: {output_path}\n")
    except IOError as e:
        print(f"错误: 写入文件失败 {output_path}: {e}", file=sys.stderr)


# ==============================================================================
# 步骤 4: 主程序入口
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="使用 LLM 评估 JSONL 文件中的预测结果。")
    parser.add_argument('--api_url', type=str, default="http://SH-IDC1-10-140-37-21:16113/v1", help="LLM 服务的 API base URL。")
    parser.add_argument('--api_key', type=str, default="not-needed", help="API key")
    # /mnt/petrelfs/share_data/ai4good_shared/models/Qwen/Qwen2.5-72B-Instruct
    # /mnt/petrelfs/share_data/ai4good_shared/models/Qwen/Qwen2.5-VL-7B-Instruct-new
    parser.add_argument('--model_name', type=str, default="/mnt/petrelfs/share_data/ai4good_shared/models/Qwen/Qwen2.5-VL-7B-Instruct-new", help="要使用的模型名称或路径。")
    parser.add_argument("--num_threads", type=int, default=64, help="并行处理时使用的线程数。")
    
    args = parser.parse_args()

    jsonl_paths = ["/mnt/petrelfs/sunhaoyu/visual-code/Tool-Data-Curation/web/test_results/7b_full_results.jsonl"]

    # 如果只有一个路径且它是目录，则获取该目录下的所有 .jsonl 文件
    resolved_paths = []
    for path_str in jsonl_paths:
        path = Path(path_str)
        if path.is_dir():
            resolved_paths.extend(path.glob('*.jsonl'))
        elif path.is_file() and path.name.endswith('.jsonl'):
            resolved_paths.append(path)
    
    if not resolved_paths:
        print(f"错误: 在指定的路径中未找到任何 .jsonl 文件: {jsonl_paths}", file=sys.stderr)
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
    print(f"使用 {args.num_threads} 个线程并行处理 JSONL 文件...")
    for path in resolved_paths:
        process_file_parallel(str(path), client, args.model_name, num_threads=args.num_threads)

if __name__ == '__main__':
    main()