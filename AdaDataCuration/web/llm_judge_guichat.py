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
# 步骤 1: 定义一个用于临时禁用代理的上下文管理器 (无变化)
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
# 步骤 2: Prompt 构建函数 (无变化)
# ==============================================================================
def get_chat_template():
# 这个要修改，因为是chat类型的
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
    """读取JSONL文件并解析其内容 (无变化)"""
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
    """向 LLM 服务发送请求并解析判断分数 (无变化)"""
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
    """辅助函数，用于安全地计算平均分 (无变化)"""
    if not scores_list:
        return 0.0
    return sum(scores_list) / len(scores_list)

# --- !!! 核心修改点 !!! ---
def process_item(item, client, model_name):
    """
    处理单个数据项：提取信息、获取LLM评分并构建最终输出字典。
    (已根据您的数据格式进行修改)
    """
    question = item.get('question')
    model_response = item.get('model_response')
    # 直接从 item 中获取 'ground_truth' 字符串
    ground_truth = item.get('ground_truth')

    # 检查关键数据是否存在
    if ground_truth is None or model_response is None or question is None:
        score = 0
        print(f"警告: 记录 ID {item.get('id')} 缺少 'question', 'ground_truth', 或 'model_response'，计为 0 分。", file=sys.stderr)
    else:
        # 如果模型返回API错误，也直接计为0分，避免不必要的API调用
        if isinstance(model_response, str) and model_response.startswith("API_ERROR"):
            score = 0
        else:
            score = get_llm_score(client, model_name, model_response, ground_truth, question)
    
    # 构建要输出的新记录，使用与输入文件一致的字段
    processed_item = {
        'id': item.get('id'),
        'image_id': item.get('image_id'),         # 使用 'image_id'
        'task_type': item.get('task_type'),       # 保留 'task_type'
        'question': question,
        'ground_truth': ground_truth,
        'model_response': model_response,
        'llm_score': score                        # 添加LLM评分
    }
    
    return score, processed_item

def process_file_parallel(jsonl_path, client, model_name, num_threads=64):
    """
    并行处理单个 jsonl 文件：读取、评分、计算平均分并保存到新文件。(无变化)
    """
    print(f"--- 开始并行处理文件 (使用 {num_threads} 线程): {jsonl_path} ---")
    
    data = read_jsonl(jsonl_path)
    if not data:
        print(f"错误: 未能从 {jsonl_path} 读取到任何数据。", file=sys.stderr)
        return

    lock = threading.Lock()
    updated_data = []
    all_scores = []

    tasks = [(item, client, model_name) for item in data]
    
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
                progress_bar.update(1)
    
    progress_bar.close()
    
    average_score = _calculate_average(all_scores)
    avg_score_summary = {"average_llm_score": average_score}
    print(f"文件处理完成。平均 LLM 分数: {average_score:.4f}")

    p = Path(jsonl_path)
    output_path = p.parent / f"{p.stem}_llm.jsonl"

    try:
        # 为了保证输出顺序与输入一致，可以根据ID排序
        updated_data.sort(key=lambda x: x.get('id', 0))
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(avg_score_summary) + '\n')
            
            for item in updated_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"结果已保存至: {output_path}\n")
    except IOError as e:
        print(f"错误: 写入文件失败 {output_path}: {e}", file=sys.stderr)


# ==============================================================================
# 步骤 4: 主程序入口 (无变化)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="使用 LLM 评估 JSONL 文件中的预测结果。")
    parser.add_argument('--api_url', type=str, default="http://SH-IDC1-10-140-37-138:16113/v1", help="LLM 服务的 API base URL。")
    parser.add_argument('--api_key', type=str, default="not-needed", help="API key")
    parser.add_argument('--model_name', type=str, default="/mnt/petrelfs/share_data/ai4good_shared/models/Qwen/Qwen2.5-VL-72B-Instruct", help="要使用的模型名称或路径。")
    parser.add_argument("--num_threads", type=int, default=64, help="并行处理时使用的线程数。")
    
    args = parser.parse_args()

    # 指定要处理的文件路径
    jsonl_paths = ["/mnt/petrelfs/sunhaoyu/visual-code/Tool-Data-Curation/web/test_results/guichat/7b_responses.jsonl"]

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

    try:
        client = openai.OpenAI(
            api_key=args.api_key,
            base_url=args.api_url
        )
    except Exception as e:
        print(f"错误: 初始化 OpenAI 客户端失败: {e}", file=sys.stderr)
        return

    print(f"使用 {args.num_threads} 个线程并行处理 JSONL 文件...")
    for path in resolved_paths:
        process_file_parallel(str(path), client, args.model_name, num_threads=args.num_threads)

if __name__ == '__main__':
    main()
