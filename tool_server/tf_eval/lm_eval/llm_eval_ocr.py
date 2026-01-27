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
import ast

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

def get_chat_template_spotting():
    """返回用于 LLM 判断的基础指令模板。"""
    return """
Your task is to determine if the [Model OCR Result] correctly and completely extracts the text from an image, based on the [Ground Truth] text.
The [Model OCR Result] is considered correct only if it contains **all** of the text present in the [Ground Truth]. The order of the text does not need to be identical, but every word and number from the ground truth must be present in the model's result. Extraneous text recognized by the model is acceptable, as long as all the ground truth text is included.
- If the [Model OCR Result] includes all text from the [Ground Truth], output `Judgement: 1`.
- If the [Model OCR Result] is missing **any** part of the text from the [Ground Truth], output `Judgement: 0`.
Only output `Judgement: 1` or `Judgement: 0` and don't output anything else.
"""

def get_chat_template_understanding():
    """返回用于 LLM 判断的基础指令模板。"""
    return """
Below are two answers to a question. [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question. Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judgement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""

def get_prompt_examples_spotting():
    """返回用于 Few-shot 学习的示例。"""
    example_1 = """[Ground Truth]: ['8110011A', '4Y27ET', 'JCS'] \n[Model OCR Result] The image contains the following words: '8110011A 4Y27ET JCS'. The OCR tool detected these words with high confidence. \nJudgement: 1"""
    example_2 = """[Ground Truth]: ['VOLVO', '30714480', '90AMA', 'NORWAY'] \n[Model OCR Result]: The image contains the following words: 'VOLVO', '30714480', '90AMA', 'NORWAY'. The OCR tool detected these words with their respective confidence scores and bounding boxes. The words 'VOLVO' and '30714480' have higher confidence scores, indicating a higher accuracy in detection. The other words, '90AMA' and 'NORWAY', have lower confidence scores, but they are still detected by the OCR tool. \nJudgement: 1"""
    example_3 = """[Ground Truth]: ['REMUS', 'MADE', 'IN', 'AUSTRIA', 'FA', '0904'] \n[Model OCR Result]: The image contains the following words: REMUS, MADE IN AUSTRIA, FA, 0904. \nJudgement: 1"""
    example_4 = """[Ground Truth]: ['REMUS', 'MADE', 'IN', 'AUSTRIA', 'FA', '0904'] \n[Model OCR Result]: The image contains the following words: REMUS, MADE IN, FA, 0904. \nJudgement: 0"""
    return [example_1, example_2, example_3, example_4]

def get_prompt_examples_understanding():
    """返回用于 Few-shot 学习的示例。"""
    example_1 = """[Question]: "What is the price of '酸汤肥牛'?A.28 B.38 C.32 D.36\n[Standard Answer]: ['A']\n[Model_answer] : The image contains a menu with various dish names and prices. The text '酸汤肥牛' is detected in the OCR results. The corresponding price next to '酸汤肥牛' is '28 元'. The image also displays the price options for '酸汤肥牛' as '28 元'. Therefore, the price of '酸汤肥牛' is 28. The options provided in the question are A.28, B.38, C.32, and D.36. The correct answer is A.28.\nJudgement: 1"""
    example_2 = """[Question]: Is the countertop tan or blue?\n[Standard Answer]: The countertop is tan.\n[Model_answer] : tan\nJudgement: 1"""
    example_3 = "[Question]: What color is the towel in the center of the picture?\n[Standard Answer]: A. The towel in the center of the picture is blue.\n[Model_answer] : Blue\nJudgement: 1"
    example_4 = """[Question]: What can you see on the screen of the machine in the image next to the company logo?A.PRODUKT B.WAHLEN C.ESC D.Saeco\n[Standard Answer]: ['D']\n[Model_answer] : B "WAHLEN".\nJudgement: 0"""
    
    return [example_1, example_2, example_3, example_4]

def get_full_prompt(predict_str, ground_truth_str, question):
    """构建最终发送给 LLM 的完整 Prompt。"""
    if "Please list all the words in this image" in question:
        # 为Spotting类型的问题
        chat_template = get_chat_template_spotting()
        examples = get_prompt_examples_spotting()
        demo_prompt = chat_template + "\n\n".join(examples) + "\n\n"
        # 对ground_truth_str进行筛选##############
        ground_truth = ast.literal_eval(ground_truth_str)
        ground_truth = [item for item in ground_truth if '#' not in item]
        ground_truth = [item for item in ground_truth if item != ' '] # 要把空格给出去掉
        ground_truth_str = str(ground_truth)
        test_prompt = f"""[Ground Truth]: {ground_truth_str}\n[Model OCR Result] : {predict_str}\nJudgement:"""
        return f'{demo_prompt}{test_prompt}'
    else: 
        # 为Understanding类型的问题
        chat_template = get_chat_template_understanding()
        examples = get_prompt_examples_understanding()
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

    is_webmmu = 'webmmu' in jsonl_path
    is_texthalubench = 'texthalubench' in jsonl_path

    # 初始化分数追踪器
    updated_data = []
    all_scores = []
    functional_scores = [] if is_webmmu else None
    general_image_understanding_scores = [] if is_webmmu else None
    complex_reasoning_scores = [] if is_webmmu else None
    spotting_scores = [] if is_texthalubench else None
    understanding_scores = [] if is_texthalubench else None
    
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
        
        if is_webmmu:
            item_type = item.get('category')
            if item_type == 'Functional':
                functional_scores.append(score)
            elif item_type == 'General Image Understanding':
                general_image_understanding_scores.append(score)
            elif item_type == 'Complex Reasoning':
                complex_reasoning_scores.append(score)
        if is_texthalubench:
            item_type = item.get('category')
            if item_type == 'Spotting':
                spotting_scores.append(score)
            elif item_type == 'Understanding':
                understanding_scores.append(score)
            else:
                raise ValueError("category不在设置范围内")
        updated_data.append(item)

    # 计算平均分
    avg_score_summary = {}
    average_score = _calculate_average(all_scores)
    avg_score_summary["average_llm_score"] = average_score


    if is_webmmu:
        avg_functional = _calculate_average(functional_scores)
        avg_general_image_understanding = _calculate_average(general_image_understanding_scores)
        avg_complex_reasoning = _calculate_average(complex_reasoning_scores)
        avg_score_summary["average_functional_score"] = avg_functional
        avg_score_summary["average_general_image_understanding_score"] = avg_general_image_understanding
        avg_score_summary["average_complex_reasoning_score"] = avg_complex_reasoning
        print(f"文件处理完成。总平均分: {average_score:.4f}, Functional 平均分: {avg_functional:.4f}, General Image Understanding 平均分: {avg_general_image_understanding:.4f}, Complex Reasoning 平均分: {avg_complex_reasoning:.4f}")
    elif is_texthalubench:
        avg_spotting = _calculate_average(spotting_scores)
        avg_understanding = _calculate_average(understanding_scores)
        avg_score_summary["avg_spotting_score"] = avg_spotting
        avg_score_summary["avg_understanding"] = avg_understanding
        print(f"文件处理完成。总平均分: {average_score:.4f}, Spotting 平均分: {avg_spotting:.4f}, Understanding 平均分: {avg_understanding:.4f}")
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
    如果路径包含 'charxiv'，则额外计算并保存分类平均分。
    如果路径包含 'vstar'，则额外计算并保存分类平均分。
    
    参数:
        jsonl_path: 输入的JSONL文件路径
        client: OpenAI API客户端
        model_name: 要使用的模型名称
        num_threads: 使用的线程数量，默认为64
        question_key: 问题字段的键名，默认为'question'
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
        is_texthalubench = task_name == "texthalubench"
        # 初始化线程安全的数据结构
        lock = threading.Lock()
        updated_data = []
        all_scores = []
        functional_scores = [] if is_webmmu else None
        general_image_understanding_scores = [] if is_webmmu else None
        complex_reasoning_scores = [] if is_webmmu else None
        spotting_scores = [] if is_texthalubench else None
        understanding_scores = [] if is_texthalubench else None
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
                        
                        if is_webmmu:
                            item_type = updated_item.get('category')
                            if item_type == 'Functional':
                                functional_scores.append(score)
                            elif item_type == 'General Image Understanding':
                                general_image_understanding_scores.append(score)
                            elif item_type == 'Complex Reasoning':
                                complex_reasoning_scores.append(score)
                        if is_texthalubench:
                            item_type = updated_item.get('category')
                            if item_type == "Spotting":
                                spotting_scores.append(score)
                            elif item_type == "Understanding":
                                understanding_scores.append(score)
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

        if is_webmmu:
            avg_functional = _calculate_average(functional_scores)
            avg_general_image_understanding = _calculate_average(general_image_understanding_scores)
            avg_complex_reasoning = _calculate_average(complex_reasoning_scores)
            avg_score_summary["average_functional_score"] = avg_functional
            avg_score_summary["average_general_image_understanding_score"] = avg_general_image_understanding
            avg_score_summary["average_complex_reasoning_score"] = avg_complex_reasoning
            print(f"文件处理完成。总平均分: {average_score:.4f}, Functional 平均分: {avg_functional:.4f}, General Image Understanding 平均分: {avg_general_image_understanding:.4f}, Complex Reasoning 平均分: {avg_complex_reasoning:.4f}")
        elif is_texthalubench:
            avg_spotting = _calculate_average(spotting_scores)
            avg_understanding = _calculate_average(understanding_scores)
            avg_score_summary["avg_spotting_score"] = avg_spotting
            avg_score_summary["avg_understanding"] = avg_understanding
            print(f"文件处理完成。总平均分: {average_score:.4f}, Spotting 平均分: {avg_spotting:.4f}, Understanding 平均分: {avg_understanding:.4f}")
        else:
            print(f"文件处理完成。平均 LLM 分数: {average_score:.4f}")
        
        data_item["llm_eval_summary"] = avg_score_summary

        # 构建新文件名并写入
        p = Path(jsonl_path)
        output_path = p.parent / f"{p.stem}_llm.jsonl"

        append_jsonl(data_item,output_path)


# ==============================================================================
# 步骤 4: 主程序入口
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="使用 LLM 评估 JSONL 文件中的预测结果。")
    # 让 jsonl_paths 成为可选参数，如果未提供，则使用硬编码的列表
    parser.add_argument('--jsonl_paths', nargs='*', help="一个或多个待处理的 .jsonl 文件的路径。如果未提供，将使用代码中定义的默认路径。")
    parser.add_argument('--api_url', type=str, default="http://SH-IDC1-10-140-37-71:16113/v1", help="vLLM 服务的 API base URL。")
    # parser.add_argument('--api_url', type=str, default="http://SH-IDC1-10-140-37-82:16113/v1", help="vLLM 服务的 API base URL。")
    parser.add_argument('--api_key', type=str, default="not-needed", help="API key")
    parser.add_argument('--model_name', type=str, default="/mnt/petrelfs/share_data/ai4good_shared/models/Qwen/Qwen2.5-72B-Instruct", help="要使用的模型名称或路径。")
    parser.add_argument("--num_threads", type=int, default=64, help="并行处理时使用的线程数。")
    
    args = parser.parse_args()

    # 如果命令行没有提供路径，则使用您预设的列表
    if args.jsonl_paths:
        jsonl_paths = args.jsonl_paths
    else:
        print("未从命令行接收到路径，使用代码中预设的路径列表。")
        # 运行记得关代理
        jsonl_paths = [
            # "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/3b_output.jsonl",
            # "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/3b_tool_rl_300_output.jsonl",
            # "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/3b_tool_rl_350_output.jsonl",
            # "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/3b_tool_rl_400_output.jsonl",
            # "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/3b_tool_sft_output.jsonl",
            # "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/3b_wo_tool_direct_rl_output.jsonl",
            # "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/3b_wo_tool_direct_sft_output.jsonl",
            # "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/7b_350_rl_output.jsonl",
            # "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/7b_output.jsonl",
            # "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/7b_tool_sft_output.jsonl",
            # "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/7b_wo_tool_direct_rl_output.jsonl",
            # "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/7b_wo_tool_direct_sft_output.jsonl",
            # "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/32b_output.jsonl",
            # "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/72b_output.jsonl",
            # "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/claude4_output.jsonl",
            # "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/gemini2_5flash_output.jsonl",
            # "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/gpt5_output.jsonl",
            # "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/v1_250_rl_output.jsonl",
            # "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/vl3_output.jsonl"

            # 9.16早
            # "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/3b_tool_rl_100_output.jsonl",
            "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/3b_tool_rl_150_output.jsonl",
            "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/3b_tool_rl_250_output.jsonl",
            "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/3b_tool_rl_250_output.jsonl",
            "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/3b_tool_sft_rl_100_output.jsonl",
            "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/3b_tool_sft_rl_150_output.jsonl",
            "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/3b_tool_sft_rl_200_output.jsonl",
            "/tool_server/tf_eval/scripts/logs/ckpt/texthalubench1/3b_tool_sft_rl_250_output.jsonl",

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
        print(f"使用 {args.num_threads} 个线程并行处理 JSONL 文件...")
        for path in jsonl_paths:
            process_and_save_jsonl_parallel(path, client, args.model_name, num_threads=args.num_threads)
    else:
        for path in jsonl_paths:
            process_and_save_jsonl(path, client, args.model_name)

if __name__ == '__main__':
    main()