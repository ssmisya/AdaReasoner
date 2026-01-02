import openai
import json
from datasets import load_dataset
from PIL import Image
import base64
from io import BytesIO
from tqdm import tqdm
import os
import concurrent.futures
import threading

# --- 1. 配置 ---
# 从您的启动脚本中获取的服务器地址和端口
api_base_url = "http://SH-IDC1-10-140-37-21:16113/v1"
# 对于自托管的vLLM，API Key不是必需的，但openai库要求提供一个值
api_key = "not-needed"
# 从您的启动脚本中获取的模型路径
model_name = "/mnt/petrelfs/share_data/ai4good_shared/models/Qwen/Qwen2.5-VL-7B-Instruct-new"

# --- 可调参数 ---
# 设置并发请求的数量，可以根据你的服务器承载能力调整
MAX_CONCURRENT_REQUESTS = 16
# 设置每多少条结果保存一次文件
SAVE_BATCH_SIZE = 100

# --- 2. 初始化客户端和线程锁 ---
# 为每个线程创建独立的 OpenAI 客户端，避免潜在的线程安全问题
thread_local = threading.local()
def get_openai_client():
    if not hasattr(thread_local, "client"):
        thread_local.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base_url,
        )
    return thread_local.client

# 文件写入锁，防止多个线程同时写入文件导致内容错乱
file_lock = threading.Lock()

# --- 3. 辅助函数 ---
def pil_to_base64(image: Image.Image) -> str:
    """将 PIL.Image 对象转换为 Base64 编码的字符串"""
    buffered = BytesIO()
    # 确保图像是 RGB 格式，避免 RGBA 带来的问题
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def save_results_batch(results, output_file):
    """将一批结果追加写入到 jsonl 文件"""
    if not results:
        return
    
    # 增加更明确的打印信息
    print(f"\n--- [File IO] Locking file to save a batch of {len(results)} results... ---")
    with file_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            # 写入前按ID排序，确保顺序
            results.sort(key=lambda x: x['id'])
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush() # 强制将缓冲区内容写入磁盘，便于实时查看
    print(f"--- [File IO] Batch saved successfully to {output_file}. ---")


# --- 4. 单个条目的处理函数 (并发执行) ---
def process_single_item(item_data):
    """
    处理单个数据项：准备数据、发送请求、返回结果。
    此函数将在并发线程中执行。
    """
    item_id, split_name, screen_id, ground_truth, question, base64_image = item_data
    
    try:
        image_url = f"data:image/jpeg;base64,{base64_image}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]

        client = get_openai_client()
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=4096,
            temperature=0,
        )
        model_response = chat_completion.choices[0].message.content

    except Exception as e:
        # 使用tqdm.write可以在不打乱进度条的情况下打印错误信息
        tqdm.write(f"[Error] Processing ID {item_id} (screen_id: {screen_id}) failed: {e}")
        model_response = f"API_ERROR: {str(e)}"

    return {
        "id": item_id,
        "question": question,
        "ground_truth": ground_truth,
        "model_response": model_response,
        "screen_id": screen_id,
        "split": split_name
    }

# --- 5. 主处理逻辑 ---
def process_dataset_concurrently():
    """
    使用并发方式加载和处理 RICO-ScreenQA 数据集。
    """
    dataset_path = "/mnt/petrelfs/sunhaoyu/visual-code/Tool-Data-Curation/web/datasets/RICO-ScreenQA"
    output_file = "test_results/7b_full_results.jsonl"
    
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. It will be cleared and rewritten.")
        # 创建一个空文件，而不是删除，这样更安全
        open(output_file, 'w').close()

    global_id_counter = 1
    for split_name in ["train", "validation", "test"]:
        print(f"\n--- Loading dataset split: {split_name} ---")
        try:
            dataset = load_dataset(dataset_path, split=split_name)
            print(f"Loaded {split_name} split with {len(dataset)} items.")
        except Exception as e:
            print(f"Failed to load dataset split '{split_name}': {e}")
            continue

        if not dataset:
            print(f"Split '{split_name}' is empty. Skipping.")
            continue

        # =================================================================
        # **核心修改点：解耦数据准备和并发执行**
        # =================================================================

        # **步骤 1: 在主线程中准备好所有任务数据**
        # 将所有数据（包括Base64转换）预处理好，放入一个简单的列表中。
        # 这可以避免在并发执行时与 datasets 库的迭代器发生冲突。
        print(f"Preparing all task data for '{split_name}' in main thread...")
        tasks_to_process = []
        for i, item in enumerate(tqdm(dataset, desc=f"Preparing {split_name} data")):
            base64_image = pil_to_base64(item['image'])
            task_data = (
                global_id_counter + i,
                split_name,
                item['screen_id'],
                item['ground_truth'],
                item['question'],
                base64_image
            )
            tasks_to_process.append(task_data)
        
        print(f"Finished preparing {len(tasks_to_process)} tasks for '{split_name}'.")

        results_buffer = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            # **步骤 2: 一次性提交所有准备好的任务**
            # 这样可以确保所有任务都已在队列中，与数据源完全分离。
            print(f"Submitting {len(tasks_to_process)} tasks to the thread pool...")
            futures = [executor.submit(process_single_item, data) for data in tasks_to_process]
            
            # **步骤 3: 从 futures 列表中安全地收集结果**
            # 这个循环现在只处理已经提交的任务，不再与数据集迭代器交互。
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks_to_process), desc=f"Processing {split_name} items"):
                try:
                    result = future.result()
                    if result:
                        results_buffer.append(result)
                    
                    # 当缓冲区满时，保存并清空
                    if len(results_buffer) >= SAVE_BATCH_SIZE:
                        save_results_batch(results_buffer, output_file)
                        results_buffer.clear()
                except Exception as exc:
                    tqdm.write(f'A task generated an exception during result retrieval: {exc}')

        # 处理完一个 split 后，保存缓冲区中剩余的结果
        if results_buffer:
            print(f"\nSaving remaining {len(results_buffer)} results for split '{split_name}'...")
            save_results_batch(results_buffer, output_file)
            results_buffer.clear()

        global_id_counter += len(tasks_to_process)

    print(f"\n--- All processing complete! Results have been saved to {output_file} ---")

if __name__ == "__main__":
    process_dataset_concurrently()
