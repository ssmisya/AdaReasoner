import openai
import json
from PIL import Image
import base64
from io import BytesIO
from tqdm import tqdm
import os
import concurrent.futures
import threading
import time

# --- 1. 配置 ---
api_base_url = "http://SH-IDC1-10-140-37-71:16113/v1"
api_key = "not-needed"
model_name = "/mnt/petrelfs/share_data/ai4good_shared/models/Qwen/Qwen2.5-VL-72B-Instruct" # 例如使用这个VL模型

# --- 文件路径配置 ---
input_jsonl_file = 'AdaReasoner/AdaDataCurationweb/datasets/GUIChat-processed/singleturn_data.jsonl'
image_path_map_file = 'AdaReasoner/AdaDataCurationweb/datasets/GUIChat-processed/images/image_id2path.json'
output_file = "AdaReasoner/AdaDataCurationweb/test_results/guichat/72b_responses.jsonl"

# --- 可调参数 ---
MAX_CONCURRENT_REQUESTS = 32
SAVE_BATCH_SIZE = 100

# --- 2. 初始化客户端和线程锁 ---
thread_local = threading.local()
def get_openai_client():
    if not hasattr(thread_local, "client"):
        thread_local.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base_url,
        )
    return thread_local.client

file_lock = threading.Lock()

# --- 3. 辅助函数 (无变化) ---
def pil_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def save_results_batch(results, output_file):
    if not results:
        return
    
    with file_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            # 在写入前排序是可选的，因为是批量追加
            # results.sort(key=lambda x: x['id']) 
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()

# --- 4. 单个条目的处理函数 (无变化) ---
def process_single_item(item_data):
    item_id, image_id, task_type, question, ground_truth, base64_image = item_data
    
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
        tqdm.write(f"[Error] Processing ID {item_id} (image_id: {image_id}) failed: {e}")
        model_response = f"API_ERROR: {str(e)}"

    return {
        "id": item_id,
        "image_id": image_id,
        "task_type": task_type,
        "question": question,
        "ground_truth": ground_truth,
        "model_response": model_response,
    }

# --- 5. 主处理逻辑 (已重构) ---
def process_dataset_concurrently():
    """
    使用流式处理和并发方式处理JSONL数据集，以保持低内存占用。
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. It will be cleared.")
        open(output_file, 'w').close()

    # **步骤 1: 加载 image_id 到路径的映射**
    print(f"Loading image path map from {image_path_map_file}...")
    try:
        with open(image_path_map_file, 'r', encoding='utf-8') as f:
            image_path_map = json.load(f)
        print(f"Loaded {len(image_path_map)} image path mappings.")
    except Exception as e:
        print(f"Fatal Error: Failed to load image path map file: {e}")
        return

    # **步骤 2: (新) 计算总行数，用于tqdm进度条**
    try:
        with open(input_jsonl_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        print(f"Found {total_lines} lines to process in {input_jsonl_file}.")
    except Exception as e:
        print(f"Fatal Error: Could not count lines in input file: {e}")
        return

    # **步骤 3: (重构) 流式处理和并发执行**
    results_buffer = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor, \
         open(input_jsonl_file, 'r', encoding='utf-8') as f_in, \
         tqdm(total=total_lines, desc="Processing items") as pbar:

        futures = {}
        for line in f_in:
            # 保持队列大小，防止一次性提交过多任务
            while len(futures) >= MAX_CONCURRENT_REQUESTS * 2:
                # 等待并处理已完成的任务
                done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    try:
                        result = future.result()
                        if result:
                            results_buffer.append(result)
                    except Exception as exc:
                        tqdm.write(f'A task generated an exception during result retrieval: {exc}')
                    
                    # 从futures字典中移除已处理的任务
                    del futures[future]
                    pbar.update(1) # 更新进度条

            # 提交新任务
            if not line.strip():
                pbar.update(1) # 如果是空行，也更新进度条
                continue
            
            item = json.loads(line)
            image_id = item.get('image_id')
            image_path = image_path_map.get(image_id)

            if not image_path:
                tqdm.write(f"[Warning] Skipping ID {item.get('id')}: image_id '{image_id}' not found.")
                pbar.update(1)
                continue
            
            try:
                with Image.open(image_path) as img:
                    base64_image = pil_to_base64(img)
            except Exception as e:
                tqdm.write(f"[Warning] Skipping ID {item.get('id')}: Failed to process image {image_path}: {e}")
                pbar.update(1)
                continue

            task_data = (item.get('id'), image_id, item.get('task_type'), item.get('Question'), item.get('Answer'), base64_image)
            future = executor.submit(process_single_item, task_data)
            futures[future] = task_data

            # 检查并保存缓冲区
            if len(results_buffer) >= SAVE_BATCH_SIZE:
                save_results_batch(results_buffer, output_file)
                tqdm.write(f"--- [File IO] Batch of {len(results_buffer)} saved. ---")
                results_buffer.clear()

        # **步骤 4: 处理剩余的任务**
        tqdm.write("\nAll tasks submitted. Waiting for remaining tasks to complete...")
        for future in concurrent.futures.as_completed(futures.keys()):
            try:
                result = future.result()
                if result:
                    results_buffer.append(result)
            except Exception as exc:
                tqdm.write(f'A task generated an exception during final result retrieval: {exc}')
            pbar.update(1)

    # **步骤 5: 保存最后批次的结果**
    if results_buffer:
        tqdm.write(f"\nSaving final batch of {len(results_buffer)} results...")
        save_results_batch(results_buffer, output_file)
        results_buffer.clear()

    print(f"\n--- All processing complete! Results have been saved to {output_file} ---")


if __name__ == "__main__":
    process_dataset_concurrently()
