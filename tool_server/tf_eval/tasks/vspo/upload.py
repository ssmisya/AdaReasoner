# upload.py
import os
import json
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage
from huggingface_hub import HfApi, login
from PIL import Image
import argparse
from tqdm import tqdm
import yaml
from collections import defaultdict

# 路径验证任务说明
PATH_VERIFY_TASK_INSTRUCTION = """
You are a maze solver. Your goal is to guide a player from the start to the goal on a grid map while avoiding holes, using the fewest moves. The player can move one square at a time in the directions left (L), right (R), up (U), or down (D). Moving off the edge has no effect, and falling into a hole results in failure. Reaching the goal means success. 

Now please determine if the action sequence is safe for the given maze. Your final answer should be formatted as \\boxed{Yes} or \\boxed{No}.

The action sequence is:

<ACTION-SEQ>
"""

# 路径导航任务说明
PATH_NAVIGATION_INSTRUCTION = """
You are a maze solver. Your goal is to guide a player from the start to the goal on a grid map while avoiding holes, using the fewest moves. The player can move one square at a time in the directions left (L), right (R), up (U), or down (D). Moving off the edge has no effect, and falling into a hole results in failure. Reaching the goal means success. Your final answer should be formatted as \\boxed{L,R,U,D}.

Please generate action plan for the input maze image.
"""

def load_config(config_path="config.yaml"):
    """加载配置文件"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        print(f"Warning: Config file {config_path} not found, using default settings")
        return {}

def read_jsonl(file_path):
    """读取JSONL文件"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        print(f"Successfully loaded {len(data)} records from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []

def convert_to_gym_map(item):
    """
    将数据项转换为gym环境可用的地图格式
    
    Args:
        item: 数据项，包含地图信息
        
    Returns:
        list: gym环境可用的地图格式
    """
    # 首先从text_map中提取文本地图
    if "text_map" in item and "output" in item["text_map"] and "text_map" in item["text_map"]["output"]:
        text_map_str = item["text_map"]["output"]["text_map"]
        # 解析文本地图
        rows = []
        for line in text_map_str.split('\n'):
            if '|' in line and ('Row' in line or 'Col' in line):
                continue  # 跳过表头行
            
            row_cells = []
            cells = line.split('|')
            for cell in cells[1:]:  # 跳过第一个空元素
                if not cell.strip():
                    continue
                
                cell_value = cell.strip()
                if cell_value == '_':  # 空格表示安全区域
                    row_cells.append('F')
                elif cell_value == '#':  # '#' 表示冰洞
                    row_cells.append('H')
                elif cell_value == '@':  # '@' 表示起点
                    row_cells.append('S')
                elif cell_value == '*':  # '*' 表示终点
                    row_cells.append('G')
            
            if row_cells:  # 如果行不为空
                rows.append(row_cells)
        
        # 检查地图是否有效
        if rows and all(len(row) == len(rows[0]) for row in rows):
            return rows
    
    # 如果无法从text_map中提取，则从坐标信息中构建
    size = item["size"]
    cell_size = 64  # 假设每个单元格是64像素
    
    # 创建一个全是安全区域的地图
    map_data = [['F' for _ in range(size)] for _ in range(size)]
    
    # 设置起点
    start_x = int(item["start_coords"][0] / cell_size)
    start_y = int(item["start_coords"][1] / cell_size)
    map_data[start_y][start_x] = 'S'
    
    # 设置终点
    goal_x = int(item["goal_coords"][0] / cell_size)
    goal_y = int(item["goal_coords"][1] / cell_size)
    map_data[goal_y][goal_x] = 'G'
    
    # 设置障碍物
    for obs in item["obstacle_coords"]:
        obs_x = int(obs[0] / cell_size)
        obs_y = int(obs[1] / cell_size)
        # 确保坐标在有效范围内
        if 0 <= obs_y < size and 0 <= obs_x < size:
            map_data[obs_y][obs_x] = 'H'
    
    return map_data

def load_path_verify_data(data_path, task_type, split, img_dir):
    """加载路径验证任务数据"""
    meta_data = []
    
    # 读取JSONL文件
    data = read_jsonl(data_path)
    
    for item in data:
        # 检查item是否包含必要的字段
        if not all(key in item for key in ["id", "image_path", "path_drawings"]):
            raise ValueError(f"Missing required keys in item: {item.keys()}")
        
        # 获取随机路径数据
        random_path = item["path_drawings"]["random"]
        if "path" not in random_path or "is_safe" not in random_path:
            raise ValueError(f"Missing 'path' or 'is_safe' in random path: {random_path}")
        
        path_string = random_path["path"]
        is_safe = random_path["is_safe"]
        
        # 构建提示
        text_prompt = PATH_VERIFY_TASK_INSTRUCTION.replace("<ACTION-SEQ>", path_string)
        
        # 转换gym地图
        gym_map = convert_to_gym_map(item)
        
        image_path = os.path.join(img_dir, item["image_path"]) if img_dir else item["image_path"]
        meta_data.append({
            "idx": f"{item['id']}_verify_{split}",
            "original_id": item["id"],
            "image_path": image_path,
            "text": text_prompt,
            "answer": "Yes" if is_safe else "No",
            "task_type": task_type,
            "split": split,
            "size": item["size"],
            "path_length": len(path_string.split(",")),
            "path": path_string,
            "gym_map": gym_map
        })
    
    print(f"Loaded {len(meta_data)} records for {task_type} from {data_path}")
    return meta_data

def load_path_navigation_data(data_path, task_type, split, img_dir):
    """加载路径导航任务数据"""
    meta_data = []
    
    # 读取JSONL文件
    data = read_jsonl(data_path)
    
    for item in data:
        # 检查item是否包含必要的字段
        if not all(key in item for key in ["id", "image_path", "start_coords", "goal_coords", "obstacle_coords"]):
            raise ValueError(f"Missing required keys in item: {item.keys()}")
        
        # 构建提示
        text_prompt = PATH_NAVIGATION_INSTRUCTION
        
        # 获取A*路径（如果有）
        astar_path = item.get("astar_path", {}).get("path", "") if "astar_path" in item else ""
        
        # 转换gym地图
        gym_map = convert_to_gym_map(item)
        
        image_path = os.path.join(img_dir, item["image_path"]) if img_dir else item["image_path"]
        meta_data.append({
            "idx": f"{item['id']}_navigation_{split}",
            "original_id": item["id"],
            "image_path": image_path,
            "text": text_prompt,
            "answer": "DYNAMIC_EVAL",  # 这个任务需要动态评估
            "task_type": task_type,
            "split": split,
            "size": item["size"],
            "start_coords": item["start_coords"],
            "goal_coords": item["goal_coords"],
            "obstacle_coords": item["obstacle_coords"],
            "astar_path": astar_path,
            "gym_map": gym_map
        })
    
    print(f"Loaded {len(meta_data)} records for {task_type} from {data_path}")
    return meta_data

def load_data_function(config):
    """加载自定义数据集的函数"""
    dataset_path = config.get("dataset_path")
    tasks = config.get("tasks", ["navigation-test", "verify-test"])
    
    # 从配置文件获取数据路径
    data_dir = config.get("data_dir", "./metadata_split")
    img_dir = config.get("image_dir", "./images")
    
    # 如果data_dir是相对路径，则相对于dataset_path
    if not os.path.isabs(data_dir) and dataset_path:
        data_dir = os.path.join(dataset_path, data_dir)
    
    verify_dir = os.path.join(data_dir, "path_verify")
    navigation_dir = os.path.join(data_dir, "path_navigation")
    
    meta_data = []
    
    # 加载不同任务的数据
    for task in tasks:
        task_prefix, task_suffix = task.split("-")
        if task_prefix == "verify":  # 路径验证任务
            # 加载验证数据
            data_path = os.path.join(verify_dir, f"{task_suffix}.jsonl")
            if os.path.exists(data_path):
                meta_data.extend(load_path_verify_data(data_path, task_prefix, task_suffix, img_dir))
            else:
                print(f"Warning: Data file not found: {data_path}")
        
        elif task_prefix == "navigation":  # 路径导航任务
            data_path = os.path.join(navigation_dir, f"{task_suffix}.jsonl")
            if os.path.exists(data_path):
                meta_data.extend(load_path_navigation_data(data_path, task_prefix, task_suffix, img_dir))
            else:
                print(f"Warning: Data file not found: {data_path}")
    
    # 数据集统计信息
    print(f"Total data loaded: {len(meta_data)}")
    
    # 统计各任务的数据量
    task_counts = defaultdict(int)
    for item in meta_data:
        task_type = item.get("task_type", "unknown")
        task_counts[task_type] += 1
    
    for task_type, count in task_counts.items():
        print(f"Task type: {task_type}, count: {count}")
    
    return meta_data

def convert_to_hf_dataset(meta_data):
    """将元数据转换为HuggingFace Dataset格式"""
    
    # 按任务类型和split分组数据
    grouped_data = {}
    for item in meta_data:
        task_type = item['task_type']
        split = item['split']
        key = f"{task_type}_{split}"
        
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(item)
    
    # 为每个组创建dataset
    dataset_dict = {}
    
    # 定义统一的特征结构（包含所有可能的字段）
    features = Features({
        'idx': Value('string'),
        'original_id': Value('string'),
        'image': HFImage(),
        'text': Value('string'),
        'answer': Value('string'),
        'task_type': Value('string'),
        'split': Value('string'),
        'size': Value('int32'),
        'gym_map': Value('string'),
        # verify任务特定字段
        'path_length': Value('int32'),
        'path': Value('string'),
        # navigation任务特定字段
        'start_coords': Value('string'),
        'goal_coords': Value('string'),
        'obstacle_coords': Value('string'),
        'astar_path': Value('string'),
    })
    
    for key, items in grouped_data.items():
        print(f"Processing {key} with {len(items)} items...")
        
        # 准备数据
        processed_items = []
        for item in tqdm(items, desc=f"Loading images for {key}"):
            try:
                # 加载图像
                image_path = item['image_path']
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}, skipping...")
                    continue
                
                image = Image.open(image_path).convert('RGB')
                
                # 构建数据项（包含所有字段，不存在的用None或默认值）
                data_item = {
                    'idx': item['idx'],
                    'original_id': item['original_id'],
                    'image': image,
                    'text': item['text'],
                    'answer': item['answer'],
                    'task_type': item['task_type'],
                    'split': item['split'],
                    'size': item['size'],
                    'gym_map': json.dumps(item['gym_map']),
                    # verify任务字段（如果不存在则为None）
                    'path_length': item.get('path_length', None),
                    'path': item.get('path', None),
                    # navigation任务字段（如果不存在则为None）
                    'start_coords': json.dumps(item['start_coords']) if 'start_coords' in item else None,
                    'goal_coords': json.dumps(item['goal_coords']) if 'goal_coords' in item else None,
                    'obstacle_coords': json.dumps(item['obstacle_coords']) if 'obstacle_coords' in item else None,
                    'astar_path': item.get('astar_path', None),
                }
                
                processed_items.append(data_item)
                
            except Exception as e:
                print(f"Error processing item {item['idx']}: {e}")
                continue
        
        if processed_items:
            # 创建dataset（使用统一的features）
            dataset = Dataset.from_list(processed_items, features=features)
            dataset_dict[key] = dataset
            print(f"Created dataset for {key} with {len(dataset)} items")
    
    return DatasetDict(dataset_dict)

def upload_to_hub(dataset_dict, repo_id, token=None):
    """上传数据集到HuggingFace Hub"""
    
    if token:
        login(token=token)
    
    print(f"Uploading dataset to {repo_id}...")
    
    try:
        # 上传数据集
        dataset_dict.push_to_hub(
            repo_id,
            private=False,  # 设置为True如果要私有仓库
            commit_message="Upload Frozen Lake dataset"
        )
        print(f"Successfully uploaded dataset to {repo_id}")
        
        # 创建并上传README
        readme_content = generate_readme(dataset_dict)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add README"
        )
        print("Successfully uploaded README")
        
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        raise

def generate_readme(dataset_dict):
    """生成数据集README"""
    
    # 统计信息
    total_samples = sum(len(ds) for ds in dataset_dict.values())
    splits_info = "\n".join([f"- `{name}`: {len(ds)} samples" for name, ds in dataset_dict.items()])
    
    readme = f"""# Frozen Lake Visual Spatial Planning Dataset

        ## Dataset Description

        This dataset contains visual maze navigation and path verification tasks based on the Frozen Lake environment.

        ### Dataset Summary

        - **Total samples**: {total_samples}
        - **Tasks**: Path Navigation and Path Verification
        - **Splits**:
        {splits_info}

        ### Task Types

        1. **Path Verification**: Given a maze image and an action sequence, determine if the path is safe (avoids holes and reaches the goal).
        2. **Path Navigation**: Given a maze image, generate an optimal action sequence from start to goal while avoiding holes.

        ### Data Fields

        Common fields:
        - `idx`: Unique identifier
        - `original_id`: Original data ID
        - `image`: RGB image of the maze
        - `text`: Task instruction and prompt
        - `answer`: Ground truth answer
        - `task_type`: Type of task (verify/navigation)
        - `split`: Data split (sft/rl/test)
        - `size`: Maze size (grid dimension)
        - `gym_map`: Gym environment map representation (JSON string)

        Path Verification specific fields:
        - `path_length`: Length of the action sequence
        - `path`: Action sequence string

        Path Navigation specific fields:
        - `start_coords`: Starting coordinates (JSON string)
        - `goal_coords`: Goal coordinates (JSON string)
        - `obstacle_coords`: Obstacle coordinates (JSON string)
        - `astar_path`: A* algorithm path (if available)

        ### Actions

        - `L`: Move left
        - `R`: Move right
        - `U`: Move up
        - `D`: Move down

        ### Maze Elements

        - `S`: Start position
        - `G`: Goal position
        - `F`: Frozen (safe) tile
        - `H`: Hole (unsafe)

        ## Usage

        ```python
        from datasets import load_dataset

        # Load specific split
        dataset = load_dataset("YOUR_USERNAME/frozen-lake-vsp", split="navigation_test")

        # Access an example
        example = dataset[0]
        image = example['image']
        prompt = example['text']
        answer = example['answer']
        ## Citation
        If you use this dataset, please cite:

        @dataset{{temp,
        title={{temp}},
        author={{Your Name}},
        year={{2024}}
        }}

        License
        Apache 2.0 """
    return readme

def main(): 
    parser = argparse.ArgumentParser(description='Upload Frozen Lake dataset to HuggingFace Hub') 
    parser.add_argument('--repo_id', type=str, default="hitsmy/AdaEval-VSPO", help='HuggingFace repository ID (username/dataset-name)') 
    parser.add_argument('--token', type=str, default=None, help='HuggingFace API token (or set HF_TOKEN env variable)') 
    parser.add_argument('--private', action='store_true', help='Make the dataset private') 
    parser.add_argument('--config', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/opensource/Tool-Factory-Filter/tool_server/tf_eval/tasks/vsp_test/config.yaml", help='Path to config file')
    
    args = parser.parse_args()

    # 获取token
    token = args.token or os.environ.get('HF_TOKEN')
    if not token:
        print("Warning: No HuggingFace token provided. You may need to login manually.")

    # 加载配置
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)

    # 加载数据
    print("Loading data...")
    meta_data = load_data_function(config)

    # 转换为HF Dataset
    print("Converting to HuggingFace Dataset format...")
    dataset_dict = convert_to_hf_dataset(meta_data)

    # 显示数据集信息
    print("\nDataset structure:")
    for name, ds in dataset_dict.items():
        print(f"  {name}: {len(ds)} samples")
        print(f"    Features: {list(ds.features.keys())}")

    # 上传到Hub
    upload_to_hub(dataset_dict, args.repo_id, token)

    print(f"\n✅ Dataset successfully uploaded to: https://huggingface.co/datasets/{args.repo_id}")

if __name__ == "__main__":
    main()