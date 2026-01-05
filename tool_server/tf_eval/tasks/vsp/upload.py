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

def convert_markdown_table_to_map(table_str):
    """将Markdown格式的表格转换为FrozenLake风格的地图表示"""
    rows = [row.strip() for row in table_str.strip().split('\n') if '|' in row]
    
    # 忽略表头和分隔符行
    if len(rows) >= 2 and all('-' in cell for cell in rows[1].split('|')):
        rows = [rows[0]] + rows[2:]
    
    # 提取实际的地图内容
    map_rows = []
    for row in rows:
        cells = [cell.strip() for cell in row.split('|')]
        cells = [cell for cell in cells if cell and not cell.startswith('Row')]
        map_rows.append(cells)
    
    # 构建FrozenLake风格的地图
    frozen_lake_map = []
    for row in map_rows:
        map_row = ""
        for cell in row:
            if cell == '@':
                map_row += 'S'
            elif cell == '#':
                map_row += 'H'
            elif cell == '*':
                map_row += 'G'
            elif cell == '_':
                map_row += 'F'
            else:
                continue
        if map_row:
            frozen_lake_map.append(map_row)
    
    return frozen_lake_map

def load_path_validation_data(task_dir, task_type, num_samples=None):
    """加载路径验证任务数据（task4）"""
    meta_data = []
    prompt_text = PATH_VERIFY_TASK_INSTRUCTION
    
    levels = [1, 3, 5, 7, 9]
    
    for level in levels:
        map_dir = os.path.join(task_dir, "maps", f"level_step{level}")
        img_dir = os.path.join(map_dir, "img")
        question_dir = os.path.join(map_dir, "question")
        answer_dir = os.path.join(map_dir, "answer")
        
        if not os.path.exists(img_dir) or not os.path.exists(question_dir):
            continue
            
        files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        if num_samples:
            max_per_level = max(1, num_samples // len(levels))
            files = files[:min(len(files), max_per_level)]
        
        for file in files:
            idx = file.split('.')[0]
            img_path = os.path.join(img_dir, file)
            
            question_path = os.path.join(question_dir, f"{idx}.txt")
            answer_path = os.path.join(answer_dir, f"{idx}.txt")
            
            if os.path.exists(question_path) and os.path.exists(answer_path):
                with open(question_path, "r") as f:
                    question = f.read().strip()
                with open(answer_path, "r") as f:
                    answer = f.read().strip()
                
                text_prompt = prompt_text.replace("<ACTION-SEQ>", question)
                
                meta_data.append({
                    "idx": f"vsp_{task_type}_level{level}_{idx}",
                    "original_id": f"{task_type}_level{level}_{idx}",
                    "image_path": img_path,
                    "text": text_prompt,
                    "answer": answer,
                    "task_type": "verify",
                    "split": "test",
                    "size": level,
                    "level": level,
                    "path_length": len(question.split(",")),
                    "path": question,
                    "gym_map": []  # task4没有gym_map信息
                })
    
    print(f"Loaded {len(meta_data)} records for {task_type}")
    return meta_data

def load_planning_data(task_dir, task_type, num_samples=None):
    """加载路径规划任务数据（task-main）"""
    meta_data = []
    prompt_text = PATH_NAVIGATION_INSTRUCTION
    
    levels = [3, 4, 5, 6, 7, 8]
    
    for level in levels:
        map_dir = os.path.join(task_dir, "maps", f"level{level}")
        img_dir = os.path.join(map_dir, "img")
        text_map_dir = os.path.join(map_dir, "table")
        
        if not os.path.exists(img_dir) or not os.path.exists(text_map_dir):
            continue
            
        files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        if num_samples:
            max_per_level = max(1, num_samples // len(levels))
            files = files[:min(len(files), max_per_level)]
        
        for file in files:
            idx = file.split('.')[0]
            img_path = os.path.join(img_dir, f"{idx}.png")
            map_text_path = os.path.join(text_map_dir, f"{idx}.txt")
            
            if os.path.exists(img_path) and os.path.exists(map_text_path):
                with open(map_text_path, "r") as f:
                    map_text = f.read()
                
                map_list = convert_markdown_table_to_map(map_text)
                
                # 从map_list中提取坐标信息
                start_coords = None
                goal_coords = None
                obstacle_coords = []
                
                for i, row in enumerate(map_list):
                    for j, cell in enumerate(row):
                        if cell == 'S':
                            start_coords = [i, j]
                        elif cell == 'G':
                            goal_coords = [i, j]
                        elif cell == 'H':
                            obstacle_coords.append([i, j])
                
                meta_data.append({
                    "idx": f"vsp_{task_type}_level{level}_{idx}",
                    "original_id": f"{task_type}_level{level}_{idx}",
                    "image_path": img_path,
                    "text": prompt_text,
                    "answer": "DYNAMIC_EVAL",
                    "task_type": "navigation",
                    "split": "test",
                    "size": level,
                    "level": level,
                    "start_coords": start_coords,
                    "goal_coords": goal_coords,
                    "obstacle_coords": obstacle_coords,
                    "astar_path": "",
                    "gym_map": map_list
                })
    
    print(f"Loaded {len(meta_data)} records for {task_type}")
    return meta_data

def load_data_function(config):
    """加载VSP数据集的函数"""
    dataset_path = config.get("dataset_path")
    tasks = config.get("tasks", ["task-main", "task4"])
    num_samples = config.get("num_sample")
    
    meta_data = []
    
    for task in tasks:
        task_dir = os.path.join(dataset_path, "maze", task)
        if not os.path.exists(task_dir):
            print(f"Warning: Task directory not found: {task_dir}")
            continue
            
        # 根据任务类型选择适当的数据加载逻辑
        if task == "task4":
            # 路径验证任务
            meta_data.extend(load_path_validation_data(task_dir, task, num_samples))
        elif task == "task-main":
            # 路径规划任务
            meta_data.extend(load_planning_data(task_dir, task, num_samples))
    
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
        'level': Value('int32'),
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
                    'level': item.get('level', item['size']),
                    'gym_map': json.dumps(item['gym_map']),
                    # verify任务字段（如果不存在则为None）
                    'path_length': item.get('path_length', None),
                    'path': item.get('path', None),
                    # navigation任务字段（如果不存在则为None）
                    'start_coords': json.dumps(item['start_coords']) if 'start_coords' in item and item['start_coords'] is not None else None,
                    'goal_coords': json.dumps(item['goal_coords']) if 'goal_coords' in item and item['goal_coords'] is not None else None,
                    'obstacle_coords': json.dumps(item['obstacle_coords']) if 'obstacle_coords' in item and item['obstacle_coords'] is not None else None,
                    'astar_path': item.get('astar_path', None),
                }
                
                processed_items.append(data_item)
                
            except Exception as e:
                print(f"Error processing item {item['idx']}: {e}")
                import traceback
                traceback.print_exc()
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
            private=False,
            commit_message="Upload VSP Frozen Lake dataset"
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
    
    readme = f"""# Visual Spatial Planning (VSP) Dataset

            ## Dataset Description

            This dataset contains visual maze navigation and path verification tasks from the Visual Spatial Planning benchmark.

            ### Dataset Summary

            - **Total samples**: {total_samples}
            - **Tasks**: Path Navigation (task-main) and Path Verification (task4)
            - **Splits**:
            {splits_info}

            ### Task Types

            1. **Path Verification (task4)**: Given a maze image and an action sequence, determine if the path is safe (avoids holes and reaches the goal).
            2. **Path Navigation (task-main)**: Given a maze image, generate an optimal action sequence from start to goal while avoiding holes.

            ### Data Fields

            Common fields:
            - `idx`: Unique identifier
            - `original_id`: Original data ID
            - `image`: RGB image of the maze
            - `text`: Task instruction and prompt
            - `answer`: Ground truth answer
            - `task_type`: Type of task (verify/navigation)
            - `split`: Data split (test)
            - `size`: Maze size (grid dimension)
            - `level`: Difficulty level
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
            dataset = load_dataset("hitsmy/AdaEval-VSPO", split="navigation_test")

            # Access an example
            example = dataset[0]
            image = example['image']
            prompt = example['text']
            answer = example['answer']

            ## Citation

            If you use this dataset, please cite:

            @dataset{{vsp_dataset,
            title={{Visual Spatial Planning Dataset}},
            author={{VSP Team}},
            year={{2024}}
            }}

            ##License

            Apache 2.0 """
    return readme



def main(): 
    parser = argparse.ArgumentParser(description='Upload VSP dataset to HuggingFace Hub') 
    parser.add_argument('--repo_id', type=str, default="hitsmy/AdaEval-VSP", help='HuggingFace repository ID (username/dataset-name)') 
    parser.add_argument('--token', type=str, default=None, help='HuggingFace API token (or set HF_TOKEN env variable)') 
    parser.add_argument('--private', action='store_true', help='Make the dataset private') 
    parser.add_argument('--config', type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/opensource/Tool-Factory-Filter/tool_server/tf_eval/tasks/vsp/config.yaml", help='Path to config file')
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