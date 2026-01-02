# action_safe_tool_supplement.py
import os
import json
import argparse
import copy
import base64
import io
from typing import Dict, List, Any, Tuple
from PIL import Image
import uuid
import random 

def convert_path_to_actions(path_string: str) -> str:
    """将路径字符串(如 'ddrr')转换为格式化的动作序列(如 'D,D,R,R')"""
    action_map = {'u': 'U', 'd': 'D', 'l': 'L', 'r': 'R'}
    actions = [action_map[char.lower()] for char in path_string if char.lower() in action_map]
    return ','.join(actions)

def pil_to_base64(img):
    """将PIL Image转换为base64字符串"""
    if isinstance(img, str):
        # 如果传入的是路径，先加载图像
        img = Image.open(img)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def create_tool_calls_for_entry(entry: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    为数据条目创建工具调用序列数据
    
    Args:
        entry: 原始数据集条目
        
    Returns:
        Tuple[Dict, Dict]: 包含工具调用数据的正确路径和随机路径条目
    """
    # 提取坐标和路径信息
    coordinates = entry["coordinates"]
    start_coords = coordinates["start"]
    holes_coords = coordinates["holes"]
    goal_coords = coordinates["goal"]
    
    # 为正确路径(gt_path)创建工具调用
    if "path_info" in entry:
        gt_path = entry["path_info"]["gt_path"]
        random_path = entry["path_info"]["random_path"]
    else:
        # 如果没有path_info，直接使用path字段
        if "path" in entry:
            path_parts = entry["path"].split(',')
            gt_path = ''.join([p.lower() for p in path_parts])
        else:
            # 如果没有任何路径信息，使用空字符串
            gt_path = ""
            random_path = ""
    
    gt_path_actions = convert_path_to_actions(gt_path)
    
    # 为正确路径创建工具链调用
    gt_tool_calls = generate_tool_chain_calls(
        entry["id"], 
        start_coords, 
        holes_coords,
        goal_coords,
        gt_path_actions, 
        is_random=False,
        image_path=entry["images"]["maze_image"] if "images" in entry else None,
        path_image=entry["images"]["vis_path_image"] if "images" in entry else None
    )
    
    # 为随机路径(random_path)创建工具调用
    if "path_info" in entry:
        random_path_actions = convert_path_to_actions(random_path)
        
        random_tool_calls = generate_tool_chain_calls(
            entry["id"], 
            start_coords, 
            holes_coords,
            goal_coords,
            random_path_actions, 
            is_random=True,
            image_path=entry["images"]["maze_image"] if "images" in entry else None,
            path_image=entry["images"]["random_path_image"] if "images" in entry else None
        )
    else:
        # 如果没有随机路径，复制正确路径并标记为不安全
        random_path_actions = gt_path_actions
        random_tool_calls = copy.deepcopy(gt_tool_calls)
        random_tool_calls["is_safe"] = False
    
    # 创建两个版本的条目：一个使用正确路径，一个使用随机路径
    gt_entry = copy.deepcopy(entry)
    random_entry = copy.deepcopy(entry)
    
    # 更新ID以区分两个条目
    if not gt_entry["id"].endswith("_gt"):
        gt_entry["id"] = f"{entry['id']}_gt"
    if not random_entry["id"].endswith("_random"):
        random_entry["id"] = f"{entry['id']}_random"
    
    # 添加工具调用数据
    gt_entry["tool_calls"] = gt_tool_calls
    random_entry["tool_calls"] = random_tool_calls
    
    # 添加路径信息
    gt_entry["path"] = gt_path_actions
    random_entry["path"] = random_path_actions
    
    # 添加期望输出
    gt_entry["output"] = "\\boxed{Yes}"  # 正确路径应该是安全的
    random_entry["output"] = "\\boxed{No}"  # 随机路径应该是不安全的
    
    # 添加指令文本
    instruction = """
You are a maze solver. Your goal is to guide a player from the start to the goal on a grid map while avoiding holes, using the fewest moves. The player can move one square at a time in the directions left (L), right (R), up (U), or down (D). Moving off the edge has no effect, and falling into a hole results in failure. Reaching the goal means success. 

Now please determine if the action sequence is safe for the given maze. Your final answer should be formatted as \\boxed{Yes} or \\boxed{No}.

<TEST-IMAGE>

The action sequence is:
<ACTION-SEQ>
"""
    gt_entry["instruction"] = instruction.strip().replace("<ACTION-SEQ>", gt_path_actions)
    random_entry["instruction"] = instruction.strip().replace("<ACTION-SEQ>", random_path_actions)
    
    return gt_entry, random_entry

def generate_tool_chain_calls(
    entry_id: str, 
    start_coords: List[float], 
    holes_coords: List[List[float]],
    goal_coords: List[float],
    path_actions: str, 
    is_random: bool = False,
    image_path: str = None,
    path_image: str = None
) -> Dict[str, Any]:
    """
    生成工具链调用序列
    
    Args:
        entry_id: 条目ID
        start_coords: 起点坐标
        holes_coords: 所有洞的坐标
        goal_coords: 目标坐标
        path_actions: 路径动作序列
        is_random: 是否为随机路径
        image_path: 原始图像路径
        path_image: 路径图像路径
        
    Returns:
        Dict: 包含工具调用序列的字典
    """
    # 生成唯一请求ID
    request_id = str(uuid.uuid4())
    
    # 1. 模拟Point工具调用 - 标记起点 (Elf)
    point_call = {
        "tool_name": "Point",
        "input": {
            "image": "img_1",  # 使用统一的图像引用方式
            "description": "Find the Elf (starting position) in the maze"
        },
        "output": {
            "tool_response_from": "Point",
            "status": "success",
            "points": [{"x": float(start_coords[0]), "y": float(start_coords[1])}],
            "message": "Successfully found the Elf at the specified coordinates.",
            "execution_time": random.uniform(0.05, 0.2),  # 模拟执行时间
            "request_id": request_id
        }
    }
    
    # 2. 模拟Draw2DPath工具调用 - 绘制路径
    path_type = "random" if is_random else "gt"
    draw_path_call = {
        "tool_name": "Draw2DPath",
        "input": {
            "image": "img_1",  # 使用统一的图像引用方式
            "start_point": [float(start_coords[0]), float(start_coords[1])],
            "directions": path_actions,  # 使用转换后的格式 "U,D,L,R"
            "step": 64,
            "pixel_coordinate": True,
            "line_width": 3,
            "line_color": "red"
        },
        "output": {
            "tool_response_from": "Draw2DPath",
            "status": "success",
            "message": f"Path drawn successfully",
            "execution_time": random.uniform(0.05, 0.2),
            "request_id": request_id
        }
    }
    
    # 记录路径是否安全
    is_safe = not is_random  # 假设正确路径是安全的，随机路径是不安全的
    
    # 创建完整的工具调用序列信息
    tool_calls_data = {
        "sequence": [point_call, draw_path_call],
        "final_image": path_image,
        "original_image": image_path,
        "is_safe": is_safe,
        "path": path_actions,
        "coordinates": {
            "start": start_coords,
            "holes": holes_coords,
            "goal": goal_coords
        }
    }
    
    return tool_calls_data

def add_tool_calls_to_dataset(input_jsonl_path: str, output_jsonl_path: str):
    """
    为整个数据集添加工具调用数据
    
    Args:
        input_jsonl_path: 输入JSONL文件路径
        output_jsonl_path: 输出JSONL文件路径
    """
    # 读取原始数据集
    entries = []
    with open(input_jsonl_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {e}")
    
    print(f"Read {len(entries)} entries from {input_jsonl_path}")
    
    # 为每个条目添加工具调用数据
    processed_entries = []
    
    for entry in entries:
        try:
            # 处理条目
            if "_gt" in entry["id"] or entry["output"] == "\\boxed{Yes}":
                # 已经是正确路径条目
                if "tool_calls" not in entry:
                    # 只有在没有工具调用数据时才添加
                    start_coords = entry["coordinates"]["start"]
                    holes_coords = entry["coordinates"]["holes"]
                    goal_coords = entry["coordinates"]["goal"]
                    path_actions = entry["path"]
                    
                    tool_calls = generate_tool_chain_calls(
                        entry["id"],
                        start_coords,
                        holes_coords,
                        goal_coords,
                        path_actions,
                        is_random=False,
                        image_path=entry["images"]["maze_image"] if "images" in entry else None,
                        path_image=entry["images"]["vis_path_image"] if "images" in entry else None
                    )
                    
                    entry["tool_calls"] = tool_calls
                
                processed_entries.append(entry)
            
            elif "_random" in entry["id"] or entry["output"] == "\\boxed{No}":
                # 已经是随机路径条目
                if "tool_calls" not in entry:
                    # 只有在没有工具调用数据时才添加
                    start_coords = entry["coordinates"]["start"]
                    holes_coords = entry["coordinates"]["holes"]
                    goal_coords = entry["coordinates"]["goal"]
                    path_actions = entry["path"]
                    
                    tool_calls = generate_tool_chain_calls(
                        entry["id"],
                        start_coords,
                        holes_coords,
                        goal_coords,
                        path_actions,
                        is_random=True,
                        image_path=entry["images"]["maze_image"] if "images" in entry else None,
                        path_image=entry["images"]["random_path_image"] if "images" in entry else None
                    )
                    
                    entry["tool_calls"] = tool_calls
                
                processed_entries.append(entry)
            
            else:
                # 没有明确标记的条目，生成两个版本
                gt_entry, random_entry = create_tool_calls_for_entry(entry)
                processed_entries.append(gt_entry)
                processed_entries.append(random_entry)
        
        except Exception as e:
            print(f"Error processing entry {entry.get('id', 'unknown')}: {e}")
    
    # 输出结果为新的JSONL文件
    with open(output_jsonl_path, 'w') as f:
        for entry in processed_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"处理完成！共生成 {len(processed_entries)} 条记录")
    print(f"结果已保存到: {output_jsonl_path}")
    
    # 数据统计
    gt_count = sum(1 for entry in processed_entries if entry["output"] == "\\boxed{Yes}")
    random_count = sum(1 for entry in processed_entries if entry["output"] == "\\boxed{No}")
    
    print(f"正确路径记录: {gt_count} ({gt_count/len(processed_entries)*100:.1f}%)")
    print(f"随机路径记录: {random_count} ({random_count/len(processed_entries)*100:.1f}%)")
    
    return processed_entries

def generate_openai_conversation_format(entries, output_file):
    """
    将数据条目转换为OpenAI对话格式
    
    Args:
        entries: 数据条目列表
        output_file: 输出文件路径
    """
    conversations = []
    
    for entry in entries:
        # 创建对话ID
        conversation_id = entry["id"]
        
        # 准备系统消息
        system_message = {
            "role": "system",
            "content": """You are a visual assistant capable of solving visual reasoning problems. You can rely on your own capabilities or use external tools to assist in solving."""
        }
        
        # 准备用户消息
        # 替换指令中的占位符
        instruction = entry["instruction"]
        instruction = instruction.replace("<TEST-IMAGE>", "")  # 图像将通过图像URL提供
        
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": instruction
                },
                {
                    "type": "image",
                    "image": {
                        "image": f"{entry['images']['maze_image']}"
                    }
                }
            ]
        }
        
        # 准备助手消息 - 思考过程
        assistant_thinking = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": """<think>
I need to determine if the given action sequence is safe for the maze. Let me first locate the player's starting position using the Point tool, then visualize the path using the Draw2DPath tool to see if it avoids all holes.
</think>

<tool_call>
{"name": "Point", "parameters": {"image": "img_1", "description": Elf"}}
</tool_call>"""
                }
            ]
        }
        
        # 准备工具输出消息 - Point工具
        point_output = entry["tool_calls"]["sequence"][0]["output"]
        tool_message_1 = {
            "role": "user",
            "content": json.dumps(point_output)
        }
        
        # 准备助手消息 - 绘制路径
        assistant_draw_path = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"""<think>
Now that I have identified the starting position at coordinates [{point_output['points'][0]['x']}, {point_output['points'][0]['y']}], I'll visualize the path using the Draw2DPath tool with the given action sequence: {entry['path']}. This will help me see if the path is safe or if it crosses any holes.
</think>

<tool_call>
{{"name": "Draw2DPath", "parameters": {{"image": "img_1", "start_point": [{point_output['points'][0]['x']}, {point_output['points'][0]['y']}], "directions": "{entry['path']}", "step": 64, "pixel_coordinate": true, "line_width": 3, "line_color": "red"}}}}
</tool_call>"""
                }
            ]
        }
        
        # 准备工具输出消息 - Draw2DPath工具
        draw_path_output = entry["tool_calls"]["sequence"][1]["output"]
        tool_message_2 = {
            "role": "tool",
            "name": "Draw2DPath",
            "content": json.dumps(draw_path_output)
        }
        
        # 准备最终助手回答
        final_answer = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"""<think>
Now I have visualized the path from the starting position following the action sequence: {entry['path']}. 

Based on the drawn path, I can see whether the sequence is safe or not. A safe path must avoid all holes in the maze.

{'The path successfully avoids all holes and reaches the goal safely.' if entry['output'] == '\\boxed{Yes}' else 'The path is unsafe because it passes through at least one hole.'}
</think>

<response>
After visualizing the path from the starting position and following the action sequence {entry['path']}, I can determine that this path is {entry['output'].replace('\\boxed{', '').replace('}', '')} safe.

{entry['output']}
</response>"""
                }
            ]
        }
        
        # 组装对话
        conversation = {
            "id": conversation_id,
            "messages": [
                system_message,
                user_message,
                assistant_thinking,
                tool_message_1,
                assistant_draw_path,
                tool_message_2,
                final_answer
            ]
        }
        
        conversations.append(conversation)
    
    # 保存为JSONL格式
    with open(output_file, 'w') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')
    
    print(f"生成了 {len(conversations)} 个OpenAI格式的对话，保存到 {output_file}")

def process_single_example():
    """处理单个示例以演示功能"""
    example = {
        "id": "frozenlake_s3_7", 
        "size": 3, 
        "map": ["SFF", "FFH", "FFG"], 
        "path_info": {
            "gt_path": "ddrr", 
            "random_path": "drrd", 
            "random_path_safe": False
        }, 
        "coordinates": {
            "start": [32.0, 32.0], 
            "holes": [[160.0, 96.0]], 
            "goal": [160.0, 160.0]
        }, 
        "images": {
            "system_figure_1": "./assets/system-figure-1.png",
            "system_figure_2": "./assets/system-figure-2.png",
            "maze_image": "./frozen_lake_dataset_new/images/size_3/frozenlake_s3_7.png", 
            "vis_path_image": "./frozen_lake_dataset_new/images/size_3/frozenlake_s3_7_path.png", 
            "random_path_image": "./frozen_lake_dataset_new/images/size_3/frozenlake_s3_7_random_path.png"
        }
    }
    
    gt_entry, random_entry = create_tool_calls_for_entry(example)
    
    print("===== 正确路径工具调用数据 =====")
    print(json.dumps(gt_entry["tool_calls"], indent=2))
    print("\n===== 随机路径工具调用数据 =====")
    print(json.dumps(random_entry["tool_calls"], indent=2))
    
    # 生成OpenAI格式对话示例
    generate_openai_conversation_format([gt_entry, random_entry], "example_conversations.jsonl")
    
    return gt_entry, random_entry

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="为FrozenLake数据集添加工具调用数据")
    parser.add_argument("--input", type=str, default="./frozen_lake_dataset_new/action_plan_dataset.jsonl",
                       help="输入JSONL文件路径")
    parser.add_argument("--output", type=str, default="./frozen_lake_dataset_new/action_plan_dataset_with_tools.jsonl",
                       help="输出JSONL文件路径")
    parser.add_argument("--conversations", type=str, default="./frozen_lake_dataset_new/conversations.jsonl",
                       help="OpenAI格式对话输出路径")
    parser.add_argument("--demo", action="store_true",
                       help="运行单个示例演示")
    parser.add_argument("--only-conversations", action="store_true",
                       help="仅生成对话格式，不修改原始数据集")
    parser.add_argument("--sample-count", type=int, default=None,
                       help="限制生成的对话数量")
    
    args = parser.parse_args()
    
    if args.demo:
        gt_entry, random_entry = process_single_example()
        print("\n示例已保存到 example_with_tools.jsonl 和 example_conversations.jsonl")
    else:
        if args.only_conversations:
            # 直接从输入文件生成对话格式
            entries = []
            with open(args.input, 'r') as f:
                for line in f:
                    entries.append(json.loads(line.strip()))
                    
            if args.sample_count and args.sample_count < len(entries):
                import random
                entries = random.sample(entries, args.sample_count)
                
            generate_openai_conversation_format(entries, args.conversations)
        else:
            # 处理数据集并生成对话格式
            processed_entries = add_tool_calls_to_dataset(args.input, args.output)
            
            if args.sample_count and args.sample_count < len(processed_entries):
                import random
                processed_entries = random.sample(processed_entries, args.sample_count)
                
            generate_openai_conversation_format(processed_entries, args.conversations)

if __name__ == "__main__":
    main()