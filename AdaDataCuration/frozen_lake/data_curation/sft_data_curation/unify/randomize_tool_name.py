import json
import os
import re
import random
import string
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set

# 定义常用工具名称和参数名称，用于识别和替换
COMMON_TOOL_NAMES = {
    "Point": ["image", "description"],
    "Draw2DPath": ["image", "start_point", "directions", "step", "pixel_coordinate", "line_width", "line_color"],
    "AStarWithPixelCoordinate": ["start", "goal", "obstacles"],
    "DetectBlackArea": ["image", "threshold", "min_area"],
    "InsertImage": ["image", "insert_image", "coordinates", "resize"]
}

def generate_random_name(length=6, prefix=""):
    """生成随机名称"""
    chars = string.ascii_lowercase + string.digits
    random_str = ''.join(random.choices(chars, k=length))
    return f"{prefix}{random_str}"

def generate_replacement_map(tool_names, param_names):
    """
    生成替换映射表
    
    Args:
        tool_names: 需要替换的工具名称列表
        param_names: 需要替换的参数名称列表
        
    Returns:
        dict: 映射表 {原名称: 新名称}
    """
    replacement_map = {}
    
    # 为工具名称生成替换
    for tool_name in tool_names:
        replacement_map[tool_name] = generate_random_name(prefix="")
    
    # 为参数名称生成替换
    for param_name in param_names:
        replacement_map[param_name] = generate_random_name(prefix="")
    
    return replacement_map

def extract_tools_and_params(instance: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    """
    从整个实例中提取工具名称和参数名称
    
    Args:
        instance: 数据实例
        
    Returns:
        Tuple[Set[str], Set[str]]: 工具名称集合和参数名称集合
    """
    tool_names = set()
    param_names = set()
    
    # 遍历对话
    if "conversations" in instance:
        for msg in instance["conversations"]:
            msg_from = msg.get("from", "")
            content = msg.get("value", "")
            
            if msg_from == "system":
                # 处理系统提示
                system_tools, system_params = extract_from_system_prompt(content)
                tool_names.update(system_tools)
                param_names.update(system_params)
            elif msg_from == "gpt":
                # 处理GPT回复，查找tool_call
                call_tools, call_params = extract_from_tool_call(content)
                tool_names.update(call_tools)
                param_names.update(call_params)
            elif msg_from == "human":
                # 处理Human回复，可能包含工具响应
                response_tools, response_params = extract_from_tool_response(content)
                tool_names.update(response_tools)
                param_names.update(response_params)
    
    # 添加常见工具和参数
    for tool_name in tool_names.copy():
        if tool_name in COMMON_TOOL_NAMES:
            param_names.update(COMMON_TOOL_NAMES[tool_name])
    
    return tool_names, param_names

def extract_from_system_prompt(system_prompt: str) -> Tuple[Set[str], Set[str]]:
    """从系统提示中提取工具名称和参数"""
    tool_names = set()
    param_names = set()
    
    # 尝试提取工具定义
    # 1. 查找function定义
    function_patterns = [
        r"'function':\s*{\s*'name':\s*'([^']+)'.*?'parameters':\s*{(.*?)}", # 单引号格式
        r'"function":\s*{\s*"name":\s*"([^"]+)".*?"parameters":\s*{(.*?)}' # 双引号格式
    ]
    
    for pattern in function_patterns:
        matches = re.finditer(pattern, system_prompt, re.DOTALL)
        for match in matches:
            tool_name = match.group(1)
            params_text = match.group(2)
            tool_names.add(tool_name)
            
            # 提取参数名称
            param_patterns = [r"'([^']+)':\s*{", r'"([^"]+)":\s*{']
            for param_pattern in param_patterns:
                param_matches = re.finditer(param_pattern, params_text)
                for param_match in param_matches:
                    param_name = param_match.group(1)
                    param_names.add(param_name)
    
    # 2. 检查已知工具引用
    for tool_name, params in COMMON_TOOL_NAMES.items():
        if tool_name in system_prompt:
            tool_names.add(tool_name)
            for param in params:
                # 只在这些参数作为独立词出现时添加，避免误匹配
                if re.search(r'[\'"]' + re.escape(param) + r'[\'"]', system_prompt):
                    param_names.add(param)
    
    return tool_names, param_names

def extract_from_tool_call(content: str) -> Tuple[Set[str], Set[str]]:
    """从工具调用中提取工具名称和参数"""
    tool_names = set()
    param_names = set()
    
    # 查找<tool_call>标签
    tool_call_blocks = re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', content, re.DOTALL)
    for block in tool_call_blocks:
        tool_call_text = block.group(1)
        
        # 尝试解析为JSON
        try:
            # 如果是格式良好的JSON，直接解析
            if tool_call_text.strip().startswith('{') and tool_call_text.strip().endswith('}'):
                tool_call_obj = json.loads(tool_call_text)
                if 'name' in tool_call_obj:
                    tool_name = tool_call_obj['name']
                    tool_names.add(tool_name)
                
                if 'parameters' in tool_call_obj:
                    for param_name in tool_call_obj['parameters']:
                        param_names.add(param_name)
                        
        except json.JSONDecodeError:
            # 如果不是格式良好的JSON，使用正则表达式提取
            name_match = re.search(r'"name":\s*"([^"]+)"', tool_call_text)
            if name_match:
                tool_name = name_match.group(1)
                tool_names.add(tool_name)
            
            # 提取参数名称
            param_matches = re.finditer(r'"([^"]+)":\s*[{\[]', tool_call_text)
            for param_match in param_matches:
                param_name = param_match.group(1)
                if param_name != 'parameters':  # 跳过parameters本身
                    param_names.add(param_name)
    
    return tool_names, param_names

def extract_from_tool_response(content: str) -> Tuple[Set[str], Set[str]]:
    """从工具响应中提取工具名称和参数"""
    tool_names = set()
    param_names = set()
    
    # 查找工具响应JSON
    json_blocks = find_json_blocks(content)
    for json_text in json_blocks:
        try:
            json_obj = json.loads(json_text)
            if isinstance(json_obj, dict):
                if 'tool_response_from' in json_obj:
                    tool_name = json_obj['tool_response_from']
                    tool_names.add(tool_name)
                
                # 递归检查所有键以查找参数
                for key in json_obj:
                    if key not in ["error_code", "status", "message", "execution_time", "width", "height"]:
                        param_names.add(key)
                        
                    # 特别处理嵌套结构，如points下的x,y
                    if key == "points" and isinstance(json_obj[key], list):
                        for point in json_obj[key]:
                            if isinstance(point, dict):
                                for point_key in point:
                                    param_names.add(point_key)
                
                # 检查image_dimensions_pixels内部
                if 'image_dimensions_pixels' in json_obj and isinstance(json_obj['image_dimensions_pixels'], dict):
                    dims = json_obj['image_dimensions_pixels']
                    for dim_key in dims:
                        param_names.add(dim_key)
        except json.JSONDecodeError:
            continue
    
    return tool_names, param_names

def find_json_blocks(text: str) -> List[str]:
    """查找文本中的JSON块"""
    json_blocks = []
    
    # 尝试找到括号匹配的JSON块
    # 这个简单方法不能处理所有嵌套情况，但对于大多数情况已足够
    start_indices = [m.start() for m in re.finditer(r'{\s*"', text)]
    for start in start_indices:
        # 尝试从这个位置开始找到一个有效的JSON
        bracket_count = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                bracket_count += 1
            elif text[i] == '}':
                bracket_count -= 1
                if bracket_count == 0:
                    # 找到一个完整的JSON块
                    json_text = text[start:i+1]
                    try:
                        # 验证是否是有效JSON
                        json.loads(json_text)
                        json_blocks.append(json_text)
                        break
                    except json.JSONDecodeError:
                        # 不是有效JSON，继续寻找
                        continue
    
    return json_blocks

def replace_json_in_text(text: str, replacement_map: Dict[str, str]) -> str:
    """替换文本中JSON部分的工具名称和参数"""
    json_blocks = find_json_blocks(text)
    
    # 按长度降序排序，确保先替换较长的块
    json_blocks.sort(key=len, reverse=True)
    
    for json_text in json_blocks:
        try:
            json_obj = json.loads(json_text)
            replaced_obj = replace_in_json_object(json_obj, replacement_map)
            replaced_json = json.dumps(replaced_obj)
            
            # 确保不会部分替换更长的字符串
            if replaced_json != json_text:
                text = text.replace(json_text, replaced_json)
        except json.JSONDecodeError:
            continue
    
    return text

def replace_in_json_object(obj: Any, replacement_map: Dict[str, str]) -> Any:
    """递归替换JSON对象中的值"""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # 替换键
            new_key = replacement_map.get(k, k)
            # 递归替换值
            new_val = replace_in_json_object(v, replacement_map)
            
            # 特殊处理工具名称和参数值
            if k == "name" and isinstance(v, str) and v in replacement_map:
                new_val = replacement_map[v]
            elif k == "tool_response_from" and isinstance(v, str) and v in replacement_map:
                new_val = replacement_map[v]
                
            new_dict[new_key] = new_val
                
        return new_dict
    elif isinstance(obj, list):
        return [replace_in_json_object(item, replacement_map) for item in obj]
    elif isinstance(obj, str) and obj in replacement_map:
        # 如果字符串值匹配替换映射中的工具名
        return replacement_map[obj]
    else:
        return obj

def replace_tool_call_block(text: str, replacement_map: Dict[str, str]) -> str:
    """替换<tool_call>标签内的工具名称和参数"""
    def replace_match(match):
        tool_call_text = match.group(1)
        
        try:
            # 尝试作为JSON解析并替换
            if tool_call_text.strip().startswith('{') and tool_call_text.strip().endswith('}'):
                tool_call_obj = json.loads(tool_call_text)
                replaced_obj = replace_in_json_object(tool_call_obj, replacement_map)
                replaced_text = json.dumps(replaced_obj, indent=2)
                return f"<tool_call>\n{replaced_text}\n</tool_call>"
        except json.JSONDecodeError:
            pass
        
        # 如果不是有效的JSON，使用正则替换
        for original, replacement in replacement_map.items():
            # 替换工具名称
            tool_call_text = re.sub(
                f'"name"\\s*:\\s*"({re.escape(original)})"', 
                f'"name": "{replacement}"', 
                tool_call_text
            )
            
            # 替换参数名
            tool_call_text = re.sub(
                f'"({re.escape(original)})"\\s*:', 
                f'"{replacement}":', 
                tool_call_text
            )
        
        return f"<tool_call>\n{tool_call_text}\n</tool_call>"
    
    return re.sub(r'<tool_call>\s*(.*?)\s*</tool_call>', replace_match, text, flags=re.DOTALL)

def replace_system_prompt(system_prompt: str, replacement_map: Dict[str, str]) -> str:
    """替换系统提示中的工具名称和参数"""
    replaced_prompt = system_prompt
    
    description_jsons = system_prompt.split('In your response, you can use the following tools:  \n')[-1].split("\n\nSteps for Each Turn\n1. **Think:** First, silently analyze the user's request to understand the goal. ")[0]
    
    prefix = system_prompt.split('In your response, you can use the following tools:  \n')[0] + 'In your response, you can use the following tools:  \n'
    suffix = "\n\nSteps for Each Turn\n1. **Think:** First, silently analyze the user's request to understand the goal. " + system_prompt.split("\n\nSteps for Each Turn\n1. **Think:** First, silently analyze the user's request to understand the goal. ")[-1]
    
    description_jsons = description_jsons.split("\n")
    renewed_dicts = []
    for description_json in description_jsons:
        desc_dict = eval(description_json)
        desc_dict["function"]["name"] = replacement_map[desc_dict["function"]["name"]]
        old_params = desc_dict["function"]["parameters"]
        desc_dict["function"]["parameters"]["properties"] = {replacement_map.get(k, k): v for k, v in old_params["properties"].items()}
        desc_dict["function"]["parameters"]["required"] = [replacement_map.get(k, k) for k in old_params.get("required", [])]
        renewed_dicts.append(desc_dict)
    
    
    renewed_description = "\n".join([str(d) for d in renewed_dicts])
    replaced_prompt = prefix + renewed_description + suffix
    return replaced_prompt

def replace_in_text(text: str, replacement_map: Dict[str, str]) -> str:
    """替换文本中的工具名称和参数名称，保留格式"""
    # 排序替换映射，确保先替换较长的词，以避免部分替换
    sorted_replacements = sorted(replacement_map.items(), key=lambda x: len(x[0]), reverse=True)
    replaced_text = text
    
    # 1. 检查并替换工具调用块
    if "<tool_call>" in replaced_text:
        replaced_text = replace_tool_call_block(replaced_text, replacement_map)
    
    # 2. 检查并替换JSON块
    replaced_text = replace_json_in_text(replaced_text, replacement_map)
    
    # 3. 直接替换其他引用，但必须确保是完整的词
    for original, replacement in sorted_replacements:
        # 工具名称替换
        replaced_text = re.sub(
            f'\\b{re.escape(original)}\\b', 
            replacement, 
            replaced_text
        )
    
    return replaced_text

def replace_tool_call_in_text(text: str, replacement_map: Dict[str, str]) -> str:
    replaced_tool_call = text
    if "<tool_call>" in text:
        tool_call_json = text.split('<tool_call>')[-1].split('</tool_call>')[0]
        prefix = text.split('<tool_call>')[0] + '<tool_call>'
        suffix = '</tool_call>' + text.split('</tool_call>')[-1]
        tool_call_json = tool_call_json.replace("true","\"true\"")
        tool_call_dict = eval(tool_call_json)
        
        tool_call_dict["name"] = replacement_map.get(tool_call_dict["name"], tool_call_dict["name"])
        
        tool_call_dict["parameters"] = {replacement_map.get(k, k): v for k, v in tool_call_dict["parameters"].items()}
        
        replaced_tool_call = prefix + "\n" + json.dumps(tool_call_dict, indent=2) + "\n" + suffix
    assert "<think>" in replaced_tool_call
    
    think_content = replaced_tool_call.split('<think>')[-1].split('</think>')[0]
    t_suffix = '</think>' + replaced_tool_call.split('</think>')[-1]
    for name,sub in replacement_map.items():
        if name in ["Point", "Draw2DPath", "AStarWithPixelCoordinate", "DetectBlackArea", "InsertImage"]:
            think_content = re.sub(f'\\b{name}\\b', sub, think_content)
    replaced_tool_call = replaced_tool_call.split('<think>')[0] + '<think>' + think_content + t_suffix
        
    return replaced_tool_call

def replace_tool_resp_in_text(text: str, replacement_map: Dict[str, str]) -> str:
    json_text = text.replace("<image>","")
    resp_dict = eval(json_text)
    resp_dict["tool_response_from"] = replacement_map.get(resp_dict["tool_response_from"], resp_dict["tool_response_from"])
    
    replaced_text = f"{json.dumps(resp_dict, indent=2)}<image>"
    return replaced_text

def process_conversation_message(msg: Dict[str, Any], replacement_map: Dict[str, str]) -> Dict[str, Any]:
    """根据消息类型处理对话消息"""
    msg_type = msg.get("from", "")
    content = msg.get("value", "")
    
    if "value" not in msg:
        return msg
    
    if msg_type == "system":
        # 系统提示特殊处理
        msg["value"] = replace_system_prompt(content, replacement_map)
    elif msg_type == "gpt":
        # 包含工具调用的GPT回复
        msg["value"] = replace_tool_call_in_text(content, replacement_map)
    elif msg_type == "human" and "tool_response_from" in content:
        # 包含工具响应的Human回复
        msg["value"] = replace_tool_resp_in_text(content, replacement_map)
    else:
        pass
        
    
    return msg



def randomize_instance(instance: Dict[str, Any]) -> Dict[str, Any]:
    """随机化一个数据实例的工具名称和参数"""
    # 1. 提取工具名称和参数
    tool_names, param_names = extract_tools_and_params(instance)
    
    if not tool_names:
        # 如果没有找到工具名称，直接返回原始实例
        return instance
    
    # 2. 生成替换映射
    replacement_map = generate_replacement_map(tool_names, param_names)
    
    # 3. 应用替换
    randomized_instance = instance.copy()
    
    # 3.1 处理对话
    if "conversations" in randomized_instance:
        for i, msg in enumerate(randomized_instance["conversations"]):
            randomized_instance["conversations"][i] = process_conversation_message(msg, replacement_map)
    
    return randomized_instance

def randomize_dataset(input_file, output_file):
    """随机化整个数据集"""
    # 加载数据
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # 尝试加载JSONL
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]
    
    # 随机化每个实例
    randomized_data = []
    for instance in tqdm(data, desc="Randomizing instances"):
        randomized_instance = randomize_instance(instance)
        randomized_data.append(randomized_instance)
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(randomized_data, f, indent=2, ensure_ascii=False)
    
    print(f"Randomized dataset saved to {output_file}")
    print(f"Original instances: {len(data)}, Randomized instances: {len(randomized_data)}")
    
    # 统计修改信息
    total_tools = 0
    total_params = 0
    
    for instance in data:
        tools, params = extract_tools_and_params(instance)
        total_tools += len(tools)
        total_params += len(params)
    
    print(f"Total tool names replaced: {total_tools}")
    print(f"Total parameter names replaced: {total_params}")

def main():
    parser = argparse.ArgumentParser(description="Randomize tool names and parameters in training data")
    parser.add_argument("--input", type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/merged_sft_file/vsp2tasks_v2_navigationa.json", help="Input JSON file path")
    parser.add_argument("--output", type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/merged_sft_file/randomized/vsp2tasks_v2_navigationa.json", help="Output JSON file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 随机化数据集
    randomize_dataset(args.input, args.output)

if __name__ == "__main__":
    main()