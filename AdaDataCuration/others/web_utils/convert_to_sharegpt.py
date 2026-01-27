import json
import uuid
from typing import List, Dict, Any
from tool_server.utils.utils import *



def convert_to_sharegpt_format(input_data: List[Dict]) -> List[Dict]:
    """将输入数据转换为ShareGPT格式"""
    sharegpt_data = []
    
    for conversation in input_data:
        # 为每个对话创建一个唯一ID
        conversation_id = str(uuid.uuid4())[:8]
        
        sharegpt_item = {
            "qid": f"conversation_{conversation_id}",
            "conversations": [],
            "images": []
        }
        
        # 提取图片URL（如果有的话）
        image_urls = []
        for message in conversation:
            if message["role"] == "user":
                for content_item in message.get("content", []):
                    if content_item.get("type") == "image_url":
                        image_url = content_item.get("image_url", {}).get("url", "")
                        if image_url and image_url not in image_urls:
                            image_urls.append(image_url)
        
        # 添加图片到images字段
        sharegpt_item["images"] = image_urls
        
        # 转换对话内容
        for message in conversation:
            role = message["role"]
            # 映射角色名称（根据ShareGPT格式要求）
            if role == "system":
                from_role = "system"
            elif role == "user":
                from_role = "human"
            elif role == "assistant":
                from_role = "gpt"
            else:
                from_role = role
            
            # 提取消息内容
            content_text = ""
            for content_item in message.get("content", []):
                if content_item.get("type") == "text":
                    content_text += content_item.get("text", "")
                elif content_item.get("type") == "image_url" or content_item.get("type") == "image":
                    # 用占位符表示图片
                    content_text += "<image>"
            
            # 添加到对话列表
            if content_text.strip():  # 确保不添加空消息
                sharegpt_item["conversations"].append({
                    "from": from_role,
                    "value": content_text
                })
        
        # 只添加有效对话（至少有一个消息）
        if sharegpt_item["conversations"]:
            sharegpt_data.append(sharegpt_item)
    
    return sharegpt_data

def main():
    # 读取输入的jsonl文件
    input_data_path = "share_data/datasets/web/gemini_test_format_conversation.jsonl"
    web_train_data = process_jsonl(input_data_path)
    
    # 转换为ShareGPT格式
    sharegpt_data = convert_to_sharegpt_format(web_train_data)
    
    # 写入输出json文件
    output_data_path = "share_data/datasets/web/webdata_sharegpt.json"
    write_json_file(sharegpt_data, output_data_path)
    
    print(f"转换完成! 共处理了 {len(web_train_data)} 条对话，生成了 {len(sharegpt_data)} 条ShareGPT格式数据。")

if __name__ == "__main__":
    main()