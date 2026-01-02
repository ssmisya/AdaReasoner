from tool_server.utils.utils import *
from tool_server.utils.prompts import tool_planning_model_prompt_one_tool_call
from tqdm import tqdm
import json, os
import numpy as np
import matplotlib
from PIL import Image
from io import BytesIO
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import deepcopy



tool_name_list = ["OCR","HighlightBox","GroundingDINO","DrawLine","LanguageModel","SegmentRegionAroundPoint","GetSubplotInfo","Crop","Point","DrawShape","MaskBox","GetBarInfo"]

def annotate_image(image_np, boxes, logits, phrases):
    """在图像上绘制边界框和标签"""
    # 设置matplotlib使用非交互模式
    matplotlib.use('Agg')
    
    # 创建一个新的图形和轴
    fig, ax = plt.subplots(1)
    ax.imshow(image_np)
    
    # 获取图像尺寸
    h, w, _ = image_np.shape
    
    # 为不同类别设置不同颜色
    unique_phrases = list(set(phrases))
    colors = plt.cm.hsv(np.linspace(0, 1, len(unique_phrases) if len(unique_phrases) > 0 else 1))
    color_map = {phrase: colors[i % len(colors)] for i, phrase in enumerate(unique_phrases)}
    
    # 绘制每个检测框
    for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
        # 获取边界框坐标
        x_min, y_min, x_max, y_max = box
        # 将归一化坐标转换为像素坐标
        x_min_px = x_min if x_min >= 1 else int(x_min * w)
        y_min_px = y_min if y_min >= 1 else int(y_min * h)
        x_max_px = x_max if x_max >= 1 else int(x_max * w)
        y_max_px = y_max if y_max >= 1 else int(y_max * h)
        # x_min_px, y_min_px = int(x_min * w), int(y_min * h)
        # x_max_px, y_max_px = int(x_max * w), int(y_max * h)
        
        # 计算宽度和高度
        width = x_max_px - x_min_px
        height = y_max_px - y_min_px
        
        # 获取当前类别的颜色
        color = color_map.get(phrase, 'red')
        
        # 创建一个矩形
        rect = patches.Rectangle(
            (x_min_px, y_min_px), width, height, 
            linewidth=2, edgecolor=color, facecolor='none'
        )
        
        # 添加矩形到轴
        ax.add_patch(rect)
        
        # 添加标签文本
        confidence = f"{logit:.2f}"
        label = f"{phrase}: {confidence}"
        plt.text(
            x_min_px, y_min_px - 5, label, 
            bbox=dict(facecolor=color, alpha=0.5),
            fontsize=8, color='white'
        )
    
    # 移除轴标签
    plt.axis('off')
    
    # 将图形保存到内存缓冲区
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    # 从缓冲区加载图像
    annotated_image = Image.open(buf).convert('RGB')
    return annotated_image

def supply_groundingdino_to_one_conv(conv, original_picture_addr, image_save_path):
    original_picture = Image.open(original_picture_addr).convert('RGB')
    original_picture_arr =  np.array(original_picture)
    if "content" not in conv:
        return
    contents = conv["content"]
    new_conv = deepcopy(conv)
    
    for content in contents:
        try:
            if "text" in content:
                content_text = content["text"]
                tool_response = json.loads(content_text)
                detections = tool_response["detections"]
                labels = []
                new_boxes = []
                logits = []
                for detection in detections:
                    boxes = detection["bbox"]
                    new_box = [boxes["x_min"], boxes["y_min"], boxes["x_max"], boxes["y_max"]]
                    new_boxes.append(new_box)
                    labels.append(detection["label"])
                    logits.append(detection["confidence"])
                new_image = annotate_image(original_picture_arr, new_boxes, logits, labels)
                new_image.save(image_save_path)
                new_conv = dict(role="user",content=[dict(type="text",text=f"{tool_response}"),dict(type="image",image=image_save_path)])
                return new_conv
        except:
            return None
    

def supply_with_gd(
    input_data, 
    image_dir_path="/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/PixelReasoner-SFT-Data/images",
    output_dir_path = "/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/PixelReasoner-SFT-Data/gd_images"
):
    qualified_data = []
    for item in tqdm(input_data):
        conversations = item["response_messages"]
        origin_qid = item["origin_qid"]
        qid = item["qid"]
        new_conversations = []
        next_round_operate = None
        terminate = False
        for conv in conversations:
            # print(f"Processing conversation: {conv}")
            if "content" not in conv or "role" not in conv:
                terminate = True
                break
            contents = conv["content"]
            
            if next_round_operate:
                new_conv = supply_groundingdino_to_one_conv(
                    conv, 
                    original_picture_addr=next_round_operate["original_picture_addr"],
                    image_save_path=next_round_operate["image_save_path"]
                )
                if new_conv is None:
                    terminate = True
                    break
                else:
                    new_conversations.append(new_conv)
                    next_round_operate = None
                    continue
            else:
                new_conversations.append(conv)
                for content in contents:
                    if isinstance(content, str) or "text" in content:
                        if isinstance(content, str):
                            content_text = content
                        elif "text" in content:
                            content_text = content["text"]
                            if "<tool_call>" in content_text:
                                tool_json = content_text.split("<tool_call>")[-1].split("</tool_call>")[0]
                                
                                try:
                                    tool_content = json.loads(tool_json)
                                    tool_name = tool_content["name"]
                                    if tool_name == "GroundingDINO":
                                        next_round_operate = {
                                            "original_picture_addr" : f"{image_dir_path}/{origin_qid}-0.jpg",
                                            "image_save_path": f"{output_dir_path}/{origin_qid}-gd.jpg",
                                        }
                                    assert tool_name in tool_name_list, f"Tool name {tool_name} not in predefined list."
                                    
                                except:
                                    tool_name = "invalid"
                                    terminate = True
                                    break
                        
               
            if terminate:
                break
            
        if terminate:
            continue
        item["response_messages"] = new_conversations
        qualified_data.append(item)
    return qualified_data
        
        
            

if __name__ == "__main__":
    input_data = "/mnt/petrelfs/songmingyang/code/reasoning/data_construction/PixelReasoner/data/pixelreasonersft_groundingcrop_formatted.jsonl"

    input_data = process_jsonl(input_data)
    output_path = "/mnt/petrelfs/songmingyang/code/reasoning/data_construction/PixelReasoner/data/supply_grounding_data/pixelreasonersft_groundingcrop_sground.jsonl"

    pixel_reasoner_basedir = "/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/PixelReasoner-SFT-Data"
    
    adjusted_data = supply_with_gd(input_data)
    write_jsonl(adjusted_data, output_path)

    
    
    