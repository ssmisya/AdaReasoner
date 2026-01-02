from datasets import Features, Value, Sequence,load_dataset
from tool_server.utils.utils import *
from tool_server.utils.prompts import tool_planning_model_prompt_one_tool_call
from tqdm import tqdm
import json, os
import random
random.seed(42)
from PIL import Image



setup_proxy()
# Set seed for reproducibility
random.seed(42)

image_dir_path = "/mnt/petrelfs/songmingyang/songmingyang/data/mm/imgs"

vstar_gqa = "/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/v_star/seal_vqa_data/GQA_data.json"
vstar_llava_focus = "/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/v_star/seal_vqa_data/llava_focus_data.json"
vstar_llava_instruct = "/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/v_star/seal_vqa_data/llava_instruct_data.json"
vstar_negative_data = "/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/v_star/seal_vqa_data/negative_data.json"
vstar_spatial_data = "/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/v_star/seal_vqa_data/spatial_relation_data.json"
vstar_vaw_spatial_data = "/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/v_star/seal_vqa_data/vaw_attribute_data.json"

vstar_data_paths = [vstar_gqa, vstar_llava_focus, vstar_llava_instruct, vstar_negative_data, vstar_spatial_data, vstar_vaw_spatial_data]
vstar_path_dict = dict(
    vstar_gqa = "/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/v_star/seal_vqa_data/GQA_data.json",
    vstar_llava_focus = "/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/v_star/seal_vqa_data/llava_focus_data.json",
    vstar_llava_instruct = "/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/v_star/seal_vqa_data/llava_instruct_data.json",
    vstar_negative_data = "/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/v_star/seal_vqa_data/negative_data.json",
    vstar_spatial_data = "/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/v_star/seal_vqa_data/spatial_relation_data.json",
    vstar_vaw_spatial_data = "/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/v_star/seal_vqa_data/vaw_attribute_data.json",
)


def vstar_spatial_get_data(data):
    new_data = []
    for idx, item in enumerate(tqdm(data)):
        image_path = os.path.join(image_dir_path, item["image"].replace("coco2017","coco").replace("coco2014","coco"))
        image = Image.open(image_path).convert("RGB")
        images = [image]
        conversations = item["conversations"]
        question = conversations[0]["value"].split(" <object>.\n")[-1].strip()
        answer = conversations[1]["value"].strip()
        
        problem = f"<image>{question}"
        new_item = {
            "images": images,
            "problem": problem,
            "answer": answer,
            "metadata": item
        }
        new_data.append(new_item)
    return new_data

def vstar_gqa_get_data(input_data):
    new_data = []
    for idx, item in enumerate(tqdm(input_data)):
        image_path = os.path.join(image_dir_path, item["image"].replace("coco2017","coco").replace("coco2014","coco"))
        image = Image.open(image_path).convert("RGB")
        images = [image]
        problem = f"<image>{item['question']}"
        answer = item["answer"]
        new_item = {
            "images": images,
            "problem": problem,
            "answer": answer,
            "metadata": item
        }
        new_data.append(new_item)
    return new_data

train_num = 5000
val_num = 330

data_length_dict = {}
vstar_data_dict = {}
for k,v in vstar_path_dict.items():
    data = load_json_file(v)
    vstar_data_dict[k] = data
    data_length_dict[k] = len(data)
    
vstar_spatial_data = vstar_data_dict["vstar_spatial_data"]
random.shuffle(vstar_spatial_data)
vstar_spatial_data_train = vstar_spatial_data[:train_num]  # Limit to 1000 items for testing
vstar_spatial_data_val = vstar_spatial_data[train_num:train_num + val_num]  # Limit to 100 items for validation

vstar_spatial_data_train_res = vstar_spatial_get_data(vstar_spatial_data_train)
vstar_spatial_data_val_res = vstar_spatial_get_data(vstar_spatial_data_val)


vstar_vaw_spatial_data = vstar_data_dict["vstar_vaw_spatial_data"]
random.shuffle(vstar_vaw_spatial_data)
vstar_vaw_spatial_data_train = vstar_vaw_spatial_data[:train_num]  # Limit to 1000 items for testing
vstar_vaw_spatial_data_val = vstar_vaw_spatial_data[train_num:train_num + val_num]  # Limit to 100 items for validation
vstar_vaw_spatial_data_train_res = vstar_spatial_get_data(vstar_vaw_spatial_data_train)
vstar_vaw_spatial_data_val_res = vstar_spatial_get_data(vstar_vaw_spatial_data_val)

vstar_gqa = vstar_data_dict["vstar_gqa"]
random.shuffle(vstar_gqa)
vstar_gqa_train = vstar_gqa[:train_num]  # Limit to 1000 items for testing
vstar_gqa_val = vstar_gqa[train_num:train_num + val_num]  # Limit to 100 items for validation
vstar_gqa_train_res = vstar_gqa_get_data(vstar_gqa_train)
vstar_gqa_val_res = vstar_gqa_get_data(vstar_gqa_val)


chart_train ="/mnt/petrelfs/songmingyang/songmingyang/data/vl_reasoning/tool_dataset/toolrl_v1_gemmareachqa"
chart_train_set = load_dataset(chart_train)

def get_chart_data(chart_data):
    new_data = []
    for item in tqdm(chart_data):
        new_item = {
            "images": item["images"],
            "problem": item["problem"],
            "answer": item["answer"],
            "metadata": {}
        }
        new_data.append(new_item)
    return new_data

chart_trainset = chart_train_set["train"]
chart_valset = chart_train_set["validation"]

chart_trainset_res = get_chart_data(chart_trainset)
chart_valset_res = get_chart_data(chart_valset)




from datasets import Dataset, DatasetDict, Features, Value, Image, Sequence
# Create new lists by combining all train and validation datasets
all_train_data = []
all_train_data.extend(vstar_gqa_train_res)
all_train_data.extend(vstar_spatial_data_train_res)
all_train_data.extend(vstar_vaw_spatial_data_train_res)
all_train_data.extend(chart_trainset_res)


all_val_data = []
all_val_data.extend(vstar_gqa_val_res)
all_val_data.extend(vstar_spatial_data_val_res)
all_val_data.extend(vstar_vaw_spatial_data_val_res)
all_val_data.extend(chart_valset_res)

# Shuffle the data
random.shuffle(all_train_data)
random.shuffle(all_val_data)

print(f"Total train examples: {len(all_train_data)}")
print(f"Total validation examples: {len(all_val_data)}")

# Convert to datasets format
train_dataset = Dataset.from_list(all_train_data, )
val_dataset = Dataset.from_list(all_val_data,)

train_dataset = train_dataset.cast_column("images", Sequence(Image()))
val_dataset = val_dataset.cast_column("images", Sequence(Image()))

# Create dataset dictionary
combined_dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

print(combined_dataset)

# Define the dataset name for HuggingFace
dataset_name = "tool_rlset_v1"

# Push to the Hugging Face Hub
combined_dataset.push_to_hub(dataset_name, token="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")  # Replace with your actual token

print(f"Dataset uploaded to HuggingFace Hub as {dataset_name}")

a = load_dataset("hitsmy/tool_rlset_v1")
