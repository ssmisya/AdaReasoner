

from tool_server.utils.utils import *
from tqdm import tqdm
from PIL import Image
from tool_server.tool_workers.tool_manager.base_manager import ToolManager

dataset_path = "/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/dataset.jsonl"
image_dir = "/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation"

dataset = process_jsonl(dataset_path)
tool_manager = ToolManager(tools=["Draw2DPath"])
for idx,item in enumerate(tqdm(dataset)):
    random_item = item["path_drawings"]["random"]
    image_path = random_item["image_path"]
    tool_input = random_item["input"]
    tool_parameters = tool_input["parameters"]
    tool_name = tool_input["name"]
    
    env_image_path = os.path.join(image_dir, item["image_path"])
    env_image = Image.open(env_image_path)
    env_image_base64 = pil_to_base64(env_image)
    tool_parameters["image"] = env_image_base64
    tool_res = tool_manager.call_tool(tool_name, tool_parameters)
    
    edited_image = base64_to_pil(tool_res["edited_image"])
    edited_image_path = os.path.join(image_dir, image_path)
    edited_image.save(edited_image_path)
    
