# base_manager_randomize.py
import os
import requests
import random
import string
import re
import json
from typing import Dict, List, Any, Optional, Tuple, Set

from ..offline_workers import get_tool_generate_fn, get_available_tools, get_all_tool_instructions
from ..offline_workers import get_tool_instruction as get_offline_tool_instruction
from tool_server.utils.utils import load_json_file, load_image, base64_to_pil
from tool_server.utils.server_utils import build_logger
from tool_server.utils.prompts import (
    one_tool_call_wo_toollist, 
    tool_planning_model_prompt_no_tool_call,
    multi_tool_call_wo_toollist,
    tool_desc_dict
)
from contextlib import contextmanager
import signal
import time

logger = build_logger("tool_manager")

class TimeoutException(Exception): 
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException("chat() timed out.")

signal.signal(signal.SIGALRM, _timeout_handler)


# 工具描述库
TOOL_DESCRIPTIONS = {
    "AStarWithPixelCoordinate": [
        "Find the shortest path from start to goal while avoiding obstacles using A* algorithm",
        "Navigate from point A to B avoiding obstacles with A* pathfinding algorithm",
        "Calculate optimal path between two points while avoiding specified obstacle areas",
        "Use A* search algorithm to determine the shortest route between coordinates",
        "Path planning tool that finds efficient routes around pixel-coordinate obstacles",
        "Compute the most efficient navigation path between start and goal coordinates using A* search",
        "Determine the minimal-cost route from origin to destination given pixel obstacles",
        "Perform pathfinding between two pixel coordinates while bypassing blocked areas",
        "Search for a traversable route using the A-star algorithm avoiding impassable zones",
        "Plan an optimal movement trajectory in pixel space from start point to goal while avoiding obstacles",
        "Generate the shortest possible movement sequence (U,D,L,R) to reach the goal avoiding restricted pixels"
    ],
    "Draw2DPath": [
        "Draw a path on an image following a sequence of directional commands",
        "Create a visual path on image using directional instructions",
        "Visualize a route on an image using movement directions",
        "Render a line path on image based on directional sequence",
        "Mark a trail on image following specified movement pattern",
        "Plot a directional trajectory on the given image canvas",
        "Illustrate motion steps over an image according to direction sequence",
        "Generate a continuous line path following given movement symbols (u,d,l,r)",
        "Paint a visual representation of step-by-step path traversal on an image",
        "Overlay directional motion trace on the target image surface",
        "Depict a path using movement commands to connect points along a defined route"
    ],
    "DetectBlackArea": [
        "Detect pure black areas in an image and return their bounding boxes, prioritizing rectangular regions",
        "Find dark regions in an image and identify their rectangular boundaries",
        "Locate and measure black zones in the provided image",
        "Identify black areas and return their coordinate boundaries",
        "Scan image for dark patches and output their bounding boxes",
        "Analyze an image to detect black or near-black regions and extract their bounding boxes",
        "Find continuous low-brightness areas and represent them as rectangular boxes",
        "Search the image for black pixel clusters exceeding minimum area size",
        "Detect regions darker than the threshold and return their coordinates",
        "Highlight areas that qualify as dark based on intensity threshold and shape similarity",
        "Locate black rectangles or blobs within the input image and output their bounding coordinates"
    ],
    "InsertImage": [
        "Insert an image into a base image at a specified position defined by a bounding box",
        "Embed one image into another at defined coordinates",
        "Place an image inside another at a specific location",
        "Overlay an image onto a base image at designated position",
        "Integrate secondary image into primary one at specified coordinates",
        "Merge two images by inserting the smaller one into the defined region of the base",
        "Composite a foreground image onto a background image using bounding box coordinates",
        "Add an image element into a target canvas at the provided position",
        "Combine two images by placing one inside another according to defined coordinates",
        "Superimpose an image over another in the given pixel rectangle",
        "Paste an image snippet into a base frame at the specified coordinates, with optional resizing"
    ],
    "Point": [
        "Identify a point in the image based on a natural language description",
        "Locate a specific point on image using descriptive text",
        "Find coordinates of described feature in the image",
        "Determine pixel location based on textual description",
        "Mark exact position matching the provided description",
        "Pinpoint a visually described spot within the image frame",
        "Interpret the natural language phrase to locate a corresponding visual point",
        "Translate a textual reference (e.g., 'the red door handle') into coordinates",
        "Detect the target location in the image that matches the description",
        "Recognize and return the pixel position referred to in text",
        "Identify the visual feature mentioned and provide its coordinates"
    ],
    "Crop": [
        "Crop an image using specified bounding box coordinates",
        "Extract a rectangular region from image using coordinates",
        "Cut out a portion of image based on boundary values",
        "Create a new image from a defined section of original",
        "Isolate rectangular area from image using coordinate bounds",
        "Trim the image to the area defined by given coordinates",
        "Generate a cropped sub-image using specified pixel boundaries",
        "Extract the part of image enclosed by the bounding box",
        "Produce a smaller image focusing on the defined coordinates",
        "Remove surrounding regions to retain only the target area",
        "Perform image clipping based on coordinate range [x_min, y_min, x_max, y_max]"
    ],
    "OCR": [
        "Extracts and localizes text from the given image using OCR",
        "Recognize and locate text elements in provided image",
        "Detect text content and positions within image",
        "Convert text in image to machine-readable format with positions",
        "Find and extract textual information from visual content",
        "Perform optical character recognition to extract text and bounding boxes",
        "Read visible text from image and return recognized strings with coordinates",
        "Analyze the image to identify all readable words and their locations",
        "Recognize textual regions and output detected text with confidence scores",
        "Scan the input image to extract printed or handwritten text information",
        "Detect and digitize textual content appearing within the image area"
    ]
}

# 参数描述库
TOOL_PARAM_DESCRIPTIONS = {
    "Point": {
        "image": [
            "The identifier of the image to edit",
            "The image to analyze and locate a point on",
            "Source image for pinpointing the described target",
            "The reference image where the point should be identified",
            "Input image that contains the region of interest",
            "The visual source used for point extraction",
            "The picture in which the point will be detected",
            "Target image context for locating the mentioned feature",
            "Image input that the model should interpret to find the specified point",
            "The canvas image used for identifying the described area",
            "The ID of the image to operate on",
            "Reference to the target picture for point identification"
        ],
        "description": [
            "A natural language description of the point of interest",
            "Textual hint about the location to identify",
            "Description specifying which part of the image to focus on",
            "Verbal description guiding the point selection process",
            "Human-language explanation of the target location",
            "Plain-text cue indicating the object or region to click",
            "Instruction describing what visual element to point at",
            "Language-based specification of the desired focus point",
            "Narrative expression of the position to identify",
            "Short phrase describing the visual target (e.g., 'the dog's nose')",
            "A textual guide to the point's location",
            "Tell me what to look for in words"
        ]
    },
    "Draw2DPath": {
        "image": [
            "The image to draw on (image identifier)",
            "Target image for visualizing the path",
            "Canvas where the path will be painted",
            "Base image for trajectory rendering",
            "The background image to overlay the path onto",
            "Output image that displays the drawn path",
            "Image frame to annotate with the generated path",
            "Visual surface where movement lines will appear",
            "The destination image for rendering directional strokes",
            "Identifier for the image that will be drawn upon"
        ],
        "start_point": [
            "Starting point coordinates [x, y]",
            "Initial coordinates marking where the path begins",
            "Origin point for drawing the trajectory",
            "Starting pixel position for rendering",
            "Beginning coordinates of the route",
            "Initial [x,y] from which the path expands",
            "Reference coordinate where motion starts",
            "Base position for the first movement step",
            "Path initiation coordinates in pixel format",
            "The [x, y] location where the drawing should commence"
        ],
        "directions": [
            "Direction sequence composed of 'u','d','l','r' commands",
            "Series of movement symbols representing path steps",
            "String defining ordered path movements (u,d,l,r)",
            "Control string describing direction flow of the route",
            "Sequence of direction codes guiding the drawing",
            "List of motion instructions used to build the trajectory",
            "Encoded path pattern using directional characters",
            "Ordered sequence of navigation commands for path generation",
            "Compact route representation using step directions",
            "A string of moves like 'uuddlr' or 'U,U,D,D,L,R'"
        ],
        "step": [
            "Step size in pixels for each directional move",
            "Distance traversed per direction command",
            "Pixel displacement for every movement instruction",
            "Incremental travel length between drawn points",
            "Magnitude of motion applied for each directional token",
            "Unit distance representing one move in the path",
            "Stride length in pixels per navigation step",
            "Granularity of movement when constructing the line",
            "How many pixels to move for each step in the directions string"
        ],
        "pixel_coordinate": [
            "If true, start_point uses pixel coordinates; if false, grid coordinates",
            "Boolean switch for coordinate system selection",
            "Flag determining whether coordinates are pixel-based or grid-based",
            "True if absolute pixel coordinates, false if grid indices",
            "Indicator of coordinate interpretation mode",
            "Specifies coordinate reference system for the start point",
            "Determines whether position is defined in pixel space or logical grid",
            "Is the start_point specified in pixels (true) or grid units (false)?"
        ],
        "line_width": [
            "Width of the line used to draw the path",
            "Pixel thickness of the rendered stroke",
            "Line size for trajectory visualization",
            "Drawing stroke width (in pixels)",
            "Thickness of the visualized movement trace",
            "Parameter controlling how bold the path appears",
            "Visual width of the path line in output image",
            "The thickness of the drawn line in pixels"
        ],
        "line_color": [
            "Color used for the drawn line",
            "Visual color representing the trajectory",
            "Stroke color value for rendering",
            "Drawing color of the path line",
            "Display color used when visualizing movement",
            "Hue specification for the path overlay",
            "Chosen color applied to the route display",
            "The color of the stroke, e.g., 'blue' or '#0000FF'"
        ]
    },
    "AStarWithPixelCoordinate": {
        "start": [
            "Starting point coordinates [x, y] in pixels",
            "Initial pixel location where the search begins",
            "Origin point of the A* pathfinding process",
            "Starting node expressed as pixel coordinates",
            "Beginning [x,y] of the planned route",
            "Coordinate pair for the initial search position",
            "Pixel-based start location for shortest path computation",
            "The origin [x, y] coordinates for the pathfinder"
        ],
        "goal": [
            "Goal point coordinates [x, y] in pixels",
            "Target destination pixel position for pathfinding",
            "End coordinates representing the destination node",
            "Terminal location in pixel coordinate system",
            "Final [x,y] coordinate the algorithm must reach",
            "Destination point for A* traversal",
            "Objective coordinate pair representing the goal",
            "The destination [x, y] coordinates the path should lead to"
        ],
        "obstacles": [
            "Array of obstacle coordinates [[x1, y1], [x2, y2], ...] in pixels",
            "List of forbidden points representing barriers",
            "Coordinate set marking impassable zones",
            "Collection of pixel locations that block traversal",
            "Spatial positions that cannot be crossed",
            "Matrix or list of obstacle positions to avoid",
            "Locations treated as blocked cells in pathfinding map",
            "A list of impassable [x, y] pixel coordinates to navigate around"
        ]
    },
    "DetectBlackArea": {
        "image": [
            "The image to analyze (image identifier)",
            "Input image for dark region detection",
            "Image source to locate black or near-black areas",
            "Target image for analyzing pixel intensity",
            "Visual input to process for darkness regions",
            "Picture to scan for low-brightness regions",
            "Identifier of the image to be scanned for dark areas"
        ],
        "threshold": [
            "Brightness cutoff value (0–255) below which pixels are considered black",
            "Intensity threshold to classify dark pixels",
            "Brightness level used as blackness boundary",
            "Grayscale threshold defining dark region segmentation",
            "Parameter controlling how dark a pixel must be to count as black",
            "The luminance value (0-255) to use as the cutoff for what is considered 'black'"
        ],
        "min_area": [
            "Minimum number of pixels for a valid detected area",
            "Smallest contiguous dark region to be recognized",
            "Lower bound of area size to consider in detection",
            "Threshold for region size filtering during detection",
            "Minimum pixel cluster size defining a valid black patch",
            "Ignore black regions that are smaller than this pixel count"
        ]
    },
    "InsertImage": {
        "base_image": [
            "Base image to insert into (image identifier)",
            "Destination image where content will be placed",
            "Target background image for embedding operation",
            "Canvas image receiving the inserted content",
            "Underlying image acting as host for insertion",
            "The ID of the background image"
        ],
        "image_to_insert": [
            "Image content to insert (image identifier)",
            "Overlay image being added into the base",
            "Visual element to be merged with base image",
            "Sub-image intended for placement",
            "The source image for compositing operation",
            "The ID of the foreground image to be placed"
        ],
        "coordinates": [
            "Bounding box coordinates '[x_min, y_min, x_max, y_max]' defining placement area",
            "Target rectangular region for positioning insertion",
            "Placement coordinates specifying where to embed the image",
            "Numeric region defining insertion bounds",
            "Pixel area where the secondary image should appear",
            "The target rectangle [xmin, ymin, xmax, ymax] for the insertion"
        ],
        "resize": [
            "Whether to resize the inserted image to fit target bounds",
            "Boolean indicating if image should be scaled proportionally",
            "Option to automatically adjust image size for bounding box",
            "Flag controlling whether to stretch or fit the content",
            "True if resizing is enabled, false to keep original dimensions",
            "Should the inserted image be stretched to fit the coordinates?"
        ]
    },
    "Crop": {
        "image": [
            "Identifier of the image to crop",
            "Source image for cropping operation",
            "Input picture from which a region is extracted",
            "Original image providing the crop target area",
            "Target image for subregion selection",
            "The ID of the image you want to cut a piece from"
        ],
        "coordinates": [
            "Bounding box coordinates '[x_min, y_min, x_max, y_max]' for cropping",
            "Region limits defining the extraction area",
            "Rectangle specifying which portion to cut out",
            "Coordinates delimiting the cropping boundaries",
            "Numeric boundaries identifying the crop box",
            "The rectangular area to cut out, defined by its top-left and bottom-right corners"
        ]
    },
    "OCR": {
        "image": [
            "Identifier of the image to perform OCR on",
            "Source image containing text to extract",
            "Target image for optical character recognition",
            "Visual input where text elements appear",
            "Image to be processed for text detection and recognition",
            "Picture containing words or symbols to read",
            "The ID of the image from which to read text"
        ]
    }
}


def generate_random_name(length=6, prefix=""):
    """生成随机名称"""
    chars = string.ascii_lowercase + string.digits
    random_str = ''.join(random.choices(chars, k=length))
    return f"{prefix}{random_str}"


class ToolManager(object):
    def __init__(self, controller_url_location=None, tools=None, randomize=False, random_seed=None):
        """
        初始化工具管理器
        
        Args:
            controller_url_location (str, optional): 控制器URL位置
            tools (list, optional): 指定要初始化的工具列表，不指定则初始化全部工具
            randomize (bool): 是否随机化工具名称和描述
            random_seed (int, optional): 随机种子，确保每次初始化时的随机化结果一致
        """
        self.controller_url_location = controller_url_location
        self.tools = tools  # 保存用户指定的工具列表
        self.randomize = randomize
        self.headers = {"User-Agent": "LLaVA-Plus Client"}
        
        # 设置随机种子（如果提供）
        if random_seed is not None:
            random.seed(random_seed)
        
        # 初始化映射表
        self.randomized_to_original = {}  # 随机名称 -> 原始名称
        self.original_to_randomized = {}  # 原始名称 -> 随机名称
        self.tool_desc_map = {}  # 工具名称 -> 随机化的描述
        self.param_desc_map = {}  # 参数名 -> {工具名: 随机化的描述}
        
        # 缓存原始指令和随机化后的指令
        self.original_instructions = {}  # 原始工具名 -> 原始指令
        self.randomized_instructions = {}  # 随机化后的工具名 -> 随机化后的指令
        
        self.image_params = ["image","base_image","image_to_insert"]
        self.init_offline_tools(tools)
        self.init_online_tools(self.controller_url_location)
        self.init_online_tool_addr_dict()
        self.available_tools = self.available_online_tools + self.available_offline_tools
        
        # 初始化时一次性加载所有工具指令
        self._load_all_instructions()
        
        # 如果启用随机化，生成映射表并随机化指令
        if self.randomize:
            self._generate_randomization_maps()
            self._randomize_all_instructions()
        
        print("available_tools", self.available_tools)
        
        if self.tools is None:
            # 如果未指定工具，检查所有常用工具
            required_tools = ["GroundingDINO", "OCR", "SegmentRegionAroundPoint", "Point", 
                             "Crop", "DrawLine", "DrawShape", "HighlightBox", "MaskBox", 
                             "GetSubplotInfo", "GetBarInfo"]
        else:
            # 仅检查指定的常用工具
            required_tools = [tool for tool in self.tools]
                
        miss_tool = [tool for tool in required_tools if tool not in self.available_tools]
        if len(miss_tool) == 0:
            logger.info("All required online tools are prepared successfully")
        else:
            logger.info(f"Not all required online tools are prepared successfully, missing: {miss_tool}")     
            
        logger.info(f"ToolManager is initialized. Randomize mode: {self.randomize}")
        
        
    
    def get_tool_real_name(self,tool_name):
        """获取工具的原始名称"""
        if self.randomize:
            return self.randomized_to_original.get(tool_name, tool_name)
        else:
            return tool_name
    
    def _load_all_instructions(self):
        """初始化时一次性加载所有工具的原始指令"""
        logger.info("Loading all tool instructions...")
        
        for tool_name in self.available_tools:
            instruction = None
            
            # 获取离线工具指令
            if tool_name in self.available_offline_tools:
                instruction = get_offline_tool_instruction(tool_name)
                    
            # 获取在线工具指令
            elif tool_name in self.available_online_tools:
                # 先尝试从在线工具获取指令
                instruction = self.get_online_tool_instruction(tool_name)
                # 如果无法从在线工具获取，尝试从预定义字典获取
                if not instruction and tool_name in tool_desc_dict:
                    instruction = tool_desc_dict[tool_name]
            
            # 缓存原始指令
            if instruction:
                self.original_instructions[tool_name] = instruction
                logger.debug(f"Loaded instruction for {tool_name}")
            else:
                logger.warning(f"No instruction found for tool {tool_name}")
        
        logger.info(f"Loaded {len(self.original_instructions)} tool instructions")
    
    def _extract_all_params_from_instruction(self, instruction: str) -> Set[str]:
        """从工具指令中提取所有参数名称（支持多种格式）"""
        params = set()
        
        try:
            # 尝试作为JSON解析
            if isinstance(instruction, str):
                # 处理单引号格式
                if "{'type':" in instruction or "{'function':" in instruction:
                    # Python dict 格式，使用 eval
                    try:
                        inst_dict = eval(instruction)
                    except:
                        inst_dict = json.loads(instruction)
                else:
                    inst_dict = json.loads(instruction)
            else:
                inst_dict = instruction
            
            # 从 function.parameters.properties 中提取参数
            if "function" in inst_dict:
                func = inst_dict["function"]
                if "parameters" in func and "properties" in func["parameters"]:
                    params.update(func["parameters"]["properties"].keys())
                    
        except Exception as e:
            # 如果JSON解析失败，使用正则表达式
            logger.debug(f"Failed to parse instruction as JSON, using regex: {e}")
            
            # 匹配 "参数名": { 或 '参数名': { 格式
            param_patterns = [
                r'"([^"]+)":\s*{[^}]*"type"',  # JSON格式
                r"'([^']+)':\s*{[^}]*'type'",  # Python dict格式
            ]
            
            for pattern in param_patterns:
                matches = re.finditer(pattern, instruction)
                for match in matches:
                    param_name = match.group(1)
                    # 过滤掉非参数的键
                    if param_name not in ['properties', 'required', 'type']:
                        params.add(param_name)
        
        return params
    
    def _generate_randomization_maps(self):
        """生成随机化映射表"""
        logger.info("Generating randomization maps...")
        
        # Step 1: 收集所有工具及其参数
        tool_params_map = {}  # {tool_name: set(params)}
        
        for tool_name, instruction in self.original_instructions.items():
            # 提取参数
            params = self._extract_all_params_from_instruction(instruction)
            tool_params_map[tool_name] = params
            logger.debug(f"Tool {tool_name} has parameters: {params}")
        
        # Step 2: 为工具名称生成随机映射
        for tool_name in self.available_tools:
            randomized_name = generate_random_name(prefix="")
            self.original_to_randomized[tool_name] = randomized_name
            self.randomized_to_original[randomized_name] = tool_name
            
            # 为工具描述生成随机映射
            if tool_name in TOOL_DESCRIPTIONS and TOOL_DESCRIPTIONS[tool_name]:
                self.tool_desc_map[tool_name] = random.choice(TOOL_DESCRIPTIONS[tool_name])
            
            logger.debug(f"Tool mapping: {tool_name} -> {randomized_name}")
        
        # Step 3: 收集所有唯一的参数名
        all_params = set()
        for params in tool_params_map.values():
            all_params.update(params)
        
        logger.info(f"Found {len(all_params)} unique parameters across all tools: {all_params}")
        
        # Step 4: 为每个参数生成随机映射
        for param_name in all_params:
            randomized_param = generate_random_name(prefix="")
            self.original_to_randomized[param_name] = randomized_param
            self.randomized_to_original[randomized_param] = param_name
            
            logger.debug(f"Param mapping: {param_name} -> {randomized_param}")
            
            # 为参数描述生成随机映射（针对每个使用该参数的工具）
            for tool_name, params in tool_params_map.items():
                if param_name in params:
                    # 检查是否有该工具-参数对的描述
                    if (tool_name in TOOL_PARAM_DESCRIPTIONS and 
                        param_name in TOOL_PARAM_DESCRIPTIONS[tool_name] and 
                        TOOL_PARAM_DESCRIPTIONS[tool_name][param_name]):
                        
                        if param_name not in self.param_desc_map:
                            self.param_desc_map[param_name] = {}
                        
                        self.param_desc_map[param_name][tool_name] = random.choice(
                            TOOL_PARAM_DESCRIPTIONS[tool_name][param_name]
                        )
        
        renew_image_params = []
        for param in self.image_params:
            if param in self.original_to_randomized:
                renew_image_params.append(self.original_to_randomized[param])
        self.image_params = renew_image_params
        
        logger.info(f"Randomization complete:")
        logger.info(f"  - {len(self.available_tools)} tools randomized")
        logger.info(f"  - {len(all_params)} parameters randomized")
        logger.info(f"  - {len(self.tool_desc_map)} tool descriptions randomized")
        logger.info(f"  - {sum(len(v) for v in self.param_desc_map.values())} parameter descriptions randomized")
    
    def _randomize_single_instruction(self, instruction: str, tool_name: str) -> str:
        """
        随机化单个工具的指令
        
        Args:
            instruction: 原始指令（可能是JSON字符串或Python dict字符串）
            tool_name: 工具名称（原始）
            
        Returns:
            str: 随机化后的指令（保持原格式）
        """
        if not instruction:
            return instruction
        
        try:
            # 检测是Python dict还是JSON格式
            is_python_dict = "{'type':" in instruction or "{'function':" in instruction
            
            # 解析指令
            if isinstance(instruction, str):
                if is_python_dict:
                    inst_dict = eval(instruction)
                else:
                    inst_dict = json.loads(instruction)
            else:
                inst_dict = instruction
            
            # 替换工具名称
            if "function" in inst_dict and "name" in inst_dict["function"]:
                if tool_name in self.original_to_randomized:
                    inst_dict["function"]["name"] = self.original_to_randomized[tool_name]
            
            # 替换工具描述
            if "function" in inst_dict and "description" in inst_dict["function"]:
                if tool_name in self.tool_desc_map:
                    inst_dict["function"]["description"] = self.tool_desc_map[tool_name]
            
            # 替换参数名称和描述
            if "function" in inst_dict and "parameters" in inst_dict["function"]:
                params = inst_dict["function"]["parameters"]
                if "properties" in params:
                    new_properties = {}
                    for param_name, param_info in params["properties"].items():
                        # 获取新参数名
                        new_param_name = self.original_to_randomized.get(param_name, param_name)
                        
                        # 替换参数描述
                        if "description" in param_info:
                            if param_name in self.param_desc_map and tool_name in self.param_desc_map[param_name]:
                                param_info["description"] = self.param_desc_map[param_name][tool_name]
                        
                        new_properties[new_param_name] = param_info
                    
                    params["properties"] = new_properties
                    
                    # 更新required列表
                    if "required" in params:
                        params["required"] = [
                            self.original_to_randomized.get(p, p) for p in params["required"]
                        ]
            
            # 返回相同格式的字符串
            if is_python_dict:
                return str(inst_dict)
            else:
                return json.dumps(inst_dict)
            
        except Exception as e:
            logger.error(f"Failed to randomize instruction for {tool_name}: {e}")
            # 如果解析失败，使用正则替换作为后备方案
            randomized = instruction
            
            # 替换工具名称
            if tool_name in self.original_to_randomized:
                randomized = re.sub(
                    f'["\']name["\']\s*:\s*["\']({re.escape(tool_name)})["\']',
                    f'"name": "{self.original_to_randomized[tool_name]}"',
                    randomized
                )
            
            # 替换所有参数名称
            for orig_param, rand_param in self.original_to_randomized.items():
                if orig_param != tool_name:  # 跳过工具名
                    # 替换 "param": 或 'param': 格式
                    randomized = re.sub(
                        f'["\']({re.escape(orig_param)})["\']\s*:',
                        f'"{rand_param}":',
                        randomized
                    )
            
            return randomized
    
    def _randomize_all_instructions(self):
        """一次性随机化所有工具指令并缓存"""
        logger.info("Randomizing all tool instructions...")
        
        for tool_name, original_instruction in self.original_instructions.items():
            randomized_instruction = self._randomize_single_instruction(original_instruction, tool_name)
            
            # 使用随机化后的工具名作为key
            randomized_tool_name = self.original_to_randomized.get(tool_name, tool_name)
            self.randomized_instructions[randomized_tool_name] = randomized_instruction
            
            logger.debug(f"Randomized instruction for {tool_name} -> {randomized_tool_name}")
        
        logger.info(f"Randomized {len(self.randomized_instructions)} tool instructions")
    
    def init_offline_tools(self, tools=None):
        """
        初始化离线工具
        
        Args:
            tools (list, optional): 指定要初始化的工具列表，不指定则初始化全部工具
        """
        # 获取所有可用的离线工具
        all_available_tools = get_available_tools()
        
        # 如果指定了工具列表，只保留指定的工具
        if tools is not None:
            self.available_offline_tools = [tool for tool in all_available_tools if tool in tools]
        else:
            self.available_offline_tools = all_available_tools
            
        logger.info(f"Offline Tools: {self.available_offline_tools}")
    
    def init_online_tools(self, controller_url_location=None):
        """初始化在线工具"""
        self.available_online_tools = []
        if controller_url_location is None:
            current_file_path = os.path.dirname(os.path.abspath(__file__))
            self.controller_addr_location = f"{current_file_path}/../online_workers/controller_addr/controller_addr.json"
            logger.info("controller_addr is None, using default from controller_addr_location")
        else:
            self.controller_addr_location = controller_url_location
            logger.info(f"controller_addr exists, controller_url_location is {controller_url_location}")
        
        if os.path.exists(self.controller_addr_location):
            self.controller_addr = load_json_file(self.controller_addr_location)["controller_addr"]
        else:
            self.controller_addr = self.controller_addr_location

        with self.disable_proxy():
            if self.controller_addr is not None and isinstance(self.controller_addr, str):
                session = requests.Session()
                session.trust_env = False
                try:
                    ret = session.post(self.controller_addr + "/list_models", proxies={}, timeout=(600, 600))
                    online_tools = ret.json()["models"]
                    
                    # 如果指定了工具列表，只保留指定的在线工具
                    if self.tools is not None:
                        online_tools = [tool for tool in online_tools if tool in self.tools]
                        
                    logger.info(f"Online Tools: {online_tools}")
                    self.available_online_tools = online_tools
                except Exception as e:
                    logger.error(f"Failed to connect to controller: {e}")
                    self.available_online_tools = []
    
    def init_online_tool_addr_dict(self):
        """初始化在线工具地址字典"""
        self.online_tool_addr_dict = {}
        for model_name in self.available_online_tools:
            with self.disable_proxy():
                session = requests.Session()
                session.trust_env = False
                try:
                    ret = session.post(self.controller_addr + "/get_worker_address",
                                      json={"model": model_name}, proxies={})
                    worker_addr = ret.json()["address"]
                    if worker_addr == "":
                        logger.error(f"worker_addr for {model_name} is empty")
                        continue
                    self.online_tool_addr_dict[model_name] = worker_addr
                except Exception as e:
                    logger.error(f"Failed to get worker address for {model_name}: {e}")
    
    def get_online_tool_instruction(self, tool_name):
        """
        从在线工具获取指令说明
        
        Args:
            tool_name (str): 工具名称（原始名称）
            
        Returns:
            str: 工具的指令说明，如果获取失败则返回None
        """
        # 检查缓存
        self.online_tool_instructions = getattr(self, 'online_tool_instructions', {})
        
        # 如果已经缓存，直接返回
        if tool_name in self.online_tool_instructions:
            return self.online_tool_instructions[tool_name]
        
        # 如果工具地址字典中没有该工具，返回None
        if tool_name not in self.online_tool_addr_dict:
            logger.warning(f"Tool {tool_name} not found in online_tool_addr_dict")
            return None
            
        tool_worker_addr = self.online_tool_addr_dict[tool_name]
        try:
            with self.disable_proxy():
                session = requests.Session()
                session.trust_env = False
                ret = session.post(tool_worker_addr + "/tool_instruction", 
                                   headers=self.headers, proxies={})
                
                if ret.status_code == 200:
                    response_data = ret.json()
                    if response_data.get("status") == "success" and "tool_instruction" in response_data:
                        # 保存到缓存
                        instruction = response_data["tool_instruction"]
                        self.online_tool_instructions[tool_name] = instruction
                        return instruction
                        
            logger.warning(f"Failed to get instruction for online tool {tool_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting instruction for online tool {tool_name}: {e}")
            return None
    
    def get_tool_instructions(self, tools=None):
        """
        获取指定工具的指令说明（从缓存中获取，确保一致性）
        
        Args:
            tools (list, optional): 指定要获取的工具列表，不指定则获取所有已初始化的工具
            
        Returns:
            dict: 工具名称到instruction的映射（如果randomize=True，则返回随机化后的名称和指令）
        """
        instructions = {}
        
        # 确定要获取指令的工具列表（原始工具名）
        if tools is not None:
            tool_list = tools
        else:
            tool_list = self.available_offline_tools + self.available_online_tools
        
        if self.randomize:
            # 从随机化后的缓存中获取
            for tool_name in tool_list:
                randomized_tool_name = self.original_to_randomized.get(tool_name, tool_name)
                if randomized_tool_name in self.randomized_instructions:
                    instructions[randomized_tool_name] = self.randomized_instructions[randomized_tool_name]
                else:
                    logger.warning(f"Randomized instruction not found for {tool_name}")
        else:
            # 从原始缓存中获取
            for tool_name in tool_list:
                if tool_name in self.original_instructions:
                    instructions[tool_name] = self.original_instructions[tool_name]
                else:
                    logger.warning(f"Original instruction not found for {tool_name}")
        
        return instructions
    
    def get_tool_prompt(self, prompt_type="one_tool_call", tools=None):
        """
        获取带有工具说明的提示语（从缓存中获取，确保一致性）
        
        Args:
            prompt_type (str): 提示语类型，可选 "one_tool_call"、"no_tool_call" 或 "multi_tool_call"
            tools (list, optional): 指定要包含在提示语中的工具列表，不指定则包含所有已初始化的工具
            
        Returns:
            str: 带有工具说明的提示语（如果randomize=True，则包含随机化后的工具名称、参数名和描述）
        """
        # 从缓存获取工具指令（已经是随机化后的，如果启用了随机化）
        tool_instructions = self.get_tool_instructions(tools)
        
        # 将工具指令拼接成字符串
        tool_list_str = "\n".join([f"{desc}" for name, desc in tool_instructions.items()])
        
        # 根据prompt类型选择基础提示语
        if prompt_type == "one_tool_call":
            base_prompt = one_tool_call_wo_toollist
        elif prompt_type == "no_tool_call":
            return tool_planning_model_prompt_no_tool_call
        elif prompt_type == "multi_tool_call":
            base_prompt = multi_tool_call_wo_toollist
        else:
            raise ValueError(f"Unsupported prompt type: {prompt_type}")
        
        # 将工具列表插入到提示语中
        prompt = base_prompt.replace("{tool_list}", tool_list_str)
        
        return prompt
    
    def call_tool(self, tool_name, params):
        """
        调用工具并返回结果
        
        Args:
            tool_name: 工具名称（如果randomize=True，则为随机化后的名称）
            params: 工具参数（如果randomize=True，则参数名为随机化后的名称）
            
        Returns:
            dict: 工具执行结果
        """
        timeout_sec = 300  # timeout per attempt
        ret_message = {"text": f"Failed to call tool {tool_name} for unknown reason", "error_code": 1}
        
        # 如果启用了随机化，需要转换回原始名称
        if self.randomize:
            # 转换工具名称
            original_tool_name = self.randomized_to_original.get(tool_name, tool_name)
            
            # 转换参数名称
            original_params = {}
            for param_name, param_value in params.items():
                original_param_name = self.randomized_to_original.get(param_name, param_name)
                original_params[original_param_name] = param_value
            
            logger.info(f"Converting randomized call: {tool_name}({list(params.keys())}) -> {original_tool_name}({list(original_params.keys())})")
        else:
            original_tool_name = tool_name
            original_params = params
        
        try:
            signal.alarm(timeout_sec)
            
            if original_tool_name in self.available_offline_tools:
                try:
                    tool_generate_fn = get_tool_generate_fn(original_tool_name)
                    if tool_generate_fn is None:
                        ret_message = {"text": f"Tool {tool_name} not found.", "error_code": 1}
                    else:
                        ret_message = tool_generate_fn(original_params)
                except Exception as e:
                    logger.error(f"Failed to call tool {tool_name}: {e}")
                    ret_message = {"text": f"Failed to call tool {tool_name}: {e}", "error_code": 1}
                
            elif original_tool_name in self.available_online_tools:
                try:
                    tool_worker_addr = self.online_tool_addr_dict[original_tool_name]
                    with self.disable_proxy():
                        session = requests.Session()
                        session.trust_env = False
                        ret = session.post(tool_worker_addr + "/worker_generate", 
                                          headers=self.headers, json=original_params, proxies={},
                                          timeout=(300, 300))
                    ret_message = ret.json()
                except Exception as e:
                    logger.error(f"Failed to call tool {tool_name}: {e}")
                    ret_message = {"text": f"Failed to call tool {tool_name}: {e}", "error_code": 1}
            else:
                ret_message = {"text": f"Tool {tool_name} not found.", "error_code": 1}
                
            edited_image = ret_message.get("edited_image", None)
            if edited_image:
                edited_image_pil = load_image(edited_image)
                width, height = edited_image_pil.size
                if width < 28 or height < 28:
                    ret_message.pop("edited_image")
            signal.alarm(0)
            
            # 将返回的工具名也随机化（如果需要）
            if "tool_response_from" in ret_message and self.randomize:
                ret_message["tool_response_from"] = tool_name
                
        except TimeoutException as te:
            logger.error(f"Timeout calling tool {original_tool_name}: {te}")
            ret_message = {"text": f"Timeout calling tool {original_tool_name}: {te}", "error_code": 1}
        finally:
            signal.alarm(0)
            return ret_message
            
    @contextmanager
    def disable_proxy(self):
        """临时禁用代理设置的上下文管理器"""
        # 保存代理环境变量
        old_HTTP = os.environ.get("HTTP_PROXY")
        old_HTTPS = os.environ.get("HTTPS_PROXY")
        old_http = os.environ.get("http_proxy")
        old_https = os.environ.get("https_proxy")
        
        # 移除代理设置
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        
        try:
            yield
        finally:
            # 恢复代理设置
            if old_http is not None:
                os.environ["http_proxy"] = old_http
            if old_https is not None:
                os.environ["https_proxy"] = old_https
            if old_HTTP is not None:
                os.environ["HTTP_PROXY"] = old_HTTP
            if old_HTTPS is not None:
                os.environ["HTTPS_PROXY"] = old_HTTPS
    
    def get_randomization_info(self) -> Dict[str, Any]:
        """
        获取随机化信息（用于调试和验证）
        
        Returns:
            dict: 包含随机化映射的字典
        """
        if not self.randomize:
            return {"randomize": False}
        
        return {
            "randomize": True,
            "tool_mappings": self.original_to_randomized,
            "tool_descriptions": self.tool_desc_map,
            "param_descriptions": self.param_desc_map,
            "num_cached_original_instructions": len(self.original_instructions),
            "num_cached_randomized_instructions": len(self.randomized_instructions)
        }