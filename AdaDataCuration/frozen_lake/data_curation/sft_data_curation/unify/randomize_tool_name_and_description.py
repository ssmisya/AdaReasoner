import json
import os
import re
import random
import string
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional
from tool_server.tool_workers.tool_manager.base_manager import ToolManager

# 定义常用工具名称和参数名称，用于识别和替换
COMMON_TOOL_NAMES = {
    "Point": ["image", "description"],
    "Draw2DPath": ["image", "start_point", "directions", "step", "pixel_coordinate", "line_width", "line_color"],
    "AStarWithPixelCoordinate": ["start", "goal", "obstacles"],
    "DetectBlackArea": ["image", "threshold", "min_area"],
    "InsertImage": ["base_image", "image_to_insert", "coordinates", "resize"],
    "Crop": ["image", "coordinates"],
    "OCR": ["image"]
}

# 为每个工具提供备选描述（预设一些，可以后续扩充）
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

# 为每个工具的参数提供独立的备选描述
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

def add_descriptions(tool_desc_map: Dict[str, List[str]], 
                     tool_param_desc_map: Dict[str, Dict[str, List[str]]]):
    """添加外部提供的描述到描述映射中"""
    # 添加工具描述
    for tool_name, descriptions in tool_desc_map.items():
        if tool_name in TOOL_DESCRIPTIONS:
            # 如果已有该工具的描述，则扩展它
            TOOL_DESCRIPTIONS[tool_name].extend(descriptions)
        else:
            # 否则创建新的描述列表
            TOOL_DESCRIPTIONS[tool_name] = descriptions
    
    # 添加工具参数描述
    for tool_name, params in tool_param_desc_map.items():
        if tool_name not in TOOL_PARAM_DESCRIPTIONS:
            TOOL_PARAM_DESCRIPTIONS[tool_name] = {}
            
        for param_name, descriptions in params.items():
            if param_name in TOOL_PARAM_DESCRIPTIONS.get(tool_name, {}):
                # 如果已有该参数的描述，则扩展它
                TOOL_PARAM_DESCRIPTIONS[tool_name][param_name].extend(descriptions)
            else:
                # 否则创建新的描述列表
                TOOL_PARAM_DESCRIPTIONS[tool_name][param_name] = descriptions

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

def generate_description_map(tool_names, param_names_by_tool):
    """
    为工具和参数生成描述替换映射
    
    Args:
        tool_names: 需要替换的工具名称列表
        param_names_by_tool: 按工具组织的参数名称字典 {工具名: [参数名列表]}
        
    Returns:
        tuple: (工具描述映射, 参数描述映射)
    """
    # 工具描述映射: {原描述: 新描述}
    tool_desc_map = {}
    # 参数描述映射: {原描述: 新描述}
    param_desc_map = {}
    
    # 为工具生成描述替换
    for tool_name in tool_names:
        if tool_name in TOOL_DESCRIPTIONS and TOOL_DESCRIPTIONS[tool_name]:
            # 随机选择一个备选描述
            selected_desc = random.choice(TOOL_DESCRIPTIONS[tool_name])
            tool_desc_map[tool_name] = selected_desc
    
    # 为每个工具的参数生成描述替换
    for tool_name, param_names in param_names_by_tool.items():
        if tool_name in TOOL_PARAM_DESCRIPTIONS:
            for param_name in param_names:
                if param_name in TOOL_PARAM_DESCRIPTIONS[tool_name] and TOOL_PARAM_DESCRIPTIONS[tool_name][param_name]:
                    # 随机选择一个备选描述
                    selected_desc = random.choice(TOOL_PARAM_DESCRIPTIONS[tool_name][param_name])
                    if param_name not in param_desc_map:
                        param_desc_map[param_name] = {}
                    param_desc_map[param_name][tool_name] = selected_desc
    
    return tool_desc_map, param_desc_map

def extract_tools_and_params(instance: Dict[str, Any]) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    从整个实例中提取工具名称和按工具组织的参数名称
    
    Args:
        instance: 数据实例
        
    Returns:
        Tuple[Set[str], Dict[str, Set[str]]]: 工具名称集合和按工具组织的参数名称字典
    """
    tool_names = set()
    param_names_by_tool = {}
    
    # 遍历对话
    if "conversations" in instance:
        for msg in instance["conversations"]:
            msg_from = msg.get("from", "")
            content = msg.get("value", "")
            
            if msg_from == "system":
                # 处理系统提示
                system_tools, system_params = extract_from_system_prompt(content)
                tool_names.update(system_tools)
                
                # 更新参数映射
                for tool, params in system_params.items():
                    if tool not in param_names_by_tool:
                        param_names_by_tool[tool] = set()
                    param_names_by_tool[tool].update(params)
                    
            elif msg_from == "gpt":
                # 处理GPT回复，查找tool_call
                call_tools, call_params = extract_from_tool_call(content)
                tool_names.update(call_tools)
                
                # 更新参数映射
                for tool, params in call_params.items():
                    if tool not in param_names_by_tool:
                        param_names_by_tool[tool] = set()
                    param_names_by_tool[tool].update(params)
                    
            elif msg_from == "human":
                # 处理Human回复，可能包含工具响应
                response_tools, response_params = extract_from_tool_response(content)
                tool_names.update(response_tools)
                
                # 更新参数映射
                for tool, params in response_params.items():
                    if tool not in param_names_by_tool:
                        param_names_by_tool[tool] = set()
                    param_names_by_tool[tool].update(params)
    
    # 添加常见工具和参数
    for tool_name in tool_names.copy():
        if tool_name in COMMON_TOOL_NAMES:
            if tool_name not in param_names_by_tool:
                param_names_by_tool[tool_name] = set()
            param_names_by_tool[tool_name].update(COMMON_TOOL_NAMES[tool_name])
    
    return tool_names, param_names_by_tool

def extract_tool_descriptions(system_prompt: str) -> Dict[str, Dict]:
    """
    从系统提示中提取工具及其参数的描述
    
    Args:
        system_prompt: 系统提示文本
        
    Returns:
        Dict[str, Dict]: 
            {工具名称: {
                "description": 工具描述,
                "params": {参数名称: 参数描述, ...}
            }}
    """
    descriptions = {}
    
    # 尝试提取工具定义和描述
    # 找到所有工具定义的JSON或类JSON格式
    function_matches = re.finditer(r"'function':\s*{.*?'name':\s*'([^']+)'.*?'description':\s*'([^']+)'.*?'parameters':\s*{(.*?)}", system_prompt, re.DOTALL)
    for match in function_matches:
        tool_name = match.group(1)
        tool_desc = match.group(2)
        params_text = match.group(3)
        
        param_descriptions = {}
        # 提取参数及其描述
        param_matches = re.finditer(r"'([^']+)':\s*{.*?'description':\s*'([^']+)'", params_text, re.DOTALL)
        for param_match in param_matches:
            param_name = param_match.group(1)
            param_desc = param_match.group(2)
            param_descriptions[param_name] = param_desc
        
        descriptions[tool_name] = {
            "description": tool_desc,
            "params": param_descriptions
        }
    
    # 如果没有找到单引号格式，尝试双引号格式
    if not descriptions:
        function_matches = re.finditer(r'"function":\s*{.*?"name":\s*"([^"]+)".*?"description":\s*"([^"]+)".*?"parameters":\s*{(.*?)}"', system_prompt, re.DOTALL)
        for match in function_matches:
            tool_name = match.group(1)
            tool_desc = match.group(2)
            params_text = match.group(3)
            
            param_descriptions = {}
            # 提取参数及其描述
            param_matches = re.finditer(r'"([^"]+)":\s*{.*?"description":\s*"([^"]+)"', params_text, re.DOTALL)
            for param_match in param_matches:
                param_name = param_match.group(1)
                param_desc = param_match.group(2)
                param_descriptions[param_name] = param_desc
            
            descriptions[tool_name] = {
                "description": tool_desc,
                "params": param_descriptions
            }
    
    return descriptions

def extract_from_system_prompt(system_prompt: str) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    从系统提示中提取工具名称和参数
    
    Returns:
        Tuple[Set[str], Dict[str, Set[str]]]: 工具名称集合和按工具组织的参数名称集合
    """
    tool_names = set()
    param_names_by_tool = {}
    
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
                    if tool_name not in param_names_by_tool:
                        param_names_by_tool[tool_name] = set()
                    param_names_by_tool[tool_name].add(param_name)
    
    # 2. 检查已知工具引用
    for tool_name, params in COMMON_TOOL_NAMES.items():
        if tool_name in system_prompt:
            tool_names.add(tool_name)
            if tool_name not in param_names_by_tool:
                param_names_by_tool[tool_name] = set()
                
            for param in params:
                # 只在这些参数作为独立词出现时添加，避免误匹配
                if re.search(r'[\'"]' + re.escape(param) + r'[\'"]', system_prompt):
                    param_names_by_tool[tool_name].add(param)
    
    return tool_names, param_names_by_tool

def extract_from_tool_call(content: str) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    从工具调用中提取工具名称和参数
    
    Returns:
        Tuple[Set[str], Dict[str, Set[str]]]: 工具名称集合和按工具组织的参数名称集合
    """
    tool_names = set()
    param_names_by_tool = {}
    
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
                
                if 'parameters' in tool_call_obj and isinstance(tool_call_obj['parameters'], dict):
                    if tool_name not in param_names_by_tool:
                        param_names_by_tool[tool_name] = set()
                        
                    for param_name in tool_call_obj['parameters']:
                        param_names_by_tool[tool_name].add(param_name)
                        
        except json.JSONDecodeError:
            # 如果不是格式良好的JSON，使用正则表达式提取
            name_match = re.search(r'"name":\s*"([^"]+)"', tool_call_text)
            if name_match:
                tool_name = name_match.group(1)
                tool_names.add(tool_name)
                
                if tool_name not in param_names_by_tool:
                    param_names_by_tool[tool_name] = set()
            
            # 提取参数名称
            param_matches = re.finditer(r'"([^"]+)":\s*[{\[]', tool_call_text)
            for param_match in param_matches:
                param_name = param_match.group(1)
                if param_name != 'parameters' and name_match:  # 跳过parameters本身
                    param_names_by_tool[tool_name].add(param_name)
    
    return tool_names, param_names_by_tool

def extract_from_tool_response(content: str) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    从工具响应中提取工具名称和参数
    
    Returns:
        Tuple[Set[str], Dict[str, Set[str]]]: 工具名称集合和按工具组织的参数名称集合
    """
    tool_names = set()
    param_names_by_tool = {}
    
    # 查找工具响应JSON
    json_blocks = find_json_blocks(content)
    for json_text in json_blocks:
        try:
            json_obj = json.loads(json_text)
            if isinstance(json_obj, dict):
                if 'tool_response_from' in json_obj:
                    tool_name = json_obj['tool_response_from']
                    tool_names.add(tool_name)
                    
                    if tool_name not in param_names_by_tool:
                        param_names_by_tool[tool_name] = set()
                
                # 递归检查所有键以查找参数
                for key in json_obj:
                    if key not in ["error_code", "status", "message", "execution_time", "width", "height", "tool_response_from"]:
                        if 'tool_response_from' in json_obj:
                            param_names_by_tool[tool_name].add(key)
                        
                    # 特别处理嵌套结构，如points下的x,y
                    if key == "points" and isinstance(json_obj[key], list):
                        for point in json_obj[key]:
                            if isinstance(point, dict):
                                for point_key in point:
                                    if 'tool_response_from' in json_obj:
                                        param_names_by_tool[tool_name].add(point_key)
                
                # 检查image_dimensions_pixels内部
                if 'image_dimensions_pixels' in json_obj and isinstance(json_obj['image_dimensions_pixels'], dict):
                    dims = json_obj['image_dimensions_pixels']
                    for dim_key in dims:
                        if 'tool_response_from' in json_obj:
                            param_names_by_tool[tool_name].add(dim_key)
        except json.JSONDecodeError:
            continue
    
    return tool_names, param_names_by_tool

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

def replace_json_in_text(text: str, replacement_map: Dict[str, str], 
                        tool_desc_map: Dict[str, str] = None,
                        param_desc_map: Dict[str, Dict[str, str]] = None,
                        current_tool_context: str = None) -> str:
    """
    替换文本中JSON部分的工具名称、参数和描述
    
    Args:
        text: 要处理的文本
        replacement_map: 名称替换映射
        tool_desc_map: 工具描述替换映射
        param_desc_map: 参数描述替换映射 {参数名: {工具名: 描述}}
        current_tool_context: 当前处理的工具名称上下文
    """
    json_blocks = find_json_blocks(text)
    
    # 按长度降序排序，确保先替换较长的块
    json_blocks.sort(key=len, reverse=True)
    
    for json_text in json_blocks:
        try:
            json_obj = json.loads(json_text)
            # 检测当前JSON是否包含工具名称
            tool_name_in_json = None
            if isinstance(json_obj, dict):
                if 'name' in json_obj:
                    tool_name_in_json = json_obj['name']
                elif 'tool_response_from' in json_obj:
                    tool_name_in_json = json_obj['tool_response_from']
                elif current_tool_context:
                    tool_name_in_json = current_tool_context
            
            replaced_obj = replace_in_json_object(
                json_obj, 
                replacement_map, 
                tool_desc_map, 
                param_desc_map,
                tool_name_in_json
            )
            replaced_json = json.dumps(replaced_obj)
            
            # 确保不会部分替换更长的字符串
            if replaced_json != json_text:
                text = text.replace(json_text, replaced_json)
        except json.JSONDecodeError:
            continue
    
    return text

def replace_in_json_object(obj: Any, replacement_map: Dict[str, str], 
                          tool_desc_map: Dict[str, str] = None,
                          param_desc_map: Dict[str, Dict[str, str]] = None,
                          current_tool: str = None) -> Any:
    """
    递归替换JSON对象中的值
    
    Args:
        obj: 要处理的JSON对象
        replacement_map: 名称替换映射
        tool_desc_map: 工具描述替换映射
        param_desc_map: 参数描述替换映射 {参数名: {工具名: 描述}}
        current_tool: 当前处理的工具名称上下文
    """
    if isinstance(obj, dict):
        new_dict = {}
        # 确定当前工具上下文
        tool_context = current_tool
        if 'name' in obj and isinstance(obj['name'], str) and obj['name'] in replacement_map:
            tool_context = obj['name']
        elif 'tool_response_from' in obj and isinstance(obj['tool_response_from'], str) and obj['tool_response_from'] in replacement_map:
            tool_context = obj['tool_response_from']
            
        for k, v in obj.items():
            # 替换键
            new_key = replacement_map.get(k, k)
            
            # 递归替换值
            new_val = replace_in_json_object(v, replacement_map, tool_desc_map, param_desc_map, tool_context)
            
            # 特殊处理工具名称
            if k == "name" and isinstance(v, str) and v in replacement_map:
                new_val = replacement_map[v]
            elif k == "tool_response_from" and isinstance(v, str) and v in replacement_map:
                new_val = replacement_map[v]
            # 特殊处理工具描述    
            elif k == "description" and isinstance(v, str) and tool_desc_map:
                # 尝试查找对应的工具描述
                if tool_context and tool_context in tool_desc_map:
                    new_val = tool_desc_map[tool_context]
                # 回退：尝试直接通过描述文本查找
                elif v in tool_desc_map.values():
                    for tool_name, desc in tool_desc_map.items():
                        if desc == v:
                            new_val = tool_desc_map[tool_name]
                            break
            # 特殊处理参数描述
            elif k == "description" and isinstance(v, str) and param_desc_map:
                # 查找父键（参数名称）
                parent_key = None
                for parent_k, parent_v in obj.items():
                    if isinstance(parent_v, dict) and "description" in parent_v and parent_v["description"] == v:
                        parent_key = parent_k
                        break
                
                # 如果找到参数名，且该参数存在于参数描述映射中
                if parent_key and parent_key in param_desc_map and tool_context and tool_context in param_desc_map[parent_key]:
                    new_val = param_desc_map[parent_key][tool_context]
                
            new_dict[new_key] = new_val
                
        return new_dict
    elif isinstance(obj, list):
        return [replace_in_json_object(item, replacement_map, tool_desc_map, param_desc_map, current_tool) for item in obj]
    elif isinstance(obj, str) and obj in replacement_map:
        # 如果字符串值匹配替换映射中的工具名
        return replacement_map[obj]
    else:
        return obj

def replace_tool_call_block(text: str, replacement_map: Dict[str, str], 
                           tool_desc_map: Dict[str, str] = None,
                           param_desc_map: Dict[str, Dict[str, str]] = None) -> str:
    """
    替换<tool_call>标签内的工具名称、参数和描述
    
    Args:
        text: 要处理的文本
        replacement_map: 名称替换映射
        tool_desc_map: 工具描述替换映射 {工具名: 描述}
        param_desc_map: 参数描述替换映射 {参数名: {工具名: 描述}}
    """
    def replace_match(match):
        tool_call_text = match.group(1)
        
        # 尝试识别当前工具
        tool_name_match = re.search(r'"name":\s*"([^"]+)"', tool_call_text)
        current_tool = None
        if tool_name_match:
            current_tool = tool_name_match.group(1)
            
        try:
            # 尝试作为JSON解析并替换
            if tool_call_text.strip().startswith('{') and tool_call_text.strip().endswith('}'):
                tool_call_obj = json.loads(tool_call_text)
                replaced_obj = replace_in_json_object(
                    tool_call_obj, 
                    replacement_map, 
                    tool_desc_map, 
                    param_desc_map,
                    current_tool
                )
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
        
        # 替换工具描述
        if tool_desc_map and current_tool and current_tool in tool_desc_map:
            # 尝试查找描述字段并替换
            desc_match = re.search(r'"description":\s*"([^"]+)"', tool_call_text)
            if desc_match:
                original_desc = desc_match.group(1)
                new_desc = tool_desc_map[current_tool]
                tool_call_text = tool_call_text.replace(f'"{original_desc}"', f'"{new_desc}"')
        
        # 替换参数描述
        if param_desc_map and current_tool:
            for param_name, tool_desc_dict in param_desc_map.items():
                if current_tool in tool_desc_dict:
                    # 查找该参数的描述字段
                    param_desc_match = re.search(f'"{param_name}"[^{{]*{{[^}}]*"description":\\s*"([^"]+)"', tool_call_text, re.DOTALL)
                    if param_desc_match:
                        original_desc = param_desc_match.group(1)
                        new_desc = tool_desc_dict[current_tool]
                        tool_call_text = tool_call_text.replace(f'"{original_desc}"', f'"{new_desc}"')
        
        return f"<tool_call>\n{tool_call_text}\n</tool_call>"
    
    return re.sub(r'<tool_call>\s*(.*?)\s*</tool_call>', replace_match, text, flags=re.DOTALL)

def replace_system_prompt(system_prompt: str, replacement_map: Dict[str, str],
                         tool_desc_map: Dict[str, str],
                         param_desc_map: Dict[str, Dict[str, str]], renew_system_prompt) -> str:
    """
    替换系统提示中的工具名称、参数和描述
    
    Args:
        system_prompt: 系统提示文本
        replacement_map: 名称替换映射 {原名称: 新名称}
        tool_desc_map: 工具描述替换映射 {工具名: 描述}
        param_desc_map: 参数描述替换映射 {参数名: {工具名: 描述}}
    """
    if renew_system_prompt:
        system_prompt = renew_system_prompt
    if "In your response, you can use the following tools:" not in system_prompt:
        # 如果找不到标准格式，尝试直接替换
        replaced_prompt = system_prompt
        
        # 替换工具名称和参数名称
        for original, replacement in replacement_map.items():
            replaced_prompt = re.sub(f'\\b{re.escape(original)}\\b', replacement, replaced_prompt)
            
        # 替换工具描述
        for tool_name, desc in tool_desc_map.items():
            # 查找原始描述字段
            desc_match = re.search(f'"description":\\s*"([^"]+)"', system_prompt)
            if desc_match:
                original_desc = desc_match.group(1)
                replaced_prompt = replaced_prompt.replace(f'"{original_desc}"', f'"{desc}"')
                replaced_prompt = replaced_prompt.replace(f"'{original_desc}'", f"'{desc}'")
            
        # 替换参数描述
        for param_name, tool_descs in param_desc_map.items():
            for tool_name, new_desc in tool_descs.items():
                # 查找该参数的原始描述
                param_desc_match = re.search(f'"{param_name}"[^{{]*{{[^}}]*"description":\\s*"([^"]+)"', system_prompt, re.DOTALL)
                if param_desc_match:
                    original_desc = param_desc_match.group(1)
                    replaced_prompt = replaced_prompt.replace(f'"{original_desc}"', f'"{new_desc}"')
                    replaced_prompt = replaced_prompt.replace(f"'{original_desc}'", f"'{new_desc}'")
                
        return replaced_prompt
    
    # 标准格式处理
    description_jsons = system_prompt.split('In your response, you can use the following tools:  \n')[-1].split("\n\nSteps for Each Turn\n1. **Think:** First, silently analyze the user's request to understand the goal. ")[0]
    
    prefix = system_prompt.split('In your response, you can use the following tools:  \n')[0] + 'In your response, you can use the following tools:  \n'
    suffix = "\n\nSteps for Each Turn\n1. **Think:** First, silently analyze the user's request to understand the goal. " + system_prompt.split("\n\nSteps for Each Turn\n1. **Think:** First, silently analyze the user's request to understand the goal. ")[-1]
    
    description_jsons = description_jsons.split("\n")
    renewed_dicts = []
    
    for description_json in description_jsons:
        try:
            # 尝试解析工具定义
            desc_dict = eval(description_json)
            
            # 获取当前工具名称
            original_name = desc_dict["function"]["name"]
            
            # 替换工具名称
            desc_dict["function"]["name"] = replacement_map.get(original_name, original_name)
            
            # 替换工具描述
            if original_name in tool_desc_map:
                desc_dict["function"]["description"] = tool_desc_map[original_name]
            
            # 处理参数
            old_params = desc_dict["function"]["parameters"]
            new_properties = {}
            
            for param_name, param_info in old_params["properties"].items():
                # 替换参数名称
                new_param_name = replacement_map.get(param_name, param_name)
                
                # 替换参数描述
                if "description" in param_info and param_name in param_desc_map and original_name in param_desc_map[param_name]:
                    param_info["description"] = param_desc_map[param_name][original_name]
                
                new_properties[new_param_name] = param_info
            
            desc_dict["function"]["parameters"]["properties"] = new_properties
            
            # 更新required参数列表
            if "required" in old_params:
                desc_dict["function"]["parameters"]["required"] = [
                    replacement_map.get(k, k) for k in old_params["required"]
                ]
            
            renewed_dicts.append(desc_dict)
            
        except (SyntaxError, ValueError) as e:
            # 如果解析失败，保留原样
            print(f"解析工具定义失败: {e}")
            renewed_dicts.append(eval(description_json))
    
    renewed_description = "\n".join([str(d) for d in renewed_dicts])
    replaced_prompt = prefix + renewed_description + suffix
    return replaced_prompt

def replace_in_text(text: str, replacement_map: Dict[str, str], 
                   tool_desc_map: Dict[str, str] = None,
                   param_desc_map: Dict[str, Dict[str, str]] = None,
                   current_tool: str = None) -> str:
    """
    替换文本中的工具名称和参数名称，保留格式
    
    Args:
        text: 要处理的文本
        replacement_map: 名称替换映射 {原名称: 新名称}
        tool_desc_map: 工具描述替换映射 {工具名: 描述}
        param_desc_map: 参数描述替换映射 {参数名: {工具名: 描述}}
        current_tool: 当前处理的工具上下文
    """
    # 排序替换映射，确保先替换较长的词，以避免部分替换
    sorted_replacements = sorted(replacement_map.items(), key=lambda x: len(x[0]), reverse=True)
    replaced_text = text
    
    # 1. 检查并替换工具调用块
    if "<tool_call>" in replaced_text:
        replaced_text = replace_tool_call_block(replaced_text, replacement_map, tool_desc_map, param_desc_map)
    
    # 2. 检查并替换JSON块
    replaced_text = replace_json_in_text(replaced_text, replacement_map, tool_desc_map, param_desc_map, current_tool)
    
    # 3. 直接替换其他引用，但必须确保是完整的词
    for original, replacement in sorted_replacements:
        # 工具名称替换
        replaced_text = re.sub(
            f'\\b{re.escape(original)}\\b', 
            replacement, 
            replaced_text
        )
    
    # 4. 替换描述（如果能确定当前工具上下文）
    if current_tool and tool_desc_map and current_tool in tool_desc_map:
        # 工具描述通常不会直接出现在文本中，不做处理
        pass
        
    return replaced_text

def replace_tool_call_in_text(text: str, replacement_map: Dict[str, str], 
                             tool_desc_map: Dict[str, str] = None,
                             param_desc_map: Dict[str, Dict[str, str]] = None) -> str:
    """
    替换工具调用中的工具名称、参数和描述
    
    Args:
        text: 要处理的文本
        replacement_map: 名称替换映射
        tool_desc_map: 工具描述替换映射
        param_desc_map: 参数描述替换映射
    """
    replaced_tool_call = text
    if "<tool_call>" in text:
        tool_call_json = text.split('<tool_call>')[-1].split('</tool_call>')[0]
        prefix = text.split('<tool_call>')[0] + '<tool_call>'
        suffix = '</tool_call>' + text.split('</tool_call>')[-1]
        
        # 识别当前工具
        tool_name_match = re.search(r'"name":\s*"([^"]+)"', tool_call_json)
        current_tool = None
        if tool_name_match:
            current_tool = tool_name_match.group(1)
        
        try:
            # 处理JSON中的true/false
            tool_call_json = tool_call_json.replace("true","\"true\"").replace("false","\"false\"")
            tool_call_dict = eval(tool_call_json)
            
            # 替换工具名称
            if "name" in tool_call_dict and tool_call_dict["name"] in replacement_map:
                tool_call_dict["name"] = replacement_map[tool_call_dict["name"]]
            
            # 替换参数名称
            if "parameters" in tool_call_dict and isinstance(tool_call_dict["parameters"], dict):
                new_params = {}
                for k, v in tool_call_dict["parameters"].items():
                    new_key = replacement_map.get(k, k)
                    new_params[new_key] = v
                
                tool_call_dict["parameters"] = new_params
            
            # 构建替换后的调用
            replaced_tool_call = prefix + "\n" + json.dumps(tool_call_dict, indent=2) + "\n" + suffix
        except (SyntaxError, ValueError) as e:
            # 如果解析失败，使用正则替换
            print(f"解析工具调用失败: {e}")
            for original, replacement in replacement_map.items():
                tool_call_json = re.sub(f'"name"\\s*:\\s*"({re.escape(original)})"', f'"name": "{replacement}"', tool_call_json)
                tool_call_json = re.sub(f'"({re.escape(original)})"\\s*:', f'"{replacement}":', tool_call_json)
            
            replaced_tool_call = prefix + tool_call_json + suffix
    
    # 处理<think>标签内容
    if "<think>" in replaced_tool_call:
        think_content = replaced_tool_call.split('<think>')[-1].split('</think>')[0]
        t_prefix = replaced_tool_call.split('<think>')[0] + '<think>'
        t_suffix = '</think>' + replaced_tool_call.split('</think>')[-1]
        
        # 尝试从think内容中找出当前工具上下文
        current_tool = None
        for tool_name in replacement_map:
            if tool_name in COMMON_TOOL_NAMES and tool_name in think_content:
                current_tool = tool_name
                break
                
        # 替换工具名称
        for name, sub in replacement_map.items():
            if name in COMMON_TOOL_NAMES:
                think_content = re.sub(f'\\b{re.escape(name)}\\b', sub, think_content)
        
        # 替换工具描述
        if tool_desc_map and current_tool and current_tool in tool_desc_map:
            # 只有在确定当前工具时才尝试替换描述
            original_desc = None
            for tool_name, desc in tool_desc_map.items():
                if tool_name == current_tool and desc in think_content:
                    original_desc = desc
                    break
                    
            if original_desc:
                think_content = think_content.replace(original_desc, tool_desc_map[current_tool])
        
        replaced_tool_call = t_prefix + think_content + t_suffix
        
    return replaced_tool_call

def replace_tool_resp_in_text(text: str, replacement_map: Dict[str, str]) -> str:
    """
    替换工具响应中的工具名称和参数
    
    Args:
        text: 要处理的文本
        replacement_map: 名称替换映射
    """
    # 处理图像标签
    has_image = "<image>" in text
    json_text = text.replace("<image>", "")
    
    try:
        # 解析响应JSON
        resp_dict = eval(json_text)
        
        # 获取当前工具上下文
        current_tool = None
        if "tool_response_from" in resp_dict:
            current_tool = resp_dict["tool_response_from"]
            resp_dict["tool_response_from"] = replacement_map.get(current_tool, current_tool)
        
        # 替换参数名
        new_dict = {}
        for k, v in resp_dict.items():
            new_key = replacement_map.get(k, k)
            new_dict[new_key] = v
            
        # 构建替换后的响应
        replaced_text = json.dumps(new_dict, indent=2)
        if has_image:
            replaced_text += "<image>"
            
        return replaced_text
    
    except (SyntaxError, ValueError) as e:
        # 如果解析失败，使用直接替换
        print(f"解析工具响应失败: {e}")
        replaced_text = text
        for original, replacement in replacement_map.items():
            replaced_text = re.sub(f'\\b{re.escape(original)}\\b', replacement, replaced_text)
            
        return replaced_text

def extract_and_map_descriptions(instance: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """
    提取并映射工具和参数的描述
    
    Args:
        instance: 数据实例
        
    Returns:
        Tuple[Dict[str, str], Dict[str, Dict[str, str]]]: 
            工具描述映射 {工具名: 描述}
            参数描述映射 {参数名: {工具名: 描述}}
    """
    tool_descriptions = {}
    param_descriptions = {}
    
    # 从系统提示中提取描述
    if "conversations" in instance:
        for msg in instance["conversations"]:
            if msg.get("from") == "system":
                system_prompt = msg.get("value", "")
                tool_defs = extract_tool_descriptions(system_prompt)
                
                # 收集工具描述
                for tool_name, info in tool_defs.items():
                    if "description" in info:
                        if tool_name in TOOL_DESCRIPTIONS and TOOL_DESCRIPTIONS[tool_name]:
                            # 随机选择一个新描述
                            new_desc = random.choice(TOOL_DESCRIPTIONS[tool_name])
                            tool_descriptions[tool_name] = new_desc
                        
                    # 收集参数描述
                    for param_name, param_desc in info.get("params", {}).items():
                        # 确保参数描述映射中有该参数
                        if param_name not in param_descriptions:
                            param_descriptions[param_name] = {}
                            
                        # 随机选择一个新的参数描述
                        if (tool_name in TOOL_PARAM_DESCRIPTIONS and 
                            param_name in TOOL_PARAM_DESCRIPTIONS[tool_name] and 
                            TOOL_PARAM_DESCRIPTIONS[tool_name][param_name]):
                            new_desc = random.choice(TOOL_PARAM_DESCRIPTIONS[tool_name][param_name])
                            param_descriptions[param_name][tool_name] = new_desc
    
    return tool_descriptions, param_descriptions

def process_conversation_message(msg: Dict[str, Any], replacement_map: Dict[str, str],
                                tool_desc_map: Dict[str, str] = None,
                                param_desc_map: Dict[str, Dict[str, str]] = None, renew_system_prompt=None) -> Dict[str, Any]:
    """
    根据消息类型处理对话消息
    
    Args:
        msg: 对话消息
        replacement_map: 名称替换映射
        tool_desc_map: 工具描述替换映射
        param_desc_map: 参数描述替换映射
    """
    msg_type = msg.get("from", "")
    content = msg.get("value", "")
    
    if "value" not in msg:
        return msg
    
    if msg_type == "system":
        # 系统提示特殊处理
        msg["value"] = replace_system_prompt(content, replacement_map, tool_desc_map, param_desc_map, renew_system_prompt=renew_system_prompt)
    elif msg_type == "gpt":
        # 包含工具调用的GPT回复
        msg["value"] = replace_tool_call_in_text(content, replacement_map, tool_desc_map, param_desc_map)
    elif msg_type == "human" and ("tool_response_from" in content):
        # 包含工具响应的Human回复
        msg["value"] = replace_tool_resp_in_text(content, replacement_map)
    
    return msg

def randomize_instance(instance: Dict[str, Any], renew_system_prompt) -> Dict[str, Any]:
    """
    随机化一个数据实例的工具名称、参数名称和描述
    
    Args:
        instance: 数据实例
        
    Returns:
        Dict[str, Any]: 随机化后的实例
    """
    # 1. 提取工具名称和按工具组织的参数
    tool_names, param_names_by_tool = extract_tools_and_params(instance)
    
    if not tool_names:
        # 如果没有找到工具名称，直接返回原始实例
        return instance
    
    # 2. 生成替换映射
    replacement_map = {}
    
    # 为工具名称生成替换
    for tool_name in tool_names:
        replacement_map[tool_name] = generate_random_name(prefix="")
    
    # 为参数名称生成替换
    all_params = set()
    for params in param_names_by_tool.values():
        all_params.update(params)
        
    for param_name in all_params:
        replacement_map[param_name] = generate_random_name(prefix="")
    
    # 3. 提取并映射工具和参数描述
    tool_desc_map, param_desc_map = extract_and_map_descriptions(instance)
    
    # 4. 应用替换
    randomized_instance = instance.copy()
    
    # 4.1 处理对话
    if "conversations" in randomized_instance:
        for i, msg in enumerate(randomized_instance["conversations"]):
            randomized_instance["conversations"][i] = process_conversation_message(
                msg, replacement_map, tool_desc_map, param_desc_map,renew_system_prompt=renew_system_prompt,
            )
    
    return randomized_instance

def randomize_dataset(input_file, output_file, external_tool_descs=None, external_param_descs=None, renew_system_prompt=None):
    """
    随机化整个数据集
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        external_tool_descs: 外部提供的工具描述 {工具名: [描述列表]}
        external_param_descs: 外部提供的参数描述 {工具名: {参数名: [描述列表]}}
    """
    # 添加外部提供的描述（如果有）
    if external_tool_descs or external_param_descs:
        add_descriptions(external_tool_descs or {}, external_param_descs or {})
    
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
        randomized_instance = randomize_instance(instance, renew_system_prompt = renew_system_prompt)
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
    total_tool_descs = 0
    total_param_descs = 0
    
    for instance in data:
        tools, params_by_tool = extract_tools_and_params(instance)
        tool_descs, param_descs = extract_and_map_descriptions(instance)
        
        total_tools += len(tools)
        total_params += sum(len(params) for params in params_by_tool.values())
        total_tool_descs += len(tool_descs)
        total_param_descs += sum(len(tool_descs) for tool_descs in param_descs.values())
    
    print(f"Total tool names replaced: {total_tools}")
    print(f"Total parameter names replaced: {total_params}")
    print(f"Total tool descriptions replaced: {total_tool_descs}")
    print(f"Total parameter descriptions replaced: {total_param_descs}")

def main():
    parser = argparse.ArgumentParser(description="Randomize tool names, parameters and descriptions in training data")
    parser.add_argument("--input", type=str, default="/mnt/petrelfs/share_data/sunhaoyu/datasets/web/webdata_ocr_sharegpt.json", help="Input JSON file path")
    parser.add_argument("--output", type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/merged_sft_file/randomized/web_v1_new_desc.json", help="Output JSON file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--desc-file", type=str, default="", help="Optional JSON file with additional descriptions")
    parser.add_argument("--renew-system-prompt", type=str, default="True", help="Whether to renew system prompt descriptions")
    # Web: /mnt/petrelfs/share_data/sunhaoyu/datasets/web/webdata_ocr_sharegpt.json
    # Jigsaw: /mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_metadata_v2/merged_sft_file/vsp2tasks_v2_navigationa.json
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 加载额外的描述（如果有）
    external_tool_descs = {}
    external_param_descs = {}
    
    if args.desc_file and os.path.exists(args.desc_file):
        with open(args.desc_file, 'r', encoding='utf-8') as f:
            desc_data = json.load(f)
            external_tool_descs = desc_data.get("tool_descriptions", {})
            external_param_descs = desc_data.get("param_descriptions", {})
    
    if args.renew_system_prompt.lower() == "true":

        tool_manager = ToolManager(tools=["OCR"])
        renew_system_prompt = tool_manager.get_tool_prompt(prompt_type="one_tool_call")
    else:
        renew_system_prompt = None
        
    # 随机化数据集
    randomize_dataset(args.input, args.output, external_tool_descs, external_param_descs, renew_system_prompt=renew_system_prompt)

if __name__ == "__main__":
    main()