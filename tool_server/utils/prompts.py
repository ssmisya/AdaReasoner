tool_planning_model_prompt_one_tool_call = """
You are a visual assistant capable of solving visual reasoning problems. You can rely on your own capabilities or use external tools to assist in solving. 

Available Tools  
In your response, you can use the following tools:  
{tool_list}

Steps for Each Turn  
1. Think: Recall relevant context and analyze the current user goal.  
2. Decide on Tool Usage: If a tool is needed, specify the tool and its parameters.  
3. Respond Appropriately: If a response is needed, generate one while maintaining consistency across user queries.

Output Format  
<think> Your thoughts and reasoning </think>  
<tool_call>  
{{"name": "Tool name", "parameters": {{"Parameter name": "Parameter content", "…": "…"}}}}  
</tool_call>  
<response> Your final response </response>

Important Notes  
1. You must always include the <think> field to outline your reasoning. Provide one of <tool_call> or <response>. You must not include both <tool_call> and <response> in the same turn because they are mutually exclusive. If tool usage is required, you must instead include both <think> and <tool_call>, and omit <response> for that turn. If no further tool usage is required and ready to answer the user's question, you can then use <think> to summarize your reasoning and include <response> with your final answer, and this indicates the ends of the conversation.

2. You can only invoke a single tool call at a time in the <tool_call> fields. The tool call should be a JSON object with a "name" field and a "parameters" field containing a dictionary of parameters. If no parameters are needed, leave the "parameters" field an empty dictionary. All images have their coordinate origin at the top-left corner.

3. Some tools require image input. You do not need to generate or upload the actual image data—simply refer to an image using a placeholder in the form of "img_n". There may be multiple images present in the dialogue. Besides the original image, additional images may appear as a result of prior tool calls (e.g., edited images returned by visual editing tools). You are free to select which image to use as input for the next tool.
The index "n" in "img_n" refers to the image's position in the dialogue history:
- The original image is always referred to as "img_1".
- Each subsequent image, including those returned from tools, is assigned "img_2", "img_3", and so on, in the order they appear in the dialogue.
For example:{{"parameters": {{"image": "img_1", "other_params": "other_values"}}}}
4. All image coordinates used must be in absolute pixel values, not relative or normalized coordinates. 
"""

# 不使用工具的prompt，一步就要输出response
tool_planning_model_prompt_no_tool_call = '''
You are a visual assistant capable of solving visual reasoning problems. You can only rely on your own capabilities in solving. 

Steps for Each Turn 
1. Think: Recall relevant context and analyze the current user goal.   
2. Respond Appropriately: If a response is needed, generate one while maintaining consistency across user queries.

Output Format  
<think> Your thoughts and reasoning </think>  
<response> Your final response </response>

Important Notes  
1. You must include the <think> field to outline your reasoning. 
'''

tool_planning_model_prompt = """
You are a visual assistant capable of solving visual reasoning problems. You can rely on your own capabilities or use external tools to assist in solving. 
 
Available Tools  
In your response, you can use the following tools:  
{tool_list}

Steps for Each Turn  
1. Think: Recall relevant context and analyze the current user goal.  
2. Decide on Tool Usage: If a tool is needed, specify the tool and its parameters.  
3. Respond Appropriately: If a response is needed, generate one while maintaining consistency across user queries.

Output Format  
<think> Your thoughts and reasoning </think>  
<tool_call>  
{{"name": "Tool name", "parameters": {{"Parameter name": "Parameter content", "…": "…"}}}}  
{{"name": "…", "parameters": {{"… …": "… …", "… …": "… …"}}}}  
…  
</tool_call>  
<response> Your final response </response>

Important Notes  
1. You must always include the <think> field to outline your reasoning. Provide one of <tool_call> or <response>. You must not include both <tool_call> and <response> in the same turn because they are mutually exclusive. If tool usage is required, you must instead include both <think> and <tool_call>, and omit <response> for that turn. If no further tool usage is required and ready to answer the user's question, you can then use <think> to summarize your reasoning and include <response> with your final answer, and this indicates the ends of the conversation.

2. You can invoke multiple tool calls simultaneously in the <tool_call> fields. Each tool call should be a JSON object with a "name" field and a "parameters" field containing a dictionary of parameters. If no parameters are needed, leave the "parameters" field an empty dictionary.  

3. Some tools require image input. You do not need to generate or upload the actual image data—simply refer to an image using a placeholder in the form of "img_n". There may be multiple images present in the dialogue. Besides the original image, additional images may appear as a result of prior tool calls (e.g., edited images returned by visual editing tools). You are free to select which image to use as input for the next tool.
The index "n" in "img_n" refers to the image's position in the dialogue history:
- The original image is always referred to as "img_1".
- Each subsequent image, including those returned from tools, is assigned "img_2", "img_3", and so on, in the order they appear in the dialogue.
For example:{{"parameters": {{"image": "img_1", "other_params": "other_values"}}}}
"""



ocr_instruction = """
{
    "type": "function",
    "function": {
        "name": "OCR",
        "description": "Extracts and localizes text from the given image using OCR. Returns bounding boxes, recognized text, and confidence scores.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "The identifier of the image to analyze, e.g., 'img_1'"
                }
            },
            "required": ["image"]
        }
    }
}
"""

point_instruction = '''
{
    "type": "function",
    "function": {
        "name": "Point",
        "description": "Identify a point in the image based on a natural language description.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "The identifier of the image to edit, e.g., 'img_1'"
                },
                "description": {
                    "type": "string",
                    "description": "A natural language description of the point of interest, e.g., 'the dog's nose', 'center of the clock', 'the tallest tree'."
                }
            },
            "required": ["image", "description"]
        }
    }
}
'''

segment_around_point_instruction = '''
{
    "type": "function",
    "function": {
        "name": "SegmentAroundPoint",
        "description": "Segments objects in an image. Can perform automatic segmentation on the entire image or segment a specific object based on a single designated point. Returns the image with segmentation masks and related processing info.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "The identifier of the image to edit, e.g., 'img_1'"
                },
                "coordinates": {
                    "type": "string",
                    "description": "Optional: Single point coordinates in format 'x=value1, y=value2', eg., 'x=50, y=100'. Using absolute pixel coordinates within image bounds. If not provided, the tool will automatically segment all objects in the image."
                }
            },
            "required": ["image", "coordinates"]
        }
    }
}
'''

drawn_line_instruction = '''
{
    "type": "function",
    "function": {
        "name": "DrawLine",
        "description": "Draw horizontal or vertical lines on an image. This tool supports drawing multiple lines of the same type simultaneously. Only accepts absolute pixel coordinates (not normalized values). Returns base64 encoded image with lines drawn.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "The identifier of the image to edit, e.g., 'img_1'"
                },
                "line_type": {
                    "type": "string",
                    "description": "Type of line to draw: 'horizontal' (requires y coordinates) or 'vertical' (requires x coordinates).",
                    "enum": ["horizontal", "vertical"]
                },
                "coordinates": {
                    "type": "string",
                    "description": "For horizontal lines, provide y coordinates in format 'y1=100, y2=200, y3=300'. For vertical lines, provide x coordinates in format 'x1=100, x2=200, x3=300'. Multiple coordinates should be separated by commas. Only absolute pixel values are supported."
                }
            },
            "required": ["image", "line_type", "coordinates"]
        }
    }
}
'''

grounding_dino_instruction = '''
{
    "type": "function",
    "function": {
        "name": "GroundingDINO",
        "description": "Locate objects in the image based on a natural language description. Returns detected objects with their bounding boxes in absolute pixel coordinates, confidence scores, and an annotated image with visualized detections.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "The identifier of the image to analyze, e.g., 'img_1'"
                },
                "description": {
                    "type": "string",
                    "description": "A natural language description of the object to locate, e.g., 'a red car', 'a man holding a dog'."
                }
            },
            "required": ["image", "description"]
        }
    }
}
'''
draw_shape_instruction = '''
{
    "type": "function",
    "function": {
        "name": "DrawShape",
        "description": 
            "Draw geometric shapes (rectangle, ellipse, or circle) with red borders at specified bounding box locations on the image. Returns the edited image in base64 format.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "The identifier of the image to edit, e.g., 'img_1'"
                },
                "bboxes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "shape": {
                                "type": "string",
                                "enum": ["rectangle", "ellipse", "circle"],
                                "description": "Type of shape to draw."
                            },
                            "coords": {
                                "type": "array",
                                "items": { "type": "integer" },
                                "description": "Bounding box coordinates in [x_min, y_min, x_max, y_max] format."
                            }
                        },
                        "required": ["shape", "coords"]
                    },
                    "description": "List of shapes to draw and their coordinates."
                }
            },
            "required": ["image", "bboxes"]
        }
    }
}
'''

get_bar_info_instruction = '''
{
    "type": "function",
    "function": {
        "name": "GetBarInfo",
        "description": 
            "Extract bounding boxes of all bars in the image along with their corresponding axis titles or labels. Returns a dictionary mapping each label to its bounding box.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "The identifier of the image to analyze, e.g., 'img_1'."
                }
            },
            "required": ["image"]
        }
    }
}
'''

get_subplot_info_instruction = '''
{
    "type": "function",
    "function": {
        "name": "GetSubplotInfo",
        "description": 
            "Extract the bounding boxes of each subplot within the image along with their corresponding titles. Returns a dictionary mapping each title to its subplot bounding box.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "The identifier of the image to analyze, e.g., 'img_1'"
                }
            },
            "required": ["image"]
        }
    }
}
'''

highlight_box_instruction = '''
{
    "type": "function",
    "function": {
        "name": "HighlightBox",
        "description": 
            "Highlight specified bounding box regions in the image using semi-transparent red overlays. Returns the edited image in base64 format.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "The identifier of the image to edit, e.g., 'img_1'"
                },
                "bboxes": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Bounding box in the format [x_min, y_min, x_max, y_max] using absolute pixel coordinates."
                    },
                    "description": "List of bounding boxes to be highlighted."
                }
            },
            "required": ["image", "bboxes"]
        }
    }
}
'''

mask_box_instruction = '''
{
    "type": "function",
    "function": {
        "name": "MaskBox",
        "description": 
            "Mask out all specified bounding box regions in the input image by overlaying white rectangles. Returns the edited image in base64 format.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "The identifier of the image to edit, e.g., 'img_1'."
                },
                "bboxes": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Bounding box in the format [x_min, y_min, x_max, y_max] using absolute pixel coordinates."
                    },
                    "description": "List of bounding boxes to be masked."
                }
            },
            "required": ["image", "bboxes"]
        }
    }
}
'''

languagemodel_instruction = '''
{
    "type": "function",
    "function": {
        "name": "LanguageModel",
        "description": 
            "Reason based on the prompt and image, and return the Language Model's inference text.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "The identifier of the image to analyze, e.g., 'img_1'."
                },
                "prompt": {
                    "type": "string",
                    "description": "The prompt to analyze."
                }
            },
            "required": ["image", "prompt"]
        }
    }
}
'''

crop_instruction = '''
{
    "type": "function",
    "function": {
        "name": "Crop",
        "description": "Crop an image using specified bounding box coordinates. This tool returns the cropped image in base64 format along with its dimensions.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "The identifier of the image to crop, e.g., 'img_1'."
                },
                "coordinates": {
                    "type": "string",
                    "description": "Coordinates in format '[x_min, y_min, x_max, y_max]', eg., '[100, 100, 200, 200]'. Only absolute pixel values (integers) are supported."
                }
            },
            "required": ["image", "coordinates"]
        }
    }
}
'''
tool_desc_dict = dict(
    OCR=ocr_instruction,
    Point=point_instruction,
    SegmentRegionAroundPoint=segment_around_point_instruction,
    DrawLine=drawn_line_instruction,
    GroundingDINO=grounding_dino_instruction,
    DrawShape=draw_shape_instruction,
    GetBarInfo=get_bar_info_instruction,
    GetSubplotInfo=get_subplot_info_instruction,
    HighlightBox=highlight_box_instruction,
    MaskBox=mask_box_instruction,
    Crop=crop_instruction,
    all=f"{ocr_instruction}\n{point_instruction}\n{segment_around_point_instruction}\n{drawn_line_instruction}\n{grounding_dino_instruction}\n{draw_shape_instruction}\n{get_bar_info_instruction}\n{get_subplot_info_instruction}\n{highlight_box_instruction}\n{mask_box_instruction}\n{crop_instruction}",
)

# 格式化 tool_planning_model_prompt
# 在vllm_models中被使用
tool_planning_model_prompt_one_tool_call = tool_planning_model_prompt_one_tool_call.format(tool_list=tool_desc_dict['all'])
tool_planning_model_prompt_no_tool_call = tool_planning_model_prompt_no_tool_call
