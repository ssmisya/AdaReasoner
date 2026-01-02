pixelreasonersft_grounding_crop="""
这是一个调用工具的SFT Data,请你帮我改造这个data，它用了zoomin 和 select_frame两个工具，现在请你帮我改造这个数据中调用工具的逻辑为我们的工具调用接口和逻辑，我们的工具调用协议为：
You are a visual assistant capable of solving visual reasoning problems. You can rely on your own capabilities or use external tools to assist in solving. 

Available Tools  
In your response, you can use the following tools:  

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


Steps for Each Turn  
1. Think: Recall relevant context and analyze the current user goal.  
2. Decide on Tool Usage: If a tool is needed, specify the tool and its parameters.  
3. Respond Appropriately: If a response is needed, generate one while maintaining consistency across user queries.

Output Format  
<think> Your thoughts and reasoning </think>  
<tool_call>  
{"name": "Tool name", "parameters": {"Parameter name": "Parameter content", "…": "…"}}  
</tool_call>  
<response> Your final response </response>

Important Notes  
1. You must always include the <think> field to outline your reasoning. Provide one of <tool_call> or <response>. You must not include both <tool_call> and <response> in the same turn because they are mutually exclusive. If tool usage is required, you must instead include both <think> and <tool_call>, and omit <response> for that turn. If no further tool usage is required and ready to answer the user's question, you can then use <think> to summarize your reasoning and include <response> with your final answer, and this indicates the ends of the conversation.

2. You can only invoke a single tool call at a time in the <tool_call> fields. The tool call should be a JSON object with a "name" field and a "parameters" field containing a dictionary of parameters. If no parameters are needed, leave the "parameters" field an empty dictionary. All images have their coordinate origin at the top-left corner.

3. Some tools require image input. You do not need to generate or upload the actual image data—simply refer to an image using a placeholder in the form of "img_n". There may be multiple images present in the dialogue. Besides the original image, additional images may appear as a result of prior tool calls (e.g., edited images returned by visual editing tools). You are free to select which image to use as input for the next tool.
The index "n" in "img_n" refers to the image's position in the dialogue history:
- The original image is always referred to as "img_1".
- Each subsequent image, including those returned from tools, is assigned "img_2", "img_3", and so on, in the order they appear in the dialogue.
For example:{"parameters": {"image": "img_1", "other_params": "other_values"}}
4. All image coordinates used must be in absolute pixel values, not relative or normalized coordinates. 

其中改写的具体要求为：
1. 忽略全部的针对视频的数据，忽略select_frame这个工具
2. 将所有的zoomin 工具的逻辑改造为：先调用groundingDINO获取想要聚焦的物体(可选)，再调用crop将其crop出来，其中groundingDINO的返回格式为：{
                "tool_response_from": self.model_name,
                "status": "success",
                "detections": [{
                    "label": label of object,
                    "confidence": confidence score,
                    "bbox": {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max
                    }
                }
],
                "image_dimensions_pixels": {
                    "width": w,
                    "height": h
                },
                "edited_image": annotated_image_base64,
                "message": f"Successfully detected {len(detections)} objects.”,
		"error_code": 0
            }
crop的返回格式为：
{
                        "tool_response_from": self.model_name,
                        "status": "success",
                        "edited_image": img_str,
                        "message": f"Image cropped successfully using absolute coordinates.",
                        "image_dimensions_pixels": {
                            "width": cropped_image.width,
                            "height": cropped_image.height
                        },
                        "error_code":0
                    }

3. 你需要生成全部的内容，包括模型和工具的输出，其中全部的生成图片的内容，例如groundingdino和crop之后的内容，可以直接用原有的图片路径代替
4. 请用content:[{“text”:”xxx”},{“image”:”xxx”}]的方式来组织对话中的消息格式，例如：{'role': 'user','content': [{'text': 'a question'}, {'image': 'images/0-0.jpg'}]}
5. 不要用这种tool call:{
                "type": "tool_call",
                "tool_call": {
                    "name": "GetBarInfo",
                    "parameters": {
                        "image": "img_1"
                    }
                }
            }
        ]
    },
而是要把它利用<tool_call>  
{"name": "Tool name", "parameters": {"Parameter name": "Parameter content", "…": "…"}}  
</tool_call>  的方式和<think></think>等放在一起
6. 你无需生成system_prompt, 对话的role只有user和assistant
"""

pixelreasonersft_crop = """
这是一个调用工具的SFT Data,请你帮我改造这个data，它用了zoomin 和 select_frame两个工具，现在请你帮我改造这个数据中调用工具的逻辑为我们的工具调用接口和逻辑，我们的工具调用协议为：
You are a visual assistant capable of solving visual reasoning problems. You can rely on your own capabilities or use external tools to assist in solving. 

Available Tools  
In your response, you can use the following tools:  

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


Steps for Each Turn  
1. Think: Recall relevant context and analyze the current user goal.  
2. Decide on Tool Usage: If a tool is needed, specify the tool and its parameters.  
3. Respond Appropriately: If a response is needed, generate one while maintaining consistency across user queries.

Output Format  
<think> Your thoughts and reasoning </think>  
<tool_call>  
{"name": "Tool name", "parameters": {"Parameter name": "Parameter content", "…": "…"}}  
</tool_call>  
<response> Your final response </response>

Important Notes  
1. You must always include the <think> field to outline your reasoning. Provide one of <tool_call> or <response>. You must not include both <tool_call> and <response> in the same turn because they are mutually exclusive. If tool usage is required, you must instead include both <think> and <tool_call>, and omit <response> for that turn. If no further tool usage is required and ready to answer the user's question, you can then use <think> to summarize your reasoning and include <response> with your final answer, and this indicates the ends of the conversation.

2. You can only invoke a single tool call at a time in the <tool_call> fields. The tool call should be a JSON object with a "name" field and a "parameters" field containing a dictionary of parameters. If no parameters are needed, leave the "parameters" field an empty dictionary. All images have their coordinate origin at the top-left corner.

3. Some tools require image input. You do not need to generate or upload the actual image data—simply refer to an image using a placeholder in the form of "img_n". There may be multiple images present in the dialogue. Besides the original image, additional images may appear as a result of prior tool calls (e.g., edited images returned by visual editing tools). You are free to select which image to use as input for the next tool.
The index "n" in "img_n" refers to the image's position in the dialogue history:
- The original image is always referred to as "img_1".
- Each subsequent image, including those returned from tools, is assigned "img_2", "img_3", and so on, in the order they appear in the dialogue.
For example:{"parameters": {"image": "img_1", "other_params": "other_values"}}
4. All image coordinates used must be in absolute pixel values, not relative or normalized coordinates. 

其中改写的具体要求为：
1. 忽略全部的针对视频的数据，忽略select_frame这个工具
2. 将所有的zoomin 工具的逻辑改造为我们的调用逻辑，即先思考要crop的坐标，再调用crop将其crop出来，其中crop的返回格式为：
{
    "tool_response_from": self.model_name,
    "status": "success",
    "edited_image": img_str,
    "message": f"Image cropped successfully using absolute coordinates.",
    "image_dimensions_pixels": {
        "width": cropped_image.width,
        "height": cropped_image.height
    },
    "error_code":0
}

3. 你需要生成全部的内容，包括模型和工具的输出，其中全部的生成图片的内容，例如groundingdino和crop之后的内容，可以直接用原有的图片路径代替
4. 请用content:[{“text”:”xxx”},{“image”:”xxx”}]的方式来组织对话中的消息格式，例如：{'role': 'user','content': [{'text': 'a question'}, {'image': 'images/0-0.jpg'}]}
5. 不要用这种tool call:{
                "type": "tool_call",
                "tool_call": {
                    "name": "GetBarInfo",
                    "parameters": {
                        "image": "img_1"
                    }
                }
            }
        ]
    },
而是要把它利用<tool_call>  
{"name": "Tool name", "parameters": {"Parameter name": "Parameter content", "…": "…"}}  
</tool_call>  的方式和<think></think>等放在一起
6. 你无需生成system_prompt, 对话的role只有user和assistant

"""

refocus_bar_highlight = """
这是一个调用工具的SFT Data，现在请你帮我改造这个数据中调用工具的逻辑为我们的工具调用接口和逻辑，他们的工具调用详细信息会在给定数据的第一句系统调用中出现，我们的工具调用协议为：

You are a visual assistant capable of solving visual reasoning problems. You can rely on your own capabilities or use external tools to assist in solving. 

Available Tools  
In your response, you can use the following tools:  

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


Steps for Each Turn  
1. Think: Recall relevant context and analyze the current user goal.  
2. Decide on Tool Usage: If a tool is needed, specify the tool and its parameters.  
3. Respond Appropriately: If a response is needed, generate one while maintaining consistency across user queries.

Output Format  
<think> Your thoughts and reasoning </think>  
<tool_call>  
{"name": "Tool name", "parameters": {"Parameter name": "Parameter content", "…": "…"}}  
</tool_call>  
<response> Your final response </response>

Important Notes  
1. You must always include the <think> field to outline your reasoning. Provide one of <tool_call> or <response>. You must not include both <tool_call> and <response> in the same turn because they are mutually exclusive. If tool usage is required, you must instead include both <think> and <tool_call>, and omit <response> for that turn. If no further tool usage is required and ready to answer the user's question, you can then use <think> to summarize your reasoning and include <response> with your final answer, and this indicates the ends of the conversation.

2. You can only invoke a single tool call at a time in the <tool_call> fields. The tool call should be a JSON object with a "name" field and a "parameters" field containing a dictionary of parameters. If no parameters are needed, leave the "parameters" field an empty dictionary. All images have their coordinate origin at the top-left corner.

3. Some tools require image input. You do not need to generate or upload the actual image data—simply refer to an image using a placeholder in the form of "img_n". There may be multiple images present in the dialogue. Besides the original image, additional images may appear as a result of prior tool calls (e.g., edited images returned by visual editing tools). You are free to select which image to use as input for the next tool.
The index "n" in "img_n" refers to the image's position in the dialogue history:
- The original image is always referred to as "img_1".
- Each subsequent image, including those returned from tools, is assigned "img_2", "img_3", and so on, in the order they appear in the dialogue.
For example:{"parameters": {"image": "img_1", "other_params": "other_values"}}
4. All image coordinates used must be in absolute pixel values, not relative or normalized coordinates. 

其中改写的具体要求为：

1. 它输入数据中包含很多关键信息，例如图中柱状图的box和对应的标题，请你把这一步改成调用工具实现，我们的GetBarInfo
的返回格式为：
{
                "tool_response_from": GetBarInfo,
                "status": "success",
                "message": "Successfully extracted bar information",
                "bars": {"bar_label1": [x_min, y_min, x_max, y_max], "bar_label2": [x_min, y_min, x_max, y_max], ...},
                "error_code": 0
}
HighightBox的文字返回格式为：{
                "tool_response_from": “HighlightBox”,
                "status": "success",
                "edited_image": img_str,
                "message": f"Successfully highlighted {len(bboxes)} bounding boxes on the image.",
                "image_dimensions_pixels": {
                    "width": result_image.width,
                    "height": result_image.height
                },
                "error_code": 0
            }
MaskBox的文字返回格式为：
{
                    "tool_response_from": “MaskBox”,
                    "status": "success",
                    "edited_image": img_str,
                    "message": f"Successfully masked {len(bboxes)} regions in the image.",
                    "image_dimensions_pixels": {
                        "width": draw_image.width,
                        "height": draw_image.height
                    },
                    "error_code": 0
                }
DrawShape的文字返回格式为：{
                "tool_response_from": “DrawShape”,
                "status": "success",
                "edited_image": img_str,
                "message": f"Successfully drew {len(bboxes)} shapes on the image.",
                "image_dimensions_pixels": {
                    "width": image.width,
                    "height": image.height
                },
                "error_code": 0
            }


2. 你需要生成全部的内容，包括模型和工具的输出，其中全部的图片输入，包括tool生成的图片，请用img_x作为占位符代替，例如原图片是img_1，工具编辑过的第一个图片为img_2，以此类推。
3. 所有工具的输出，除了要输出给定的文字输出，还要输出编辑后的图片(用刚才说的占位符)，例如content:[{“text”:”xxx”},{“image”:”xxx”]
4. 请用content:[{“text”:”xxx”},{“image”:”xxx”}]的方式来组织对话中的消息格式，例如：{'role': 'user','content': [{'text': 'a question'}, {'image': 'images/0-0.jpg'}]} 
5. 不要用这种tool call:{
                "type": "tool_call",
                "tool_call": {
                    "name": "GetBarInfo",
                    "parameters": {
                        "image": "img_1"
                    }
                }
            }
        ]
    },
而是要把它利用<tool_call>  
{"name": "Tool name", "parameters": {"Parameter name": "Parameter content", "…": "…"}}  
</tool_call>  的方式和<think></think>等放在一起
6. 我们会给出原对话和这个图片的详细信息，包括bboxes，groundtruths，请你利用信息里提供的内容补全对话中可能缺失的部分。
7. 你无需生成system_prompt, 对话的role只有user和assistant

Conversation:
"""

refocus_selfbar_highlight = """
这是一个调用工具的SFT Data，现在请你帮我改造这个数据中调用工具的逻辑为我们的工具调用接口和逻辑，其中我们的工具调用协议为：

You are a visual assistant capable of solving visual reasoning problems. You can rely on your own capabilities or use external tools to assist in solving. 

Available Tools  
In your response, you can use the following tools:  

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


Steps for Each Turn  
1. Think: Recall relevant context and analyze the current user goal.  
2. Decide on Tool Usage: If a tool is needed, specify the tool and its parameters.  
3. Respond Appropriately: If a response is needed, generate one while maintaining consistency across user queries.

Output Format  
<think> Your thoughts and reasoning </think>  
<tool_call>  
{"name": "Tool name", "parameters": {"Parameter name": "Parameter content", "…": "…"}}  
</tool_call>  
<response> Your final response </response>

Important Notes  
1. You must always include the <think> field to outline your reasoning. Provide one of <tool_call> or <response>. You must not include both <tool_call> and <response> in the same turn because they are mutually exclusive. If tool usage is required, you must instead include both <think> and <tool_call>, and omit <response> for that turn. If no further tool usage is required and ready to answer the user's question, you can then use <think> to summarize your reasoning and include <response> with your final answer, and this indicates the ends of the conversation.

2. You can only invoke a single tool call at a time in the <tool_call> fields. The tool call should be a JSON object with a "name" field and a "parameters" field containing a dictionary of parameters. If no parameters are needed, leave the "parameters" field an empty dictionary. All images have their coordinate origin at the top-left corner.

3. Some tools require image input. You do not need to generate or upload the actual image data—simply refer to an image using a placeholder in the form of "img_n". There may be multiple images present in the dialogue. Besides the original image, additional images may appear as a result of prior tool calls (e.g., edited images returned by visual editing tools). You are free to select which image to use as input for the next tool.
The index "n" in "img_n" refers to the image's position in the dialogue history:
- The original image is always referred to as "img_1".
- Each subsequent image, including those returned from tools, is assigned "img_2", "img_3", and so on, in the order they appear in the dialogue.
For example:{"parameters": {"image": "img_1", "other_params": "other_values"}}
4. All image coordinates used must be in absolute pixel values, not relative or normalized coordinates. 

其中改写的具体要求为：

1. 它原本只会告诉工具它要关注highlight或mask掉哪个标题所对应的柱状图，但现在要求planning model自己思考确定需要highlight或mask掉的柱状图的标题和对应的box，然后再调用我们的highlight或mask工具进行操作，请你帮我改写这一逻辑。也就是说模型需要自己思考要操作的box坐标，而不通过任何工具比如getbarInfo获取到这个坐标。

其中HighightBox的文字返回格式为：{
                "tool_response_from": “HighlightBox”,
                "status": "success",
                "edited_image": img_str,
                "message": f"Successfully highlighted {len(bboxes)} bounding boxes on the image.",
                "image_dimensions_pixels": {
                    "width": result_image.width,
                    "height": result_image.height
                },
                "error_code": 0
            }
MaskBox的文字返回格式为：
{
                    "tool_response_from": “MaskBox”,
                    "status": "success",
                    "edited_image": img_str,
                    "message": f"Successfully masked {len(bboxes)} regions in the image.",
                    "image_dimensions_pixels": {
                        "width": draw_image.width,
                        "height": draw_image.height
                    },
                    "error_code": 0
                }
DrawShape的文字返回格式为：{
                "tool_response_from": “DrawShape”,
                "status": "success",
                "edited_image": img_str,
                "message": f"Successfully drew {len(bboxes)} shapes on the image.",
                "image_dimensions_pixels": {
                    "width": image.width,
                    "height": image.height
                },
                "error_code": 0
            }


2. 你需要生成全部的内容，包括模型和工具的输出，其中全部的图片输入，包括tool生成的图片，请用img_x作为占位符代替，例如原图片是img_1，工具编辑过的第一个图片为img_2，以此类推。
3. 所有工具的输出，除了要输出给定的文字输出，还要输出编辑后的图片(用刚才说的占位符)，例如content:[{“text”:”xxx”},{“image”:”xxx”]
4. 请用content:[{“text”:”xxx”},{“image”:”xxx”}]的方式来组织对话中的消息格式，例如：{'role': 'user','content': [{'text': 'a question'}, {'image': 'images/0-0.jpg'}]} 
5. 不要用这种tool call:{
                "type": "tool_call",
                "tool_call": {
                    "name": "GetBarInfo",
                    "parameters": {
                        "image": "img_1"
                    }
                }
            }
        ]
    },
而是要把它利用<tool_call>  
{"name": "Tool name", "parameters": {"Parameter name": "Parameter content", "…": "…"}}  
</tool_call>  的方式和<think></think>等放在一起
6. 我们会给出原对话和这个图片的详细信息，包括bboxes，groundtruths，请你利用信息里提供的内容补全对话中可能缺失的部分。
7. 你无需生成system_prompt, 对话的role只有user和assistant

Conversation:
"""

prompt_dict = {
    "pixelreasonersft_grounding_crop": pixelreasonersft_grounding_crop,
    "pixelreasonersft_crop": pixelreasonersft_crop,
    "refocus_bar_highlight": refocus_bar_highlight,
    "refocus_selfbar_highlight": refocus_selfbar_highlight
}