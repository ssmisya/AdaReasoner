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
"""

# 不使用工具的prompt
tool_planning_model_prompt_no_tool_call = '''
You are a visual assistant capable of solving visual reasoning problems. You can only rely on your own capabilities in solving. 

Steps for Each Turn  
1. Think: Recall relevant context and analyze the current user goal.   
2. Respond Appropriately: If a response is needed, generate one while maintaining consistency across user queries.

Output Format  
<think> Your thoughts and reasoning </think>  
<response> Your final response </response>

Important Notes  
1. You must always include the <think> field to outline your reasoning. 
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


policy_model_system_prompt = """
You are a visual assistant capable of solving visual reasoning problems. You can rely on your own capabilities or use external tools to assist in solving. Here are the available tools and their protocols:

{tool_desc}

Notes:
1. You can select actions from the provided tools list, combining them logically and building on previous steps. Call one action at a time, using its output for the next.
2. To use `SegmentRegionAroundPoint`, `DrawHorizontalLineByY`, or `DrawVerticalLineByX`, first call "Point" to get coordinates for further actions.
3. When you have the final answer, use the "Terminate" action to end the task and provide the answer. Be sure the answer is correct before terminating and the "Terminate" action should be the last action you call.
4. Your output should be in a strict JSON format as follows:
{"thought": "the reasoning process", "actions": [{"name": "the name of the tool you selected", "arguments": {"argument1": "value1", "argument2": "value2"}}]}
"""

ocr_instruction = """
{
    "type": "function",
    "function": {
        "name": self.model_name,
        "description": "Extracts and localizes text from the given image using OCR. Returns bounding boxes, recognized text, and confidence scores.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "The identifier of the image in which to locate the object, e.g., 'img_1'."
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
        "name": self.model_name,
        "description": "Identify a point in the image based on a natural language description. This tool returns the absolute pixel coordinates of the identified point along with an edited image showing the point. Only absolute coordinates are supported for output.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "The identifier or path of the image in which to locate the point, e.g., 'img_1'."
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
        "name": self.model_name,
        "description": "Segments objects in an image. Can perform automatic segmentation on the entire image or segment a specific object based on a single designated point. Returns the image with segmentation masks and related processing info.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "The identifier of the image in which to locate the object, e.g., 'img_1'."
                },
                "description": {
                    "type": "string",
                    "description": "Optional: Single point coordinates in format 'x=value1, y=value2', eg., 'x=50, y=100'. Using absolute pixel coordinates within image bounds. If not provided, the tool will automatically segment all objects in the image."
                }
            },
            "required": ["image", "description"]
        }
    }
}
'''

drawn_line_instruction = '''
{
    "type": "function",
    "function": {
        "name": self.model_name,
        "description": "Draw horizontal or vertical lines on an image. This tool supports drawing multiple lines of the same type simultaneously. Only accepts absolute pixel coordinates (not normalized values). Returns base64 encoded image with lines drawn.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "The identifier of the image in which to locate the object, e.g., 'img_1'."
                },
                "line_type": {
                    "type": "string",
                    "description": "Type of line to draw: 'horizontal' (requires y coordinates) or 'vertical' (requires x coordinates).",
                    "enum": ["horizontal", "vertical"]
                },
                "description": {
                    "type": "string",
                    "description": "For horizontal lines, provide y coordinates in format 'y1=100, y2=200, y3=300'. For vertical lines, provide x coordinates in format 'x1=100, x2=200, x3=300'. Multiple coordinates should be separated by commas. Only absolute pixel values are supported."
                }
            },
            "required": ["image", "line_type", "description"]
        }
    }
}
'''

grounding_dino_instruction = '''
{
    "type": "function",
    "function": {
        "name": self.model_name,
        "description": "Locate objects in the image based on a natural language description. Returns detected objects with their bounding boxes in absolute pixel coordinates, confidence scores, and an annotated image with visualized detections.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "The identifier of the image in which to locate the object, e.g., 'img_1'."
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


tool_desc_dict = dict(
   OCR=ocr_instruction,
   Point=point_instruction,
   SegmentRegionAroundPoint=segment_around_point_instruction,
   DrawLine=drawn_line_instruction,
   GroundingDINO=grounding_dino_instruction,
#    Terminate=terminate_instruction, # 根据prompt应该就不需要这个东西了吧
   all=f"{ocr_instruction}\n{point_instruction}\n{segment_around_point_instruction}\n{drawn_line_instruction}\n{grounding_dino_instruction}",
)

# 格式化 tool_planning_model_prompt
# 在vllm_models中被使用
tool_planning_model_prompt_one_tool_call = tool_planning_model_prompt_one_tool_call.format(tool_list=tool_desc_dict['all'])
tool_planning_model_prompt_no_tool_call = tool_planning_model_prompt_no_tool_call



# print(tool_planning_model_prompt)





# ocr_instruction = """
# OCR: 
# Extracts all texts from an image. 
# Input: The image that you want to apply OCR. 
# Output: All texts that appear on the input image. 
# Format Requirement: `{"name": "OCR", "arguments": {"image": "img_1"}}`
# Example: `{"name": "OCR", "arguments": {"image": "img_1"}}`
# """

# point_instruction = """
# Point:
# Identifies a point in the image based on description and returns coordinates. 
# Input: The image that you want to point at and the description of the point location.
# Output: Coordinates of the point in the image.
# Format Requirement: `{"name": "Point", "arguments": {"image": "img_1", "param": "The description that you want to point at."}}`
# Example: `{"name": "Point", "arguments": {"image": "img_1", "param": "A dog."}}`
# """

# segment_around_point_instruction = """
# SegmentRegionAroundPoint:
# Segments a region around a given point. 
# Input: The image that you want to segement and the point coordinate of the point that you want to segment near.
# Output: The segemented image.
# Format Requirement: `{"name": "SegmentRegionAroundPoint", "arguments": {"image": "img_1", "param": "x=the x coordinate of the point y=the y coordinate of the point"}}`
# Example: `{"name": "SegmentRegionAroundPoint", "arguments": {"image": "img_1", "param": "x=21.5 y=28.5"}}`
# """

# drawhorizontal_line_instruction = """
# DrawHorizontalLineByY:
# Draws a horizontal line at a given y-coordinate.
# Input: The image that you want to draw a horizontal line on and the y-coordinate of the line.
# Output: The image with a horizontal line drawn.
# Format Requirement: `{"name": "DrawHorizontalLineByY", "arguments": {"image": "img_1", "param": "y=the y coordinate of the line"}}`
# Example: `{"name": "DrawHorizontalLineByY", "arguments": {"image": "img_1", "param": "y=28.5"}}`
# """

# drawvertical_line_instruction = """
# DrawVerticalLineByX:
# Draws a vertical line at a given x-coordinate.
# Input: The image that you want to draw a vertical line on and the x-coordinate of the line.
# Output: The image with a vertical line drawn.
# Format Requirement: `{"name": "DrawVerticalLineByX", "arguments": {"image": "img_1", "param": "x=the x coordinate of the line"}}`
# Example: `{"name": "DrawVerticalLineByX", "arguments": {"image": "img_1", "param": "x=21.5"}}`
# """

# grounding_dino_instruction = """
# GroundingDINO:
# Locates objects in the image based on a description.
# Input: The image that you want to locate objects in and the description of the object.
# Output: The coordinates of the located object in the image.
# Format Requirement: `{"name": "GroundingDINO", "arguments": {"image": "img_1", "param": "The description of the object you want to locate."}}`
# Example: `{"name": "GroundingDINO", "arguments": {"image": "img_1", "param": "A Dog"}}`
# """

# terminate_instruction = """
# Terminate:
# Ends the task and provides the final answer.
# Input: The final answer to the task.
# Output: No outputs.
# Format Requirement: `{"name": "Terminate", "arguments": {"ans": "The final answer"}}`
# Example: `{"name": "Terminate", "arguments": {"ans": "1985"}}`
# """


