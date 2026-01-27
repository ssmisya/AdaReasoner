import argparse
import json
import gradio as gr
from typing import List, Dict, Any, Optional, Iterator
import copy
from PIL import Image
import logging
import re
import io
import base64

# Mock imports for dependency check (Ensure you have your original imports)
# Assuming these exist in your environment as per your provided code
from tool_server.tf_eval.models.vllm_models import VllmModels
from tool_server.tf_eval.models.openai import OpenaiModels
from tool_server.tool_workers.tool_manager.base_manager_randomize import ToolManager
from tool_server.utils.utils import load_image, pil_to_base64, base64_to_pil
from tool_server.utils.debug import remote_breakpoint
from vllm import SamplingParams
from tool_server.tf_eval.tool_inferencer.dynamic_batch_manager import DynamicBatchItem
import textwrap

# Initialize logger
logger = logging.getLogger("gradio_demo")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Default configuration
DEFAULT_MODEL_PATH = "/mnt/petrelfs/songmingyang/songmingyang/runs/tool_factory/rl/v2/tool_rl/test_ckpts/unified_all_randomized_sft_randomized_rl_4tasks_7b/global_step_250/actor/huggingface"
DEFAULT_CONTROLLER_URL = "http://SH-IDC1-10-140-37-45:21112"
DEFAULT_TOOLS = "Point,Draw2DPath,AStarWithPixelCoordinate,DetectBlackArea,InsertImage,OCR,Crop".split(",")

def pil_to_html(img: Image.Image, max_width: int = 400) -> str:
    """Helper to convert PIL image to HTML tag for inline display"""
    try:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f'<img src="data:image/png;base64,{img_str}" style="max-width: {max_width}px; border-radius: 8px; margin: 10px 0;" />'
    except Exception as e:
        return f"[Image Error: {e}]"

class StreamingInferencer:
    """Streaming inferencer with tool calling"""
    
    def __init__(
        self,
        model_path: str,
        controller_url: str,
        tools: List[str],
        tensor_parallel: int = 1,
        limit_mm_per_prompt: int = 10,
        max_rounds: int = 20,
        use_api: bool = False,
        api_model_name: str = "gpt-4o",
    ):
        self.use_api = use_api
        self.max_rounds = max_rounds
        
        # Initialize model (local or API)
        if use_api:
            logger.info(f"Using API model: {api_model_name}")
            self.model = OpenaiModels(
                pretrained=api_model_name,
                temperature=0.0,
                max_retry=2,
            )
        else:
            logger.info(f"Using local VLLM model: {model_path}")
            self.model = VllmModels(
                pretrained=model_path,
                tensor_parallel=str(tensor_parallel),
                limit_mm_per_prompt=str(limit_mm_per_prompt),
            )
        
        # Initialize tool manager
        self.tool_manager = ToolManager(
            controller_url_location=controller_url,
            tools=tools,
            randomize=False
        )
        
        # Get system prompt
        system_prompt = self.tool_manager.get_tool_prompt(prompt_type="one_tool_call")
        self.model.system_prompt = system_prompt
        
        self.available_tools = self.tool_manager.available_tools
        self.image_keys = ["image", "base_image", "image_to_insert"]
        # remote_breakpoint(port=7119)
        logger.info(f"Model initialized with tools: {self.available_tools}")
    
    def extract_think_and_response(self, text: str) -> Dict[str, str]:
        think = ""
        response = ""
        
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match:
            think = think_match.group(1).strip()
        
        response_match = re.search(r'<response>(.*?)</response>', text, re.DOTALL)
        if response_match:
            response = response_match.group(1).strip()
        
        # Fallback if no tags found, treat everything as response if valid
        if not think and not response:
            if "<tool_call>" not in text:
                response = text
            else:
                # If only tool call exists, response might be empty
                pass

        return {"think": think, "response": response}
    
    def extract_tool_call(self, text: str) -> Optional[List[Dict]]:
        try:
            tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
            tool_call_match = re.search(tool_call_pattern, text, re.DOTALL)
            
            if not tool_call_match:
                return None
                
            tool_call_content = tool_call_match.group(1).strip()
            
            # Handle potential JSON parsing errors more gracefully
            if tool_call_content.startswith('[') and tool_call_content.endswith(']'):
                json_array = json.loads(tool_call_content)
                if isinstance(json_array, list):
                    return [obj for obj in json_array if isinstance(obj, dict) and "name" in obj and "parameters" in obj]
            
            if tool_call_content.startswith('{') and tool_call_content.endswith('}'):
                json_obj = json.loads(tool_call_content)
                if "name" in json_obj and "parameters" in json_obj:
                    return [json_obj]
                    
            return None
            
        except Exception as e:
            logger.error(f"Error extracting tool call: {e}")
            return None
    
    def call_tool(self, tool_name: str, tool_params: Dict, image_history: Dict) -> Dict:
        try:
            for image_key in self.image_keys:
                if image_key in tool_params:
                    img_key = tool_params[image_key]
                    image = image_history.get(img_key, None)
                    if image is not None:
                        image = load_image(image)
                        image = pil_to_base64(image)
                        tool_params[image_key] = image
                    else:
                        return {"status": "failed", "text": f"Image {img_key} not found."}
            
            tool_response = self.tool_manager.call_tool(tool_name, tool_params)
            return tool_response
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return {"status": "failed", "text": f"Tool {tool_name} failed: {str(e)}"}
    
    def stream_generate(self, conversation: List[Dict], temperature: float = 0.0, max_tokens: int = 2048) -> Iterator[str]:
        if self.use_api:
            batch_item = DynamicBatchItem(
                max_rounds=self.max_rounds,
                current_round=0,
                conversation=conversation,
                image_history={},
                status="processing"
            )
            self.model.generate([batch_item])
            full_text = batch_item.model_response[-1]
            words = re.split(r'(\s+)', full_text) # Split keeping delimiters
            current_text = ""
            for word in words:
                current_text += word
                yield current_text
        else:
            sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
            outputs = self.model.model.chat([conversation], sampling_params, use_tqdm=False)
            full_text = outputs[0].outputs[0].text
            # Simulating streaming for local just so UI updates progressively
            words = re.split(r'(\s+)', full_text)
            current_text = ""
            for word in words:
                current_text += word
                yield current_text

    # ==========================================
    # 核心修改：inference_multiturn
    # ==========================================
    def inference_multiturn(
        self,
        text_input: str,
        images: List[Any],
        max_rounds: int,
        temperature: float,
    ) -> Iterator[List[List[str]]]:
        
        if not text_input or not text_input.strip():
            yield [["", "Please enter a question."]]
            return
        
        # 1. 处理输入图片
        image_history = {}
        processed_images_html = ""
        processed_images_pil = []
        
        if images:
            for idx, img in enumerate(images):
                # 兼容不同的 Gradio 版本图片格式
                if isinstance(img, tuple):
                    img_pil = img[0]
                elif isinstance(img, dict):
                    img_pil = img['image']
                else:
                    img_pil = img
                
                img_pil = load_image(img_pil)
                img_base64 = pil_to_base64(img_pil)
                img_key = f"img_{idx+1}"
                image_history[img_key] = img_base64
                processed_images_pil.append((img_pil, img_key))
                
                # 图片直接显示，不缩进
                processed_images_html += f'<br><b>[{img_key}]</b><br>{pil_to_html(img_pil, max_width=300)}'

        user_message_html = f"{text_input}{processed_images_html}"
        
        # 生成初始对话
        conversation = self.model.generate_conversation_fn(
            text=text_input,
            images=[pil_to_base64(img[0]) for img in processed_images_pil],
            role="user"
        )
        
        chat_history = [[user_message_html, ""]] 
        yield chat_history
        
        bot_response_html = ""
        
        for round_idx in range(max_rounds):
            logger.info(f"Starting round {round_idx + 1}")
            
            # 使用 textwrap.dedent 去除缩进，防止 Markdown 识别为代码块
            # 同时也去掉了字符串开头的所有空格
            round_header = textwrap.dedent(f"""
                <div style="border-left: 3px solid #667eea; padding-left: 10px; margin-bottom: 20px;">
                    <p style="color: #667eea; font-weight: bold; margin: 0 0 5px 0;">🔄 Round {round_idx + 1}</p>
            """).strip()
            
            current_round_content = ""
            full_response = ""
            
            # === 流式生成 ===
            for partial_response in self.stream_generate(conversation, temperature):
                full_response = partial_response
                parsed = self.extract_think_and_response(full_response)
                tool_calls = self.extract_tool_call(full_response)
                
                stream_display = ""
                
                # 思考过程 (正在生成时默认展开)
                if parsed["think"]:
                    # 注意：这里 HTML 紧贴左边，没有任何缩进
                    # stream_display += f"""<details open><summary style="cursor: pointer; color: #666;">💭 Thinking Process</summary><div style="background: #f5f5f5; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.9em; white-space: pre-wrap; margin-top: 5px;">{parsed['think']}</div></details>"""
                    stream_display += f"""<details open><summary style="cursor: pointer; color: #666;">💭 Thinking Process</summary><div style="background: #f5f5f5; color: #333; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.9em; white-space: pre-wrap; margin-top: 5px;">{parsed['think']}</div></details>"""
                if tool_calls:
                     stream_display += f"""<div style="margin-top: 10px; color: #d97706; font-weight: bold;">🛠️ Planning Tool Call...</div>"""
                
                if not parsed["think"] and not tool_calls and parsed["response"]:
                     stream_display += f"""<div style="margin-top:10px;">{parsed['response']}</div>"""
                elif not parsed["think"] and not tool_calls:
                     stream_display += f"""<div style="opacity: 0.7;">{full_response}</div>"""

                # 拼接并更新
                chat_history[-1][1] = bot_response_html + round_header + stream_display + "</div>"
                yield chat_history
            
            # === 本轮生成结束，整理格式 ===
            parsed = self.extract_think_and_response(full_response)
            tool_calls = self.extract_tool_call(full_response)
            
            final_round_display = ""
            
            if parsed["think"]:
                # 生成结束后，默认折叠 (details 不带 open)

                final_round_display += f"""<details open><summary style="cursor: pointer; color: #555;">💭 Thinking Process (Click to expand)</summary><div style="background: #f0f2f5; color: #333; padding: 10px; border-radius: 5px; margin-top: 5px; font-family: monospace; font-size: 0.85em; white-space: pre-wrap;">{parsed['think']}</div></details>"""
            
            if tool_calls:
                tool_call = tool_calls[0]
                tool_json = json.dumps(tool_call, indent=2, ensure_ascii=False)
                # 使用 pre 标签但要注意内部内容不要被自动缩进影响
                final_round_display += f"""<details open><summary style="cursor: pointer; color: #d97706; font-weight: bold;">🛠️ Tool Call: {tool_call['name']}</summary><pre style="background: #fffbeb; color: #333; padding: 10px; border-radius: 5px; border: 1px solid #fcd34d; font-size: 0.85em; overflow-x: auto; margin-top: 5px;">{tool_json}</pre></details>"""
            elif parsed["response"]:
                final_round_display += f"""<div style="margin-top: 10px; font-size: 1.05em;">{parsed['response']}</div>"""
            
            current_round_content += final_round_display
            bot_response_html += round_header + current_round_content + "</div>"
            chat_history[-1][1] = bot_response_html
            yield chat_history

            conversation = self.model.append_conversation_fn(
                conversation=conversation,
                text=full_response,
                image=None,
                role="assistant"
            )

            # === 执行工具 ===
            if not tool_calls:
                break
            
            if round_idx >= max_rounds - 1:
                chat_history[-1][1] += "<div style='color: red; font-weight: bold; margin-top: 10px;'>⚠️ Max rounds reached!</div>"
                yield chat_history
                break
            
            # UI显示：正在执行
            tool_call = tool_calls[0]
            tool_name = tool_call["name"]
            tool_params = tool_call["parameters"]
            
            chat_history[-1][1] = bot_response_html + f"""<div style='margin-left: 15px; color: #666; margin-bottom: 5px;'><i>⚙️ Executing {tool_name}...</i></div>"""
            yield chat_history

            # 调用工具
            tool_response = self.call_tool(tool_name, tool_params, image_history)
            
            # 处理工具返回
            tool_response_copy = copy.deepcopy(tool_response)
            edited_image = None
            img_html = ""
            
            if "edited_image" in tool_response_copy:
                edited_image = tool_response_copy.pop("edited_image")
                img_idx = len(image_history) + 1
                img_key = f"img_{img_idx}"
                image_history[img_key] = edited_image
                tool_response_copy["new_image_key"] = img_key
                
                # 图片转 HTML
                pil_img = base64_to_pil(edited_image)
                img_html = f"<div style='margin-top:5px;'><b>🖼️ Generated: {img_key}</b><br>{pil_to_html(pil_img)}</div>"

            tool_res_json = json.dumps(tool_response_copy, indent=2, ensure_ascii=False)
            
            # 添加工具输出 HTML (无缩进)
            tool_output_html = f"""<div style="margin-left: 15px; margin-bottom: 10px;"><details open><summary style="cursor: pointer; color: #059669; font-weight: bold;">✅ Tool Output</summary><pre style="background: #ecfdf5; color: #333; padding: 10px; border-radius: 5px; border: 1px solid #6ee7b7; font-size: 0.85em; overflow-x: auto; margin-top: 5px;">{tool_res_json}</pre></details>{img_html}</div>"""
            
            bot_response_html += tool_output_html
            chat_history[-1][1] = bot_response_html
            yield chat_history

            # 更新对话历史
            if edited_image:
                conversation = self.model.append_conversation_fn(
                    conversation=conversation,
                    text=f"{tool_response_copy} New image saved as: {img_key}.",
                    image=edited_image,
                    role="user"
                )
            else:
                conversation = self.model.append_conversation_fn(
                    conversation=conversation,
                    text=str(tool_response_copy),
                    image=None,
                    role="user"
                )
        
        # 结束标记
        chat_history[-1][1] += "<div style='margin-top: 15px; padding-top: 10px; border-top: 1px dashed #ccc; color: #2563eb; font-weight: bold;'>🏁 Inference Complete</div>"
        yield chat_history
# 在 create_demo 函数中添加示例案例

def create_demo(
    model_path: str,
    controller_url: str,
    tools: List[str],
    use_api: bool = False,
    api_model_name: str = "gpt-4o",
    tensor_parallel: int = 2,
    share: bool = False,
    server_port: int = 7860,
):
    """Create Gradio demo interface"""
    
    inferencer = StreamingInferencer(
        model_path=model_path,
        controller_url=controller_url,
        tools=tools,
        tensor_parallel=tensor_parallel,
        use_api=use_api,
        api_model_name=api_model_name,
    )
    
    # ===== 定义示例案例 =====
    examples = [
        {
            "name": "VSP Navigation",
            "question": "You are a maze solver. Your goal is to guide a player from the start to the goal on a grid map while avoiding holes, using the fewest moves. The player can move one square at a time in the directions left (L), right (R), up (U), or down (D). Moving off the edge has no effect, and falling into a hole results in failure. Reaching the goal means success. Your final answer should be formatted as \\boxed{L,R,U,D}.\n\nPlease generate action plan for the input maze image.",
            "images": ["examples/nav1.png"],  # 需要准备示例图片
            "description": "Find a path from start to goal while avoiding obstacles."
        },
        {
            "name": "VSP Verify",
            "question": "Question:\nYou are a maze solver. Your goal is to guide a player from the start to the goal on a grid map while avoiding holes, using the fewest moves. The player can move one square at a time in the directions left (L), right (R), up (U), or down (D). Moving off the edge has no effect, and falling into a hole results in failure. Reaching the goal means success.\n\nNow please determine if the action sequence is safe for the given maze. Your final answer should be formatted as \\boxed{Yes} or \\boxed{No}.\n\nThe action sequence is:\n\nULULDRRRR",
            "images": ["examples/verify1.png"],
            "description": "Given a path, verify if is safe or not."
        },
        {
            "name": "Jigsaw Puzzle",
            "question": "Given the first image (img_1) with one part missing, can you tell which one of the second image (img_2) or the third image (img_3) is the missing part? Imagine which image would be more appropriate to place in the missing spot. You can also carefully observe and compare the edges of the images.\n\nSelect from the following choices.\n\n(A) The second image (img_2)\n(B) The third image (img_3)\n(C) The fourth image (img_4)\n\nYour final answer should be formatted as \\boxed{Your Choice}, for example, \\boxed{A} or \\boxed{B} or \\boxed{C}.",
            "images": ["examples/jigsaw1-1.jpg", "examples/jigsaw1-2.jpg", "examples/jigsaw1-3.jpg", "examples/jigsaw1-4.jpg"],
            "description": "Find the correct jigsaw piece to complete the puzzle."
        },
        {
            "name": "GUI Chat",
            "question": "Can I book an appointment for lash services on this website?",
            "images": ["examples/web1.png"],
            "description": "Answer GUI-related questions based on screenshot."
        },
        {
            "name": "Visual Reasoning",
            "question": "What is the item of furniture that the mobile phone is on?",
            "images": ["examples/vqa1.jpg",],
            "description": "Answer general VQA questions based on image."
        },
    ]
    
    # Custom CSS
    css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    #chatbot {
        height: 700px !important; 
        overflow-y: auto;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        background-color: #f9fafb;
    }
    footer {display: none !important;}
    """
    
    with gr.Blocks(css=css, title="Tool-Planning Model Demo", theme=gr.themes.Soft(primary_hue="indigo")) as demo:
        
        model_type = "API" if use_api else "Local VLLM"
        model_display_name = api_model_name if use_api else model_path.split("/")[-1]
        
        gr.Markdown(f"""
        # 🤖 AdaReasoner: Dynamic Tool Orchestration for Iterative Visual Reasoning
        <div style="font-size: 0.9em; color: #666;">
            <b>Type:</b> {model_type} &nbsp;|&nbsp; <b>Model:</b> {model_display_name}<br>
            <b>Tools:</b> {', '.join(tools)}
        </div>
        """)
        
        with gr.Row():
            # LEFT COLUMN: Inputs + Examples
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 📝 Input")
                
                text_input = gr.Textbox(
                    label="Your Question",
                    placeholder="E.g., Detect the black area in this image...",
                    lines=3
                )
                
                image_input = gr.Gallery(
                    label="Upload Images",
                    type="pil",
                    columns=2,
                    height=200,
                    object_fit="contain"
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 Run Inference", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                
                with gr.Accordion("⚙️ Settings", open=False):
                    max_rounds = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Max Rounds")
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Temperature")
                
                # ===== 示例案例区域 - 只创建按钮，不绑定事件 =====
                with gr.Accordion("💡 Example Cases", open=True):
                    gr.Markdown("Click on any example to load it:")
                    example_buttons = []
                    for i, example in enumerate(examples):
                        example_btn = gr.Button(
                            value=f"{example['name']}: {example['description']}",
                            size="sm",
                            variant="secondary"
                        )
                        example_buttons.append((example_btn, example))

            # RIGHT COLUMN: Chat Output - 先定义 chatbot
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="Agent Trace",
                    show_label=False,
                    bubble_full_width=False,
                    avatar_images=(None, "https://api.iconify.design/fluent-emoji:robot.svg")
                )
        
        # ===== 在所有组件定义之后，再绑定事件 =====
        
        # 绑定示例按钮事件
        def create_load_example_fn(ex):
            """创建加载示例的函数"""
            def load_example():
                imgs = []
                for img_path in ex["images"]:
                    try:
                        img = Image.open(img_path)
                        imgs.append(img)
                    except Exception as e:
                        logger.warning(f"Failed to load example image {img_path}: {e}")
                return ex["question"], imgs, []
            return load_example
        
        for example_btn, example in example_buttons:
            example_btn.click(
                fn=create_load_example_fn(example),
                inputs=[],
                outputs=[text_input, image_input, chatbot]
            )
        
        # 绑定主要按钮事件
        submit_btn.click(
            fn=inferencer.inference_multiturn,
            inputs=[text_input, image_input, max_rounds, temperature],
            outputs=[chatbot]
        )
        
        clear_btn.click(
            fn=lambda: ("", [], []),
            outputs=[text_input, image_input, chatbot]
        )
    
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--controller-url", type=str, default=DEFAULT_CONTROLLER_URL)
    parser.add_argument("--tools", type=str, default=",".join(DEFAULT_TOOLS))
    parser.add_argument("--tensor-parallel", type=int, default=2)
    parser.add_argument("--use-api", action="store_true", help="Use API model")
    parser.add_argument("--api-model-name", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-port", type=int, default=10022)
    
    args = parser.parse_args()
    # Ensure tool list logic is correct
    tool_list = args.tools.split(",") if args.tools else DEFAULT_TOOLS
    
    # Forcing API mode as per your snippet (remove in prod if needed)
    args.use_api = True 
    # args.share=True

    demo = create_demo(
        model_path=args.model_path,
        controller_url=args.controller_url,
        tools=tool_list,
        use_api=args.use_api,
        api_model_name=args.api_model_name,
        tensor_parallel=args.tensor_parallel,
        share=args.share,
        server_port=args.server_port
    )
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.server_port,
        share=args.share,
        allowed_paths=["."] # Allow local file serving if needed
    )