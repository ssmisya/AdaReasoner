import os,sys
from tool_server.utils.prompts import tool_planning_model_prompt_one_tool_call
from tool_server.utils.utils import *
from prompts import prompt_dict
import random
from tqdm import tqdm
import os
from google import genai
from google.genai import types
from transformers import HfArgumentParser
from dataclasses import dataclass
from typing import Optional
from copy import deepcopy




@dataclass
class InferenceArguments:
    input_path: str = None
    output_path: str = None
    first_prompt: str = None
    model_name: str = "gemini-2.5-flash"  
    inference_type: str = "pixelreasoner"  # or "refocus"
    
class GeminiInferencer():
    def __init__(self, args):
        self.input_path = args.input_path
        self.output_path = args.output_path
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model_name = args.model_name
        assert self.api_key, "Please set GEMINI_API_KEY environment variable"
        
        self.first_prompt = prompt_dict[args.first_prompt]
        self.load_data()
        self.load_ckpt()
        
    
    def load_data(self):
        """
        Load the data from the input path.
        """
        raw_data = process_jsonl(self.input_path)
        self.meta_data = []
        for item in raw_data:
            message_list = item["message_list"]
            qid = item["qid"]
            new_idx = f"pixelreasonersft/{qid}"
            question_prompt = f"{message_list}"
            res = dict(question_prompt=question_prompt, qid=new_idx, origin_qid=qid)
            self.meta_data.append(res)
        print(f"Loaded {len(self.meta_data)} items.")
    
    def load_ckpt(self):
        if os.path.exists(self.output_path):
            ckpt = process_jsonl(self.output_path)
            qids = [item["qid"] for item in ckpt]
            renewed_meta_data = []
            for item in self.meta_data:
                qid = item["qid"]
                if qid not in qids:
                    renewed_meta_data.append(item)

            self.meta_data = renewed_meta_data
            print(f"Loaded CKPT, remaining {len(self.meta_data)} items to process.")

    def output_item(self, item):
        append_jsonl(item, self.output_path)
        
    def gemini_generate(
        self,
        first_prompt: str = None,
        question_prompt: str = None,
        api_key: str = os.environ.get("GEMINI_API_KEY"),  
        model = "gemini-2.5-flash",
    ):
        assert api_key, "Please set GEMINI_API_KEY environment variable"
        assert first_prompt, "first_prompt must be provided"
        assert question_prompt, "question_prompt must be provided"
        
        client = genai.Client(
            api_key=api_key,
        )

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=first_prompt),
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=question_prompt),
                ],
            ),
        ]
        # print(f"first_prompt: {first_prompt}")
        # print(f"question_prompt: {question_prompt}")
        generate_content_config = types.GenerateContentConfig(
            thinking_config = types.ThinkingConfig(
                thinking_budget=-1,
            ),
            response_mime_type="text/plain",
        )

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        
        response_text = response.text
        # print(f"Response text: {response_text}")
        return response_text
    
    def generate(self):
        for idx, item in enumerate(tqdm(self.meta_data, desc="Generating responses")):
            qid = item["qid"]
            question_prompt = item["question_prompt"]
            first_prompt = self.first_prompt
            # print(f"first_prompt: {first_prompt}")
            # print(f"question_prompt: {question_prompt}")
            
            response_text = self.gemini_generate(
                first_prompt=first_prompt,
                question_prompt=question_prompt,
                api_key=self.api_key,
                model=self.model_name,
            )
            # print(f"Response text: {response_text}")
            try:
                filtered_text = response_text.split("```json")[1].split("```")[0].strip()
                new_message_list = json.loads(filtered_text)
                new_item = deepcopy(item)
                new_item["response_messages"] = new_message_list
                self.output_item(new_item)
                
            except KeyboardInterrupt:
                print("KeyboardInterrupt received. Exiting...")
                sys.exit(0)
            except:
                print(f"Error processing item {idx}: {item}")
            

class RefocusGeminiInferencer(GeminiInferencer):
    def load_data(self):
        """
        Load the data from the input path.
        """
        raw_data = process_jsonl(self.input_path)
        self.meta_data = []
        for item in raw_data:
            message_list = item["message_list"]
            qid = item["qid"]
            extra_info = item["extra_info"]
            new_idx = f"refocus_chartqa_v_bar_wbb/{qid}"
            
            question_prompt = f"{message_list}\n\nInfo: {extra_info}"
            res = dict(question_prompt=question_prompt, qid=new_idx, origin_qid=qid)
            self.meta_data.append(res)
        print(f"Loaded {len(self.meta_data)} items.")
    
inferencer_dict = {
    "pixelreasoner": GeminiInferencer,
    "refocus": RefocusGeminiInferencer,
}

if __name__ == "__main__":
    random.seed(42)
    setup_openai_proxy()
    # setup_proxy()
    parser = HfArgumentParser(InferenceArguments)
    args = parser.parse_args_into_dataclasses()[0]
    inferencer_class = inferencer_dict[args.inference_type]
    
    inferencer = inferencer_class(args)
    inferencer.generate()