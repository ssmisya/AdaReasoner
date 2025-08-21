# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入必要的库
import os
import re
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint
from transformers import set_seed
logger = logging.getLogger(__name__)

# 导入数学验证相关模块
from math_verify import parse, verify
# 从trl库导入GRPO(Generative Reward Processed Optimization)相关组件
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
# 导入工具配置解析函数
from r1_v.open_r1.trainer.tool_generation import parse_tool_config
# 导入自定义的训练器
# 在__init__.py中from .tool_vllm_grpo_trainer import Qwen2VLGRPOVLLMTrainer as Qwen2VLGRPOToolVLLMTrainer
# 所以Qwen2VLGRPOToolVLLMTrainer就是tool_vllm_grpo_trainer.py中的Qwen2VLGRPOVLLMTrainer
# 但是在tool_vllm_grpo_trainer.py和vllm_grpo_trainer.py中都定义了名为Qwen2VLGRPOVLLMTrainer的类
from r1_v.open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLGRPOToolTrainer, Qwen2VLGRPOToolVLLMTrainer

'''
命令中的所有参数都是在GRPOconfig或者ModelConfig中定义的，有默认值，二者没有的都是在GRPOScriptArguments中定义的
ModelConfig的定义：https://huggingface.co/docs/trl/main/en/sft_trainer
'''

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    根据grpo的文档，除了reward_funcs，其他参数应该是新定义的，然后额外加了进去
    运行的命令当中一些其他参数都是在GRPOconfig中定义的，有默认值
    https://huggingface.co/docs/trl/grpo_trainer#trl.GRPOConfig
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy"], # "format","accuracy"
        metadata={"help": "奖励函数列表，可选值: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "图像的最大像素数"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "图像的最小像素数"},
    )
    use_tool: Optional[bool]  = field(
        default=False,
        metadata={"help": "是否使用工具训练器进行训练"},
    )
    query_key: Optional[str] = field(
        default="question",
    )
    controller_addr: Optional[str] = field(
        # shy修改的
        default="http://SH-IDCA1404-10-140-54-2:20001",
        metadata={"help": "控制器的地址"},
    )


def accuracy_reward(completions, solution, **kwargs):
    """
    奖励函数：检查完成内容是否正确，使用符号验证或精确字符串匹配
    
    Args:
        completions: 模型生成的完成内容
        solution: 标准答案
        **kwargs: 其他参数
    
    Returns:
        rewards: 奖励值列表，正确为1.0，错误为0.0
    """

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    output_texts = kwargs.get("model_output_texts", None)
    
    # 获取更新后的内容
    if output_texts:
        renewed_contents = []
        for output_text in output_texts:
            renewed_contents.append(output_text[-1])
    else:
        renewed_contents = contents
        
    for item, sol in zip(output_texts, solution):
        content = item[-1] # 这里为什么要取最后一个，不懂：因为item是包含所有对话的列表，最后一个包含模型生成的答案
        reward = 0.0
        # 首先尝试符号验证
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # 如果符号验证失败，继续尝试下一种验证方法
        
        # 如果符号验证失败，尝试字符串匹配
        if reward == 0.0:
            try:
                # 从solution中提取答案数据集的答案是包含在<answer>标签中的
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # 返回的是全是字典的列表，但是不知道有多少个，每个字典的格式为：
                # {
                #     "API_name": action["name"],
                #     "API_params": action["arguments"]
                # }
                tool_cfg = parse_tool_config(content)
                if tool_cfg:
                    # 获取tool_cfg的第一个字典的API_params中的ans的值
                    # 每一步还是只有一个动作
                    student_answer = tool_cfg[0].get("API_params", {}).get("ans", content)
                else:
                    student_answer = content
                
                # 比较提取的答案
                if student_answer == ground_truth or float(verify(parse(student_answer), parse(ground_truth))) > 0:
                    reward = 1.0
            except Exception:
                pass  # 如果两种方法都失败，保持reward为0.0
                
        rewards.append(reward)
        # 调试模式下记录日志
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {item}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """
    奖励函数：检查完成内容是否符合特定格式
    
    Args:
        completions: 模型生成的完成内容
        **kwargs: 其他参数
    
    Returns:
        rewards: 奖励值列表，格式正确为1.0，错误为0.0
    """
    output_texts = kwargs.get("model_output_texts", None)
    if not output_texts:
        # 检查是否包含'Terminate'字样
        pattern = r'.*Terminate.*'
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]
    else:
        rewards = []
        # 检查JSON格式是否正确
        pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*\}'
        for output_text in output_texts:
            current_rewards = []
            for output_text_item in output_text:
                reward = 0.0
                try:
                    match = re.search(pattern, output_text_item)
                    assert match is not None
                    data = json.loads(match.group(0))
                    assert "thought" in data or "thoughts" in data or "actions" in data
                    reward = 1.0
                except Exception:
                    reward = 0.0
                current_rewards.append(reward)
            current_reward = sum(current_rewards) / len(current_rewards) if current_rewards else 0.0
            rewards.append(current_reward)
        return rewards
        

# 奖励函数注册表
reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

# 系统提示词，定义助手的行为和输出格式
SYSTEM_PROMPT = """You are a visual assistant capable of generating and solving steps for chart-based reasoning. Your goal is to answer chart-related questions. You can rely on your own capabilities or use external tools to assist in solving. Here are the available actions:
- **OCR**: Extracts text from an image. Example: `{"name": "OCR", "arguments": {"image": "img_1"}}`
- **Point**: Identifies a point in the image based on description and returns coordinates. Example: `{"name": "Point", "arguments": {"image": "img_1", "param": "x-axis value 1970"}}`
- **ZoomInSubfigure**: Crops the image to the specified subfigure. Example: `{"name": "ZoomInSubfigure", "arguments": {"image": "img_1", "param": "Downstream vs. Concept: Toy"}}`
- **SegmentRegionAroundPoint**: Segments a region around a given point. Example: `{"name": "SegmentRegionAroundPoint", "arguments": {"image": "img_1", "param": "x=\"21.5\" y=\"28.5\""}}`
- **DrawHorizontalLineByY**: Draws a horizontal line at a given y-coordinate. Example: `{"name": "DrawHorizontalLineByY", "arguments": {"image": "img_1", "param": "y=28.5"}}`
- **DrawVerticalLineByX**: Draws a vertical line at a given x-coordinate. Example: `{"name": "DrawVerticalLineByX", "arguments": {"image": "img_1", "param": "x=21.5"}}`
- **Terminate**: Ends the task and provides the final answer. Example: `{"name": "Terminate", "arguments": {"ans": "1985"}}`

To solve the problem:
1. Select actions from the provided tools list, combining them logically and building on previous steps. Call one action at a time, using its output for the next.
2. To use `SegmentRegionAroundPoint`, `DrawHorizontalLineByY`, or `DrawVerticalLineByX`, first call "Point" to get coordinates for further actions.

Your output should be in a strict JSON format as follows:
{"thought": "the reasoning process", "actions": [{"name": "action", "arguments": {"argument1": "value1", "argument2": "value2"}}]}
"""

def main(script_args, training_args, model_args):
    """
    主函数：处理训练流程
    
    Args:
        script_args: 脚本参数
        training_args: 训练参数
        model_args: 模型参数
    """
    # 设置随机种子以确保可重现性
    set_seed(training_args.seed)

    # 检查最后一个检查点
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"检测到检查点，在 {last_checkpoint} 处恢复训练。")

    # 获取奖励函数
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # 加载一个json格式的数据集
    dataset = load_dataset('json', data_files=script_args.dataset_name)
    
    # 从数据集加载图像
    def load_image_from_path(example):
        if "solution" not in example:
            example["solution"] = example["label"]
        # 删除label列
        if "label" in example:
            # 如果存在label列，则删除，不存在则返回None
            example.pop("label", None)

        image = Image.open(example["image_path"])  
        image = image.convert("RGBA") 
        example["image"] = image  
        return example

    # 格式化为对话
    def make_conversation(example):
        """
        将示例格式化为对话格式
        
        Args:
            example: 数据示例
        
        Returns:
            对话格式的数据
        """
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                # query_key默认为"question"，应该是为了索引
                {"role": "user", "content": example[script_args.query_key]},
            ],
        }

    # 问题模板
    QUESTION_TEMPLATE = "{Question}"

    def make_conversation_image(example):
        """
        将带图像的示例格式化为对话格式
        
        Args:
            example: 数据示例
        
        Returns:
            带图像的对话格式数据
        """
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SYSTEM_PROMPT},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example[script_args.query_key])},
                    ],
                },
            ],
        }

    # 处理数据集：根据是否包含图像路径选择不同的处理方式；dataset_train_split默认为"train"
    if "image_path" in dataset[script_args.dataset_train_split].features:
        print("image in dataset")
        dataset = dataset.map(load_image_from_path)
        dataset = dataset.map(make_conversation_image)
    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("query")
    
    # 选择适当的训练器：根据use_tool和use_vllm参数选择
    # 命令当中又有use_tool又有use_vllm，根据逻辑选择Qwen2VLGRPOToolVLLMTrainer
    if script_args.use_tool:
        trainer_cls = Qwen2VLGRPOToolTrainer if not training_args.use_vllm else Qwen2VLGRPOToolVLLMTrainer
        trainer = trainer_cls(
            model=model_args.model_name_or_path,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            # 默认没有传入，在GRPOConfig中定义，为'no'
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            peft_config=get_peft_config(model_args),
            # 没有传入，在ModelConfig中被定义，默认值为None，在Qwen2VLGRPOToolVLLMTrainer中重新定义默认值为"flash_attention_2"
            attn_implementation=model_args.attn_implementation,
            max_pixels=script_args.max_pixels,
            min_pixels=script_args.min_pixels,
            controller_addr=script_args.controller_addr,
        )
    else:
        trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
        trainer = trainer_cls(
            model=model_args.model_name_or_path,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            peft_config=get_peft_config(model_args),
            attn_implementation=model_args.attn_implementation,
            max_pixels=script_args.max_pixels,
            min_pixels=script_args.min_pixels,
        )
    print("using: ", trainer_cls)

    # 开始训练
    trainer.train()

    # 从保存的检查点继续训练
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # 保存模型并推送到Hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    # 解析命令行参数
    # trlparser：https://huggingface.co/docs/trl/main/script_utils
    # parser.parse_args_and_config()：https://huggingface.co/docs/trl/main/script_utils#trl.TrlParser.parse_args_and_config
    # 传了那么多参数进来，竟然能够自动解析，各自属于GRPOConfig还是ModelConfig
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
