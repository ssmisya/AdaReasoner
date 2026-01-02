import os
import json
import torch
from verl import DataProto


class ValidationRLLoggingBoardLogger:
    
    def __init__(
        self,
        root_log_dir: str,
        project_name: str,
        experiment_name: str
    ):
        self.save_path = os.path.join(
            root_log_dir, 
            project_name, 
            experiment_name
        )
        try:
            os.makedirs(self.save_path, exist_ok=True)
        except:
            pass
        
        # 查找当前正在使用的训练数据文件rank，保持一致
        rank = self._find_current_training_rank()
        
        # 使用相同的rank创建验证数据文件
        val_filename = f"rollout_data_rank_val{rank}.jsonl"
        self.file_path = os.path.join(self.save_path, val_filename)
    
    def _find_current_training_rank(self):
        """
        找到当前正在使用的训练数据文件的rank
        采用和RLLoggingBoardLogger相同的逻辑来确定rank
        """
        import glob
        import os
        
        # 采用和训练数据记录器相同的逻辑：找到第一个不存在的rank
        rank = 0
        while True:
            filename = f"rollout_data_rank{rank}.jsonl"
            file_path = os.path.join(self.save_path, filename)
            if not os.path.exists(file_path):
                # 如果这个rank的文件不存在，说明当前应该使用这个rank
                return rank
            rank += 1
            # 防止无限循环，设置一个合理的上限
            if rank > 1000:
                return 0

    def log(
        self,
        step: int,
        inputs: list,
        outputs: list,
        scores: list,
        tokenizer
    ):
        """
        记录验证数据到jsonl文件
        
        参数:
            step (int): 当前训练步数
            inputs (list): 输入prompt列表
            outputs (list): 模型输出列表
            scores (list): 对应的奖励分数列表
            tokenizer: 分词器
        """
        with open(self.file_path, "a") as f:
            for i, (input_text, output_text, score) in enumerate(zip(inputs, outputs, scores)):
                cur_sample = {
                    "step": step,
                    "prompt": input_text,
                    "response": output_text,
                    "reward": score,
                    "validation": True  # 标记这是验证数据
                }
                
                f.write(f"{json.dumps(cur_sample, ensure_ascii=False)}\n")


class RLLoggingBoardLogger:
    
    def __init__(
        self,
        root_log_dir: str,
        project_name: str,
        experiment_name: str
    ):
        self.save_path = os.path.join(
            root_log_dir, 
            project_name, 
            experiment_name
        )
        try:
            os.makedirs(self.save_path, exist_ok=True)
        except:
            pass
        
        # 自动决定文件路径，避免文件名冲突
        rank = 0
        while True:
            filename = f"rollout_data_rank{rank}.jsonl"
            self.file_path = os.path.join(self.save_path, filename)
            if not os.path.exists(self.file_path):
                break
            rank += 1

    def log(
        self,
        data: dict,
        step: int,
        batch: DataProto,
        *args,
        **kwargs
    ):
        if 'tokenizer' not in kwargs:
            raise ValueError("Please provide a tokenizer.")
        
        tokenizer = kwargs['tokenizer']
        
        rm_response_list = kwargs['rm_response_list'] if 'rm_response_list' in kwargs else None
        with open(self.file_path, "a") as f:
            for i in range(len(batch)):
                data_item = batch[i]
                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                rm_response = rm_response_list[i] if rm_response_list else None

                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                prompt_str = tokenizer.decode(valid_prompt_ids)
                response_str = tokenizer.decode(valid_response_ids)
                response_tokens = [tokenizer.decode([token]) for token in valid_response_ids]
                cur_sample = {
                    "step": step,
                    "prompt": prompt_str,
                    "response": response_str,
                    "response_tokens": response_tokens,
                    "logprobs": data_item.batch['old_log_probs'][:valid_response_length].cpu().tolist(),
                    # "ref_logprobs": data_item.batch['ref_log_prob'][:valid_response_length].cpu().tolist(),
                    # "values": data_item.batch['values'][:valid_response_length].cpu().tolist(),
                    "token_rewards": data_item.batch['token_level_rewards'][:valid_response_length].cpu().tolist(),     # with KL penalty
                    "reward": data_item.batch['token_level_scores'][:valid_response_length].cpu().sum().item(),         # without KL penalty"
                }
                
                if "ground_truth" in data_item.non_tensor_batch['reward_model']:
                    cur_sample["ground_truth"] = data_item.non_tensor_batch['reward_model']["ground_truth"]

                if "values" in data_item.batch:
                    cur_sample['values'] = data_item.batch['values'][:valid_response_length].cpu().tolist()
                
                if rm_response is not None:
                    cur_sample['rm_response'] =rm_response
                
                f.write(f"{json.dumps(cur_sample, ensure_ascii=False)}\n")