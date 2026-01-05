

from .models import get_model
from .tasks import get_task_object, get_task_functions
from .tasks.base_dataset.base_evaluation_dataset import BaseEvalDataset, DataCollatorForSupervisedDataset

from .utils.utils import *
from .utils.arguments import *

from .utils.log_utils import get_logger, set_verbosity
from .tool_inferencer import BaseToolInferencer
from .tool_inferencer import BaseInferencer
from pathlib import Path
from tool_server.utils.debug import remote_breakpoint
try:
    from math_verify import parse, verify
except ImportError:
    print("math_verify package not found. Please install it to use math verification features.")

logger = get_logger(__name__)



class TFEvaluator():
    def __init__(self, model_args, task_args, script_args):
        self.config = script_args.config
        self.model_args = model_args
        self.task_args = task_args
        self.script_args = script_args
        self.tasks = self.task_args.task_name
        
        # remote_breakpoint(port=7119)
        
        self.model = get_model(self.model_args.model)(**self.model_args.model_args)
        self.if_use_tool = self.script_args.if_use_tool
        self.if_randomize_tool = self.script_args.if_randomize_tool
        self.model.set_enable_tool(self.if_use_tool)
        
        
        max_rounds = self.model_args.max_rounds
        stop_token = self.model_args.stop_token
        
        set_verbosity(self.script_args.verbosity)
        
        # 获取日志目录，与save_to_ckpt在同一文件夹
        log_dir = None
        if hasattr(self.task_args, 'save_to_ckpt') and self.task_args.save_to_ckpt:
            # 获取第一个任务的保存路径
            first_task = list(self.task_args.save_to_ckpt.keys())[0]
            ckpt_path = self.task_args.save_to_ckpt[first_task]
            
            # 指定文件路径
            ckpt_file_path = Path(ckpt_path)
            ckpt_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 提取目录路径
            log_dir = os.path.dirname(ckpt_path)
            logger.info(f"Tool call statistics will be saved to {log_dir}")
        
        # 不存在就创建
        output_file_path = Path(self.script_args.output_path)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.if_use_tool:
            self.inferencer = BaseToolInferencer(
                tp_model=self.model,
                batch_size=self.model_args.batch_size,
                model_mode=self.model_args.model_mode,
                max_rounds = max_rounds,
                stop_token = stop_token,
                controller_addr = self.script_args.controller_addr,
                if_use_tool = self.if_use_tool,
                if_randomize_tool = self.if_randomize_tool,
            )
        else:
            self.inferencer = BaseInferencer(
                tp_model=self.model,
                batch_size=self.model_args.batch_size,
                if_use_tool = self.if_use_tool,
            )

    
    def evaluate(self):

        for task_name in self.tasks:
            logger.info(f"evaluating {task_name}")
            task_dict = get_task_functions(task_name)
            load_data_function, evaluate_function, task_config = task_dict["load_data_function"], task_dict["evaluate_function"], task_dict["task_config"]
            self.model.set_generation_config(task_config.generation_config)
            
            # Generate the first batch
            dataset = BaseEvalDataset(
                load_data_function=load_data_function,
                getitem_function=self.model.getitem_fn,
                evaluate_function=evaluate_function,
                task_config = task_config,
                task_args = self.task_args,
                model_args = self.model_args,
            )
            # 设置任务名称，用于保存工具调用统计
            dataset.task_name = task_name
            # 设置可用工具
            self.inferencer.set_tool_selection(dataset.tool_selection)
            
            self.inferencer.batch_inference(dataset)

            res_log = dataset.evaluate()
            if is_main_process() or "vllm_models" in self.model_args.model:
                logger.info(f"evaluation of {task_name} completed")
                res_log = remove_non_serializable(res_log)
                append_jsonl(res_log, self.script_args.output_path)
            

