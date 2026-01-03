
 # AdaEval: Unified Evaluation Framework for Tool Planning Models

<!-- <div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</div> -->
> ⚠️ **Important:** Online tools require the **tool server** to be running.  
> Please start the tool server before invoking any online tools.
## 📖 Overview

**AdaEval** is a unified and extensible evaluation framework designed for assessing tool planning models across diverse visual reasoning tasks. It provides a flexible architecture that supports arbitrary combinations of models and benchmarks, enabling systematic evaluation of vision-language models with tool interaction capabilities.

### Key Features

- 🔧 **Unified Tool Interface**: Seamless integration with Tool Server for consistent tool interaction
- 🎯 **Multi-Task Support**: Evaluate on various benchmarks including VSP, Jigsaw, WebMMU, and more
- 🚀 **Model Agnostic**: Support for multiple model backends (VLLM, Qwen2VL, Gemini, OpenAI, etc.)
- 📊 **Checkpoint Management**: Automatic result caching and resume-from-checkpoint functionality
- 🔄 **Dynamic Batch Processing**: Efficient multi-round tool interaction with dynamic batch management
- 🎲 **Tool Randomization**: Optional tool name randomization for robust evaluation
- 💾 **Comprehensive Logging**: Detailed tracking of tool calls, intermediate images, and evaluation metrics

---

## 🏗️ Architecture

AdaEval consists of three core components:

```
AdaEval (tf_eval)
├── 📁 models/              # Model adapters (VLLM, Qwen2VL, Gemini, etc.)
├── 📁 tasks/               # Task-specific datasets and evaluation logic
├── 📁 tool_inferencer/     # Tool interaction and batch management
└── 📁 utils/               # Utilities (logging, arguments, helpers)
```

### Component Breakdown

| **Component** | **Functionality** | **Key Files** |
|---------------|-------------------|---------------|
| **Models** | Abstract model interface + specific implementations | `abstract_model.py`, `vllm_models.py` |
| **Tasks** | Dataset loading + task-specific evaluation metrics | `base_evaluation_dataset.py` |
| **Tool Inferencer** | Multi-round tool calling + dynamic batch processing | `base_inferencer.py` |
| **Evaluator** | Main orchestration logic | `evaluator.py` |

---

## 🚀 Quick Start

### Prerequisites

Ensure **Tool Server** is running before evaluation (see Tool Server Setup).

### Installation

```bash
# Create environment
conda create -n adaeval python=3.10
conda activate adaeval

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Install AdaEval dependencies
cd OpenThinkIMG
pip install -e .
pip install -r requirements/inference_requirements.txt
```

### Basic Usage

#### Option 1: Command-Line Arguments

```bash
accelerate launch --config_file ${accelerate_config} \
  -m tool_server.tf_eval \
  --model vllm_models \
  --model_args pretrained=Qwen/Qwen2-VL-7B-Instruct,tensor_parallel=2 \
  --task_name vsp,jigsaw_coco \
  --tool_selection Point,Crop,OCR,Draw2DPath \
  --batch_size 50 \
  --max_rounds 10 \
  --if_use_tool True \
  --if_randomize_tool False \
  --output_path ./results/qwen2vl_evaluation.jsonl \
  --verbosity INFO
```

#### Option 2: Config File (Recommended)

```bash
accelerate launch --config_file ${accelerate_config} \
  -m tool_server.tf_eval \
  --config configs/unified_evaluation.yaml
```

**Example Config File** (`configs/unified_evaluation.yaml`):

```yaml
model_args:
  model: vllm_models
  model_args: pretrained=/path/to/model,tensor_parallel=2,limit_mm_per_prompt=10
  batch_size: 50
  max_rounds: 11
  model_mode: general

task_args:
  task_name: vsp,jigsaw_coco,jigsaw_blink,web_guichat
  tool_selection: Point,Draw2DPath,AStarWithPixelCoordinate,OCR,Crop
  
  # Resume from previous checkpoints (optional)
  resume_from_ckpt:
    vsp: ./logs/ckpt/vsp.jsonl
    jigsaw_coco: ./logs/ckpt/jigsaw_coco.jsonl
  
  # Save checkpoints during evaluation
  save_to_ckpt:
    vsp: ./logs/ckpt/vsp.jsonl
    jigsaw_coco: ./logs/ckpt/jigsaw_coco.jsonl
  
  # Save intermediate images (optional)
  middle_images_save_dir:
    vsp: ./logs/middle_images/vsp
    jigsaw_coco: ./logs/middle_images/jigsaw_coco

script_args:
  verbosity: INFO
  output_path: ./results/evaluation_results.jsonl
  if_use_tool: True
  if_randomize_tool: False  # Enable tool name randomization
  controller_addr: ./log/controller_addr.json  # Tool Server address
```

---

## 📋 Configuration Guide

### Model Arguments

| **Parameter** | **Description** | **Example** |
|---------------|-----------------|-------------|
| `model` | Model backend type | `vllm_models`, `qwen2vl`, `gemini` |
| `pretrained` | Model checkpoint path or HF model ID | `Qwen/Qwen2-VL-7B-Instruct` |
| `tensor_parallel` | Number of GPUs for tensor parallelism | `2` |
| `batch_size` | Inference batch size | `50` |
| `max_rounds` | Maximum tool interaction rounds | `10` |
| `limit_mm_per_prompt` | Max images per conversation turn | `10` |

### Task Arguments

| **Parameter** | **Description** | **Example** |
|---------------|-----------------|-------------|
| `task_name` | Comma-separated task list | `vsp,jigsaw_coco,web_guichat` |
| `tool_selection` | Available tools for this evaluation | `Point,Crop,OCR,Draw2DPath` |
| `resume_from_ckpt` | Path to checkpoint files for resuming | `{vsp: ./ckpt/vsp.jsonl}` |
| `save_to_ckpt` | Path to save checkpoints during eval | `{vsp: ./ckpt/vsp.jsonl}` |
| `middle_images_save_dir` | Directory to save intermediate images | `{vsp: ./images/vsp}` |

### Script Arguments

| **Parameter** | **Description** | **Example** |
|---------------|-----------------|-------------|
| `if_use_tool` | Enable tool usage during inference | `True` / `False` |
| `if_randomize_tool` | Randomize tool names for robustness | `True` / `False` |
| `controller_addr` | Tool Server controller address file | `./log/controller_addr.json` |
| `output_path` | Final results output path | `./results/output.jsonl` |
| `verbosity` | Logging level | `INFO`, `DEBUG`, `WARNING` |

---

## 🎯 Supported Tasks

AdaEval supports evaluation on multiple benchmarks:

| **Task** | **Description** | **Metrics** |
|----------|-----------------|-------------|
| **VSP** | Visual Spatial Planning | Accuracy, Tool Usage |
| **Jigsaw (COCO)** | Image puzzle solving | Success Rate |
| **Jigsaw (BLINK)** | Complex puzzle reasoning | Success Rate |
| **WebMMU** | Web-based multimodal understanding | Accuracy |
| **Web GUIChat** | GUI interaction | Success Rate |
| **ChartGemma** | Chart question answering | Accuracy |

### Adding Custom Tasks

To add a new task:

1. **Create task directory**: `tasks/your_task/`
2. **Implement dataset loader**: Inherit from `BaseEvalDataset`
3. **Define evaluation function**: Custom metric computation
4. **Register task**: Add to __init__.py

**Example**:

```python
# tasks/your_task/__init__.py
def load_data_function():
    """Load your dataset"""
    return dataset_items

def evaluate_function(results, meta_data):
    """Compute task-specific metrics"""
    accuracy = compute_accuracy(results)
    return {"accuracy": accuracy}

task_config = {
    "task_name": "your_task",
    "generation_config": {"max_new_tokens": 2048}
}
```

---

## 🔧 Advanced Features

### 1. Checkpoint Management

**Resume from checkpoint**:
```yaml
task_args:
  resume_from_ckpt:
    vsp: ./ckpt/vsp.jsonl
```

**Auto-save during evaluation**:
```yaml
task_args:
  save_to_ckpt:
    vsp: ./ckpt/vsp.jsonl
```

### 2. Tool Name Randomization

Enable tool randomization to test model robustness:

```yaml
script_args:
  if_randomize_tool: True
```

This feature:
- Randomizes tool names (e.g., `Point` → `cnpama`)
- Randomizes parameter names (e.g., `image` → `cfngnf`)
- Uses deterministic seed for reproducibility
- Automatically maps back to original names during execution

### 3. Intermediate Image Saving

Save all intermediate images generated during tool calls:

```yaml
task_args:
  middle_images_save_dir:
    vsp: ./logs/images/vsp
```

Images are saved in structured directories:
```
./logs/images/vsp/
├── sample_0/
│   ├── round_0_image.png
│   ├── round_1_image.png
│   └── ...
└── sample_1/
    └── ...
```

### 4. Multi-Round Tool Interaction

AdaEval supports iterative tool calling with dynamic batch management:

```python
# Automatic multi-round processing
max_rounds = 10  # Set in config

# Each round:
# 1. Model generates tool call
# 2. Tool Server executes tool
# 3. Result appended to conversation
# 4. Repeat until stop condition or max_rounds
```

---

## 📊 Output Format

### Result JSONL Structure

Each line in the output file contains:

```json
{
  "task_name": "vsp",
  "model_name": "vllm_models",
  "accuracy": 0.85,
  "results": [
    {
      "idx": 0,
      "original_question": "Navigate from A to B",
      "ground_truth_answer": "path_coordinates",
      "model_response": ["<tool_call>...</tool_call>", "Final answer"],
      "tool_cfg": [{"tool": "Point", "args": {...}}],
      "tool_response": [{"status": "success", "result": {...}}],
      "final_answer": "extracted_answer",
      "is_correct": true
    }
  ]
}
```

### Checkpoint JSONL Structure

```json
{
  "task_name": "vsp",
  "model_name": "vllm_models",
  "results": {
    "idx": 0,
    "model_response": ["response1", "response2"],
    "tool_cfg": [{"tool": "Point"}],
    "final_answer": "answer"
  }
}
```

---

## 🛠️ Model Backend Support

AdaEval supports multiple model backends:

| **Backend** | **Description** | **Key Features** |
|-------------|-----------------|------------------|
| **VLLM** | High-throughput inference engine | Parallel batch processing, efficient memory |
| **Qwen2VL** | Qwen2-VL native implementation | Full conversation support |
| **Gemini** | Google Gemini API | Cloud-based evaluation |
| **OpenAI** | OpenAI API (GPT-4V, etc.) | Multimodal API support |
| **LMDeploy** | LMDeploy inference engine | Optimized for deployment |

### Adding Custom Model Backend

1. **Create model file**: `models/your_model.py`
2. **Inherit from `tp_model`**: Implement required methods
3. **Register model**: Add to __init__.py

**Required Methods**:
```python
class YourModel(tp_model):
    def generate_conversation_fn(self, text, images, role, **kwargs):
        """Format input as conversation"""
        pass
    
    def append_conversation_fn(self, conversation, text, image, role):
        """Append to conversation history"""
        pass
    
    def generate(self, batch):
        """Batch inference"""
        pass
```

---

## 🔍 Debugging & Logging

### Enable Debug Mode

```yaml
script_args:
  verbosity: DEBUG
```

### Log Structure

AdaEval generates detailed logs including:

- Model responses for each round
- Tool call configurations
- Tool execution results
- Errors and exceptions
- Timing information

**Example Log Output**:
```
2025-01-13 10:00:00 | INFO | evaluating vsp
2025-01-13 10:00:05 | INFO | Converting randomized call: cnpama(cfngnf) -> Point(image)
2025-01-13 10:00:06 | INFO | Tool Point executed successfully
2025-01-13 10:00:10 | INFO | Round 1/10 completed
```

---

## 📈 Performance Tips

1. **Batch Size Tuning**: Increase `batch_size` for better GPU utilization
2. **Tensor Parallelism**: Use multiple GPUs with `tensor_parallel`
3. **Checkpoint Frequently**: Set `save_to_ckpt` to avoid re-computation
4. **Limit Image History**: Reduce `limit_mm_per_prompt` to save memory
5. **Use VLLM**: For maximum throughput on supported models

---

## 🤝 Contributing

We welcome contributions! To add new tasks or model backends:

1. Fork the repository
2. Create your feature branch
3. Add tests and documentation
4. Submit a pull request

---

## 📝 Citation

If you use AdaEval in your research, please cite:

```bibtex
@article{openthinkimg2025,
  title={OpenThinkIMG: Dynamic Tool Orchestration for Iterative Visual Reasoning},
  author={Your Team},
  journal={arXiv preprint arXiv:2505.08617},
  year={2025}
}
```

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/OpenThinkIMG/OpenThinkIMG/issues)
- **Discussions**: [GitHub Discussions](https://github.com/OpenThinkIMG/OpenThinkIMG/discussions)
- **Documentation**: Full Docs

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.