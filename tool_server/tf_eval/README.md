# AdaEval: Evaluation Framework for Tool Planning Models

> ⚠️ **Important**  
> AdaEval relies on **live tool interactions**.  
> Please ensure that the **Tool Server is running** before starting evaluation, especially when using online (GPU-backed) tools.

---

## 📖 Overview

**AdaEval** is the core evaluation module for **tool planning models** in AdaReasoner.  
It provides a unified inference and evaluation pipeline for **multi-turn, tool-augmented reasoning**, built on top of **HuggingFace Accelerate** for efficient batch inference and multi-GPU parallelism.

AdaEval supports evaluating a wide range of vision-language models—local models (via VLLM or native implementations) and API-based models—on diverse visual reasoning benchmarks that require **iterative tool use**, such as VSP, Jigsaw, GUIQA, and WebQA.

All evaluation logic is implemented under:

tool_server/tf_eval

---

## ✨ Key Capabilities

- 🔧 **Live Tool Interaction**  
  Direct integration with the Tool Server for online tool execution during inference.

- 🔁 **Multi-Round Tool Planning**  
  Supports iterative model–tool interaction with configurable maximum rounds.

- 🚀 **Accelerate-based Parallel Inference**  
  Scalable batch inference and multi-GPU parallelism using HuggingFace Accelerate.

- 📌 **Checkpoint Resume & Save**  
  Intermediate results can be saved and resumed at the task level.

- 🧩 **Model–Task Decoupling**  
  Models and tasks are modular and connected through a unified dataset interface.

---

## 🏗️ Framework Structure

```bash
tool_server/tf_eval
├── models/            # Tool-planning model implementations
├── tasks/             # Task definitions and evaluation logic
├── tool_inferencer/   # Batch inference + sequential tool calling
├── utils/             # Argument parsing and helpers
└── scripts/           # Example configs and launch scripts
```
---

## 🧠 Core Abstractions

AdaEval is organized around **two core concepts**:

### 1. Model

A **tool planning model** defines how to:
- Construct multimodal conversations
- Generate tool calls
- Incorporate tool responses
- Produce the next reasoning step

Models are implemented under:

`tool_server/tf_eval/models/`

Each model must inherit from:

tp_model (tool_server.tf_eval.models.abstract_model.tp_model)

---

### 2. Task

A **task** defines:
- How data is loaded
- What information is passed to the model
- How results are evaluated

Tasks are implemented under:

tool_server/tf_eval/tasks/<task_name>/

Each task must implement:
- `load_data_function()`
- `evaluate_function(results, meta_data)`

---

## 🔄 Data Flow & Evaluation Logic

The model and task are connected through a **PyTorch-style dataset** (`base_dataset`):

- **Task** provides:
  - `load_data_function()` → list of samples
  - `evaluate_function()` → final metrics

- **Model** provides:
  - `getitem_fn()` → construct one inference instance
  - `generate()` → batch inference
  - Conversation construction and update logic

Evaluation proceeds as:

1. Load task data
2. Construct dynamic batches
3. Perform multi-round inference
4. Execute tools via Tool Server
5. Store intermediate results with `dataset.store_results(res)`
6. Compute final metrics via `evaluate_function()`

---

## 🚀 Running AdaEval

### 1. Prepare Config File (Recommended)

AdaEval supports **YAML-based configuration**, either as:
- a single dict, or
- a list of dicts (for multiple runs)

Example config:

```yaml
model_args:
  model: vllm_models
  model_args: pretrained=/path/to/model,tensor_parallel=2,limit_mm_per_prompt=10
  batch_size: 50
  max_rounds: 6
  model_mode: general

task_args:
  task_name: vsp
  tool_selection: Point,Draw2DPath
  resume_from_ckpt:
    vsp: ./logs/ckpt/vsp.jsonl
  save_to_ckpt:
    vsp: ./logs/ckpt/vsp.jsonl
  middle_images_save_dir:
    vsp: ./logs/middle_images/vsp

script_args:
  verbosity: INFO
  output_path: ./logs/results/vsp_results.jsonl
  if_use_tool: True

```
⸻

### 2. Launch Evaluation

python \
  -m tool_server.tf_eval \
  --config ${config_file}

📌 Note
	•	Batch inference and multi-GPU parallelism are handled by Accelerate
	•	When evaluating API-based models (e.g., OpenAI, Gemini):
	•	batch_size must be 1
	•	Do not use multi-process parallelism

⸻

## ⚡ VLLM Backend Support

AdaEval provides a built-in VLLM backend:

`tool_server/tf_eval/models/vllm_models.py`

To use VLLM, simply set:
```yaml
model_args:
  model: vllm_models
  model_args: pretrained=/path/to/model,tensor_parallel=4
```
This enables:
	•	High-throughput batch inference
	•	Tensor parallelism
	•	Seamless integration with the tool planning loop

⸻

## 🧩 Extending AdaEval

### ➕ Add a New Model
	1.	Implement model under:

`tool_server/tf_eval/models/`


	2.	Inherit from tp_model
	3.	Implement:
	•	getitem_fn
	•	generate
	•	generate_conversation_fn
	•	append_conversation_fn
	4.	Register in:

tool_server/tf_eval/models/__init__.py



⸻

### ➕ Add a New Task
	1.	Create:

tool_server/tf_eval/tasks/your_task/


	2.	Implement:
	•	config.yaml
	•	task.py
	3.	Define:
	•	load_data_function()
	•	evaluate_function(results, meta_data)
	4.	Set task_name in config to match folder name

⸻

## 🧪 Tool Interaction & Dynamic Batch
	•	Tool execution is handled by the Tool Server
	•	AdaEval manages:
	•	Sequential tool calling
	•	Round-based stopping
	•	Metadata tracking via DynamicBatchItem
	•	Batch inference logic lives in:

tool_server/tf_eval/tool_inferencer/base_inferencer.py



⸻

## 📌 Notes & Best Practices
	•	Always start the Tool Server first
	•	Use checkpoints for long evaluations
	•	Save intermediate images for debugging
	•	Limit limit_mm_per_prompt to control memory
	•	Prefer VLLM for large-scale evaluation

