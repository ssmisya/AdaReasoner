<div align="center">
  <img src="docs/logo.png" alt="Logo" width="300">
  <h1 align="center">Dynamic Tool Orchestration for Iterative Visual Reasoning</h1>

  <a href="#">
    <img src="https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper">
  </a>
  <a href="docs/README.md">
    <img src="https://img.shields.io/badge/Docs-1f6feb?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Docs">
  </a>
  <a href="https://huggingface.co/collections/hitsmy/adareasoner">
    <img src="https://img.shields.io/badge/Data-fcd022?style=for-the-badge&logo=huggingface&logoColor=000" alt="Data">
  </a>
  <a href="https://your-homepage-link-here">
    <img src="https://img.shields.io/badge/Homepage-2ea44f?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Homepage">
  </a>
  <a href="https://huggingface.co/hitsmy/AdaReasoner-7B">
    <img src="https://img.shields.io/badge/Model-fcd022?style=for-the-badge&logo=huggingface&logoColor=000" alt="Model">
  </a>
  <a href="https://your-gradio-demo-link-here">
  <img src="https://img.shields.io/badge/Demo-FF7C00?style=for-the-badge&logo=gradio&logoColor=white" alt="Demo">
  </a>
  
</div>

<!-- ## AdaReasoner: A Full-Stack Recipe for Intelligent Visual Agents -->

<div align="center">
  <img src="docs/proj_structure.png" alt="structure" width="800">
  <br>
  <em>Overview of the AdaReasoner framework.</em>
</div>



## 🧩 Project Architecture Overview
| **Module Name** | **Functionality** | **Location** | **Dependency** |
|---|---|---|---|
| **Tool Server** | Supports both online and offline tools and serves as the core of all components.| [`tool_server/tool_workers/`](./tool_server/tool_workers/) | [`tool-server`](#️-installation) |
| **AdaTC** | Performs Tool Cold Start (TC, SFT) of tool planning models, concretely, we are using LLaMA Factory as our backbone. | [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) |[LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) |
| **AdaTG** | Performs Tool GRPO (TG, RL) with tool interaction. | [`AdaTG`](./AdaTG) | [`adatg`](./AdaTG) |
| **AdaEval** | An evaluation framework that supports any combinations of models and tasks for tool planning evaluation. | [`tool_server/tf_eval/`](./tool_server/tf_eval/) |[`tool-server`](#️-installation) |
| **AdaDataCuration** | Constructs data for SFT (TC), RL (TG), and evaluation (AdaEval). | [`tool_server/ada_data_curation/`](./tool_server/ada_data_curation/) | TBD |

---


## News
- **[2026/01]** AdaReasoner paper is now available on [arXiv](#).
- **[2026/01]** The models and datasets are released on [HuggingFace](https://huggingface.co/collections/hitsmy/adareasoner).
- **[2026/01]** AdaReasoner Toolkit codebase is released along with evaluation scripts. Try it out!




## 🚀 Quick Start
This framework comprises three main components: the fundamental tool service supplier ``tool_server``, the inference evaluation framework `AdaEval`, and the RL work ``AdaRL``. Each component has its own environment requirements. The `tool_server` serves as the foundation and must be successfully launched before performing any inference or training.

### 🖥️ Prerequisite: Launch Tool Server
The Tool Server provides two types of tools.
**Online tools** mainly offer compute-intensive functionalities and need to be deployed on GPU servers. In contrast, **offline tools** provide lightweight utilities that do not require GPU resources and can be executed efficiently on CPUs.
If you only need offline tools, you can simply install the environment without running the Tool Server.

### 🛠️ Installation
First of all, provide a pytorch-based environment.

```bash
# [Optional] Create a clean Conda environment
conda create -n tool-server python=3.10
conda activate tool-server
# Install PyTorch or prepare a torch-based environment
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install this project
git clone https://github.com/ssmisya/AdaReasoner.git
# You can reference our requirements.txt for other dependencies
pip install -r AdaReasoner/requirements/requirements.txt 
pip install -e AdaReasoner # We didn't add too many constraints for easier installation

```
⚠️ **Be aware:**

We **deliberately selected minimal dependencies** in this project to reduce the risk of conflicts. As a result, you may need to manually install any missing packages based on your environment.

#### Option 2.1 Start Tool Server through SLURM
It's recommended to start the tool server through SLURM because it's more flexible.
```bash
## First, modify the config to adapt to your own environment
## AdaReasoner/tool_server/tool_workers/scripts/launch_scripts/config/all_service_example.yaml

## Start all services
cd AdaReasoner/tool_server/tool_workers/scripts/launch_scripts
python start_server_config.py --config ./config/all_service_example.yaml

## Press Ctrl + C to shutdown all services automatically.
```

#### Option 2.2 Start Tool Server Locally
We made a slight modification to ``start_server_config.py`` to create ``start_server_local.py``, primarily by removing the logic related to SLURM job detection and adapting it for local execution instead.
```bash
## First, modify the config to adapt to your own environment
## AdaReasoner/tool_server/tool_workers/scripts/launch_scripts/config/all_service_example_local.yaml

## Start all services
cd AdaReasoner/tool_server/tool_workers/scripts/launch_scripts
python start_server_local.py --config ./config/all_service_example_local.yaml

## Press Ctrl + C to shutdown all services automatically.
```

You can then inspect the log files to diagnose and resolve any potential issues. Due to the complexity of this project, we cannot guarantee that it will run without errors on every machine.

### 🔍 Usage1: Run Evaluation with AdaEval

### 🛠️ Installation
This environment is the same with tool-server, first of all, provide a pytorch-vllm-based environment.
* vllm>=0.7.3
* torch>=2.5.1
* transformers>=4.49.0
* flash_attn>=2.7.3

```bash
# [Optional] Create a clean Conda environment
conda create -n vllm python=3.10
conda activate tool-server

# Install this project
git clone https://github.com/ssmisya/AdaReasoner.git
pip install -r AdaReasoner/requirements/inference_requirements.txt # Tool Server Requirements
pip install -e AdaReasoner

```

#### ✅ Option 1: Direct Evaluation (e.g., Qwen2VL on ChartGemma)

```bash
accelerate launch  --config_file  ${accelerate_config} \
-m tool_server.tf_eval \
--model qwen2vl \
--model_args pretrained=Qwen/Qwen2-VL-7B-Instruct \
--task_name chartgemma \
--verbosity INFO \
--output_path ./tool_server/tf_eval/scripts/logs/results/chartgemma/qwen2vl.jsonl \
--batch_size 2 \
--max_rounds 3 \
```

#### 🧩 Option 2: Evaluation via Config File (Recommended)


```bash
accelerate launch  --config_file  ${accelerate_config} \
-m tool_server.tf_eval \
--config ${config_file}
```

#### Config file example:

```yaml
    - model_args:
    # The model series you want to test, must be the same with the file name under tf_eval/models
    model: vllm_models
    # The arguments to pass to the model, specify model path, tensor parallel size, and other parameters
    model_args: pretrained=/mnt/petrelfs/songmingyang/songmingyang/runs/tool_factory/sft/v1/Qwen2.5-VL-7B-Instruct-pathverify_v0,tensor_parallel=2,limit_mm_per_prompt=10
    # Batch size for inference. Adjust according to your GPU memory.
    batch_size: 50
    # Maximum number of rounds for tool-model interaction
    max_rounds: 6
    # Model operation mode
    model_mode: general
  task_args:
    # The task to evaluate (options include: chartgemma,chartqa,charxiv,docvqa,ocrbench,reachqa,vstar)
    task_name: vsp
    # Specific tools to use for evaluation, don't set this if you want all tools available.
    tool_selection: Point,Draw2DPath
    # Checkpoint to resume from, organized as task_name: path
    resume_from_ckpt:
      vsp: ./tool_server/tf_eval/scripts/logs/ckpt/frozen_lake/pathverify_v0_qwen25_7b/vsp.jsonl
    # Path to save checkpoint, organized as task_name: path
    save_to_ckpt:
      vsp: ./tool_server/tf_eval/scripts/logs/ckpt/frozen_lake/pathverify_v0_qwen25_7b/vsp.jsonl
    # Directory to save intermediate images generated during evaluation
    middle_images_save_dir:
      vsp: ./tool_server/tf_eval/scripts/logs/ckpt/frozen_lake/pathverify_v0_qwen25_7b/middle_images
  script_args:
    # Logging verbosity level
    verbosity: INFO
    # Path to save final evaluation results
    output_path: ./tool_server/tf_eval/scripts/logs/results/frozen_lake/pathverify_v0_qwen25_7b/qwen25_allres.jsonl
    # Whether to use tools during inference
    if_use_tool: True
```

For detailed information and config setting please refer to our [documentation](docs/README.md).



## 🧠 Usage2: Run Tool Cold Start (TC, SFT) with AdaTC

Once the vision tools are properly deployed, we provide a flexible training pipeline to teach models **how to plan and invoke tools** effectively through **SFT** and our proposed **V-ToolRL** methods.

Our training pipeline builds on the solid foundation of [OpenR1](https://github.com/OpenR1), integrating visual tools as external reasoning capabilities.



### 🔁 Usage3: Run Tool GRPO (TG, RL) with AdaTG


## 📚 Citation
If you find this project useful in your research, please consider citing our paper:

```bibtex
@article{your2024adareasoner,
  title={Dynamic Tool Orchestration for Iterative Visual Reasoning},
  author={Your Name and Coauthor Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```



