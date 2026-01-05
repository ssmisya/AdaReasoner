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
| **Module Name** | **Functionality** | **Location** |
|---|---|---|
| **Tool Server** | Supports both online and offline tools and serves as the core of all components.| [`tool_server/tool_workers/`](./tool_server/tool_workers/) |
| **AdaTC** | Performs Tool Cold Start (TC, SFT) of tool planning models, concretely, we are using LLaMA Factory as our backbone. | [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) |
| **AdaTG** | Performs Tool GRPO (TG, RL) with tool interaction. | [`AdaTG`](./AdaTG) |
| **AdaEval** | An evaluation framework that supports any combinations of models and tasks for tool planning evaluation. | [`tool_server/tf_eval/`](./tool_server/tf_eval/) |
| **AdaDataCuration** | Constructs data for SFT (TC), RL (TG), and evaluation (AdaEval). | [`tool_server/ada_data_curation/`](./tool_server/ada_data_curation/) |

---


## News
- **[2026/01]** AdaReasoner paper is now available on [arXiv](#).
- **[2026/01]** The models and datasets are released on [HuggingFace](https://huggingface.co/collections/hitsmy/adareasoner).
- **[2026/01]** AdaReasoner Toolkit codebase is released along with evaluation scripts. Try it out!




## 🚀 Quick Start
This framework comprises three main components: the fundamental tool service supplier ``tool_server``, the inference evaluation framework `AdaEval`, and the RL work ``AdaRL``. Each component has its own environment requirements. The `tool_server` serves as the foundation and must be successfully launched before performing any inference or training.

### 🖥️ Step 1: Launch Vision Tool Server
You can either run our tool server using the provided `Docker image` or launch the `tool_server` locally, depending on your environment preferences.

### 🐳 Option 1: Docker Image
It's recommended to try our ``Tool Server`` docker image. You can either download our provided `tool_server` image or build it by your self!

📌 **Note:**
1. It’s recommended to use the `-v /path/to/your/logdir:/log` option to mount a host directory to the container’s `/log` directory, which allows you to view runtime logs and receive the `controller_addr` output.
2. The controller address is saved at ``/path/to/your/logdir/controller_addr.json``, which is no longer the default location. Make sure to provide this path to ``tool_manager`` when using it.
3. By default, the **molmoPoint** worker is configured to run in 4-bit mode to minimize VRAM usage. To customize GPU behavior or access advanced settings, you can log into the container and edit ``/app/OpenThinkIMG/tool_server/tool_workers/scripts/launch_scripts/config/service_apptainer.yaml``.

#### Option 1.1 Start Tool Server with Our Docker Image

We have released the docker image for `tool_server` and its slim version `tool_server_slim`. 

The `tool_server_slim` image is a lightweight version of the `tool_server` image, which removes the model weights to reduce the image size. To use it, manually prepare the model weights and place them under the `/weights` folder of the container as described in [Build Docker Image by Yourself](#option-12-build-docker-image-by-yourself) section. You can pull the image from either Aliyun or Docker Hub.

| Image         | Aliyun         | Docker Hub                  |
|------------------|-------------------|------------------------------------------|
| `tool_server`     | `crpi-fs6w5qkjtxy37mko.cn-shanghai.personal.cr.aliyuncs.com/hitsmy/tool_server:v0.1`             | `hitsmy/tool_server:v0.1`              |
| `tool_server_slim`| `crpi-fs6w5qkjtxy37mko.cn-shanghai.personal.cr.aliyuncs.com/hitsmy/tool_server_slim:v0.1`        | `hitsmy/tool_server_slim:v0.1`         |

```bash
# Pull the docker image and run
docker pull crpi-fs6w5qkjtxy37mko.cn-shanghai.personal.cr.aliyuncs.com/hitsmy/tool_server:v0.1
docker run -it \
  --gpus all \
  --name tool_server \
  -v /path/to/your/logdir:/log \
  -w /app/OpenThinkIMG \
  --network host \
  crpi-fs6w5qkjtxy37mko.cn-shanghai.personal.cr.aliyuncs.com/hitsmy/tool_server:v0.1 \
  bash -c \
  "python /app/OpenThinkIMG/tool_server/tool_workers/scripts/launch_scripts/start_server_local.py \
  --config /app/OpenThinkIMG/tool_server/tool_workers/scripts/launch_scripts/config/service_apptainer.yaml"

# Test the server 
pthon OpenThinkIMG/tool_server/tool_workers/online_workers/test_cases/worker_tests/test_all.py
```
#### Option 1.2 Build Docker Image by Yourself

We have provided the dockerfile at `OpenThinkIMG/Dockerfile`, you can build the docker Image according to it.

**sub-step 1** Prepare the weights


Some tools require specific pretrained weights. Please ensure that these model weights are prepared and placed in the appropriate paths before building the image. The image building process will copy them into the `/weights` folder of the container automatically.

The directory structure is organized as follows:

```bash
project-root/
├── weights/
│   ├── Molmo-7B-D-0924/ # allenai/Molmo-72B-0924
│   ├── sam2-hiera-large/ # facebook/sam2-hiera-large
│   ├── groundingdino_swint_ogc.pth # https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
│   └── GroundingDINO_SwinT_OGC.py # https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
├── OpenThinkIMG/
│   └── Dockerfile
```

**sub-step 2** Start the building procedure!

```bash 
git clone https://github.com/OpenThinkIMG/OpenThinkIMG.git
cd OpenThinkIMG
docker build -f Dockerfile -t tool_server:v0.1 ..  # This might take a while ...
```

**sub-step 3** Run the image and test!
```bash 
docker run -it \
  --gpus all \
  --name tool_server \
  -v /path/to/your/logdir:/log \
  -w /app/OpenThinkIMG/ \
  --network host \
  tool_server:v0.1 \

# Test the server 
pthon tool_server/tool_workers/online_workers/test_cases/worker_tests/test_all.py
```
### Option 2. Start Tool Server From Source Code
You can choose to start tool_server through SLURM or just run it on local machine.

### 🛠️ Installation
First of all, provide a pytorch-based environment.
* torch==2.0.1+cu118

```bash
# [Optional] Create a clean Conda environment
conda create -n tool-server python=3.10
conda activate tool-server
# Install PyTorch and dependencies (make sure CUDA version matches)
pip install -e git+https://github.com/facebookresearch/sam2.git
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install this project
git clone https://github.com/OpenThinkIMG/OpenThinkIMG.git
pip install -e OpenThinkIMG
pip install -r OpenThinkIMG/requirements/requirements.txt # Tool Server Requirements
```
⚠️ **Be aware:**

We **deliberately selected minimal dependencies** in this project to reduce the risk of conflicts. As a result, you may need to manually install any missing packages based on your environment.

#### Option 2.1 Start Tool Server through SLURM
It's recommended to start the tool server through SLURM because it's more flexible.
```bash
## First, modify the config to adapt to your own environment
## OpenThinkIMG/tool_server/tool_workers/scripts/launch_scripts/config/all_service_example.yaml

## Start all services
cd OpenThinkIMG/tool_server/tool_workers/scripts/launch_scripts
python start_server_config.py --config ./config/all_service_example.yaml

## Press Ctrl + C to shutdown all services automatically.
```

#### Option 2.2 Start Tool Server Locally
We made a slight modification to ``start_server_config.py`` to create ``start_server_local.py``, primarily by removing the logic related to SLURM job detection and adapting it for local execution instead.
```bash
## First, modify the config to adapt to your own environment
## OpenThinkIMG/tool_server/tool_workers/scripts/launch_scripts/config/all_service_example_local.yaml

## Start all services
cd OpenThinkIMG/tool_server/tool_workers/scripts/launch_scripts
python start_server_local.py --config ./config/all_service_example_local.yaml

## Press Ctrl + C to shutdown all services automatically.
```

You can then inspect the log files to diagnose and resolve any potential issues. Due to the complexity of this project, we cannot guarantee that it will run without errors on every machine.

### 🔍 Step 2: Run Inference with OpenThinkIMG

### 🛠️ Installation
First of all, provide a pytorch-vllm-based environment.
* vllm>=0.7.3
* torch==2.5.1+cu121
* transformers>=4.49.0
* flash_attn>=2.7.3

```bash
# [Optional] Create a clean Conda environment
conda create -n vllm python=3.10
conda activate tool-server
# Install PyTorch and dependencies (make sure CUDA version matches)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install this project
git clone https://github.com/OpenThinkIMG/OpenThinkIMG.git
pip install -e OpenThinkIMG
pip install -r OpenThinkIMG/requirements/inference_requirements.txt # Tool Server Requirements
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



## 🧠 Training

Once the vision tools are properly deployed, we provide a flexible training pipeline to teach models **how to plan and invoke tools** effectively through **SFT** and our proposed **V-ToolRL** methods.

Our training pipeline builds on the solid foundation of [OpenR1](https://github.com/OpenR1), integrating visual tools as external reasoning capabilities.

### 📦 Install Additional Dependencies

To run the training code, make sure to install the additional required packages:

```
pip install -r requirements/requirements_train.txt
```

### 🔁 V-ToolRL: Reinforcement Learning with Vision Tools

We provide a customized implementation of V-ToolRL for training models to leverage vision tools dynamically in complex tasks.

```
torchrun --nproc_per_node=${nproc_per_node} \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port=${master_port} \
    src/open_r1/tool_grpo.py --use_vllm True \
    --output_dir ${output_dir} \
    --model_name_or_path ${model_path} \
    --dataset_name ${data_path} \
    --max_prompt_length 16000 \
    --max_completion_length 2048 \
    --temperature 1.0 \
    --seed 42 \
    --learning_rate 1e-6 \
    --num_generations 8 \
    --lr_scheduler_type "constant" \
    --vllm_gpu_memory_utilization 0.8 \
    --deepspeed ${DS_CONFIG} \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 12 \
    --logging_steps 1 \
    --bf16 true \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 200000 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --controller_addr http://SH-IDCA1404-10-140-54-15:20001 \
    --use_tool true
```

> 📈  This helps the model learn **dynamic planning & tool invocation** using environment feedback.

### 🧪 Tool Cold Start (TC)

We also support supervised fine-tuning for training models on curated tool usage demonstrations. Modify the config according to your use case:

```
    accelerate launch --num_machines 1 --num_processes 6 --main_process_port 29502 --multi_gpu\
    src/open_r1/sft.py \
    --output_dir ${output_dir} \
    --model_name_or_path ${model_path} \
    --dataset_name ${data_path} \
    --seed 42 \
    --learning_rate 2e-5 \
    --max_seq_length 4096 \
    --deepspeed config/deepspeed/ds_z3_offload_config.json \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --bf16 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --warmup_ratio 0.1 \
    --save_only_model true
```



## 🚧 Project Status

OpenThinkIMG is currently an **alpha release** but is actively being developed. The core end-to-end system, including tool integration, trajectory generation, SFT (Cold-Start), and V-ToolRL training, is functional and can be used to replicate the results in our paper.

The project team is actively working on the following key milestones:

*   **🥇 Release of Pre-trained Models**: Providing readily usable SFT-initialized and V-ToolRL-trained agent models (e.g., based on Qwen2-VL-2B).
*   **🛠️ Expanding the Vision Toolset**: Integrating more diverse and powerful vision tools (e.g., advanced image editing, 3D analysis tools).
*   **🤖 Broader LVLM Backbone Support**: Adding easy integration for more open-source LVLMs (e.g., LLaVA series, MiniGPT-4).
*   **📊 More Benchmarks & Evaluation Suites**: Extending evaluation to a wider range of visual reasoning tasks beyond chart reasoning.
*   **🌐 Community Building**: Fostering an active community through GitHub discussions, contributions, and collaborations.

We welcome contributions and feedback to help us achieve these goals!

---



## 🔧 Vision Toolset

| **Tool**                    | **Input**                        | **Output**                             | **Description**                                                                                  |
|-----------------------------|----------------------------------|----------------------------------------|--------------------------------------------------------------------------------------------------|
| **GroundingDINO**           | image + text query               | bounding boxes                         | Object detection producing boxes for any target                                                  |
| **SAM**                     | image + bounding box             | segmentation mask                      | Generates precise segmentation masks based on provided regions                    |
| **OCR**                     | image                            | text strings + bounding boxes          | Optical character recognition for extracting text from images                                    |
| **Crop**                    | image + region coordinates       | cropped image                          | Extracts a sub-region of the image for focused analysis                                          |
| **Point**                   | image + target description       | point coordinates                      | Uses a model to predict the location of a specified object                                      |
| **DrawHorizontalLineByY**   | image + Y-coordinate             | annotated image                        | Draws a horizontal line at the given Y-coordinate                                                |
| **DrawVerticalLineByX**     | image + X-coordinate             | annotated image                        | Draws a vertical line at the given X-coordinate                                                  |
| **ZoominSubplot**           | image + description (title/pos)  | subplot images                 | Zoomin subplot(s) based on description                                                         |
| **SegmentRegionAroundPoint**| image + point coordinate         | localized mask                         | Refines segmentation around a specified point                                                    |
> 💡 More vision tools are coming soon!








