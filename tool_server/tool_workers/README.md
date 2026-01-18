
# Tool Server

Tool Server is the core component of the AdaReasoner framework, responsible for hosting and orchestrating all vision tools. It adopts a distributed architecture, supporting both Online Workers (online tools) and Offline Workers (offline tools).

---

## 📚 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tool Types](#tool-types)
- [Quick Start](#quick-start)
- [Deployment Methods](#deployment-methods)
- [Configuration Guide](#configuration-guide)
- [Tool Manager Usage](#tool-manager-usage)
- [Available Tools](#available-tools)
- [Custom Tool Development](#custom-tool-development)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
Tool Server Architecture
├── Controller
│   └── Manages tool registration, request routing, and load balancing
├── Online Workers
│   ├── GPU-required deep learning inference tools
│   ├── Run as independent processes
│   └── Communicate via HTTP API
├── Offline Workers
│   ├── Lightweight tools without GPU requirements
│   ├── Direct function calls
│   └── Managed by Tool Manager
└── Tool Manager
  └── Unified interface for managing and invoking all tools
```

---

## Tool Types

### 🌐 Online Workers

**Characteristics**:
- **GPU Required**: Used for deep learning model inference
- **Independent Processes**: Each tool runs in a separate service process
- **HTTP Communication**: Invoked via RESTful API
- **Resource Intensive**: Requires significant computational and memory resources
- **Concurrent Support**: Supports concurrent requests and load balancing

**Use Cases**:
- Tasks requiring deep learning models (e.g., object detection, image segmentation)
- Compute-intensive operations
- Tasks requiring large GPU memory

**Workflow**:
```
1. Controller receives request
2. Looks up Worker address by tool name
3. Forwards request to Worker
4. Worker performs inference and returns result
5. Controller returns result to client
```

### 💾 Offline Workers

**Characteristics**:
- **No GPU Required**: Uses CPU computation or simple image processing
- **Direct Invocation**: Function calls without network communication
- **Lightweight**: Fast startup with minimal resource usage
- **Synchronous Execution**: Returns results immediately
- **Easy Extension**: Just inherit from `BaseOfflineWorker`

**Use Cases**:
- Image processing operations (crop, rotate, draw)
- Algorithm tools (A* pathfinding)
- Coordinate transformations, geometric calculations
- Simple data processing

**Workflow**:
```
1. Tool Manager directly calls tool function
2. Tool executes computation/processing
3. Returns result (no network communication)
```

### Comparison Summary

| Feature | Online Workers | Offline Workers |
|---------|----------------|-----------------|
| **Resource Requirements** | High (GPU required) | Low (CPU only) |
| **Communication** | HTTP API | Direct function call |
| **Startup Time** | Slow (model loading) | Fast |
| **Memory Usage** | Large (model weights) | Small |
| **Extension Method** | Independent service | Inherit base class |
| **Concurrent Processing** | Load balancing supported | Controlled by caller |

---

## Quick Start

### Environment Setup

```bash
# Create Conda environment
conda create -n tool-server python=3.10
conda activate tool-server

# Install PyTorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Clone repository
git clone https://github.com/ssmisya/AdaReasoner.git

# Install dependencies
pip install -r AdaReasoner/requirements/requirements.txt 
pip install -e AdaReasoner
```

### Using Offline Tools Only

If you only need Offline tools, you can use them directly after installation without starting Tool Server.

You can invoke [Tool Manager](#tool-manager-usage) to call any offline tools directly.

### Starting Complete Tool Server

To use Online tools, start Tool Server first:

```bash
cd AdaReasoner/tool_server/tool_workers/scripts/launch_scripts

# Local startup
python start_server_local.py --config ./config/all_service_example_local.yaml

# Or via SLURM
python start_server_config.py --config ./config/all_service_example.yaml
```

---

## Deployment Methods

### Method 1: SLURM Cluster Deployment

Suitable for environments with SLURM clusters.

**Configuration Example** (`all_service_example.yaml`):

```yaml
base_dir: ./tool_server/tool_workers
log_folder: ./tool_server/tool_workers/logs/server_log
partition: "ai_moe2"
default_calculate_gpus: 1
default_calculate_cpus: 12
default_control_cpus: 2
default_control_gpus: 0
default_heavy_calculate_gpus: 4
default_heavy_calculate_cpus: 32
retry_interval: 1
request_timeout: 10

controller_config:
  worker_name: controller
  job_name: controller
  calculate_type: control
  conda_env: "tool_server"
  srun_kwargs:
    w: "SH-IDC1-10-140-37-138"  # Specific node
  cmd:
    script-addr: ./online_workers/controller.py
    port: 21112
    host: "0.0.0.0"

tool_worker_config:
  - Point:
      worker_name: Point
      job_name: point
      calculate_type: control
      conda_env: "tool-server"
      srun_kwargs:
        w: "SH-IDC1-10-140-37-82"
      cuda_visible_devices: "4,5,6,7"
      cmd:
        script-addr: tool_server.tool_workers.online_workers.point_worker_parallel
        port: 50002
        host: "0.0.0.0"
        model_path: allenai/Molmo-7B-D-0924

  - OCR:
      worker_name: OCR
      job_name: ocr
      calculate_type: calculate
      node_list: "SH-IDC1-10-140-37-138"
      cuda_visible_devices: "6,7"
      use_apptainer: true
      apptainer_image: "./images/ubuntu20.04-py3.10-cuda11.8-cudnn8-transformer4.30.0_v1.0.0.sif"
      conda_env_path: "./envs/tool-server"
      cmd:
        script-addr: tool_server.tool_workers.online_workers.ocr_worker
        port: 20009
        host: "0.0.0.0"
        gpu_ids: "6,7"
        enable_multi_gpu: true
        max_concurrency: 4
```

> [!IMPORTANT]
> **OCR Tool Requirements**: The OCR tool relies on PaddlePaddle. Please ensure you have installed the correct version in your environment:
> - `paddlepaddle-gpu==3.0.0`
> - `paddleocr==3.2.0`
>
> Refer to the [official documentation](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html) for installation details.

**Start Service**:

You can use the provided script to start the server. It is recommended to run this in a tmux session:

```bash
tmux send-keys -t $session_name "python -m tool_server.tool_workers.scripts.launch_scripts.start_server_config --config tool_server/tool_workers/scripts/launch_scripts/config/all_service_example.yaml" C-m
```

### Method 2: Local Deployment

**Configuration Example** (`all_service_local.yaml`):

```yaml
base_dir: /path/to/AdaReasoner/tool_server/tool_workers
log_folder: ./logs
retry_interval: 1
request_timeout: 10

controller_config:
  worker_name: controller
  job_name: controller
  calculate_type: control
  controller_addr_location: ./controller_addr.json
  cmd:
  script-addr: ./online_workers/controller.py
  port: 20001
  host: "0.0.0.0"

tool_worker_config:
  - Point:
    worker_name: Point
    job_name: point
    calculate_type: control
    cuda_visible_devices: "0"
    cmd:
    script-addr: ./online_workers/molmo_point_worker.py
    port: 20027
    host: "0.0.0.0"
    model_path: /path/to/Molmo-7B-D-0924
    load-4bit: yes

  - OCR:
    worker_name: OCR
    job_name: ocr
    calculate_type: control
    cuda_visible_devices: "1"
    cmd:
    script-addr: ./online_workers/ocr_worker.py
    port: 20009
    host: "0.0.0.0"
```

**Start Service**:

```bash
cd AdaReasoner/tool_server/tool_workers/scripts/launch_scripts
python start_server_local.py --config ./config/all_service_local.yaml

# Press Ctrl+C to stop all services
```

---

## Configuration Guide

### Basic Configuration

```yaml
# Required fields
base_dir: /path/to/tool_workers        # Tool Server working directory
log_folder: /path/to/logs              # Log directory

# Optional fields
retry_interval: 1                      # Retry interval (seconds)
request_timeout: 10                    # Request timeout (seconds)
```

### Controller Configuration

```yaml
controller_config:
  worker_name: controller              # Controller name
  job_name: controller                 # Job name
  calculate_type: control              # Calculation type
  controller_addr_location: /path/to/controller_addr.json  # Address save location
  conda_env: "tool-server"            # Conda environment (optional)
  
  cmd:
  script-addr: ./online_workers/controller.py
  port: 20001                        # Controller port
  host: "0.0.0.0"                   # Listen address
```

### Worker Configuration Template

```yaml
tool_worker_config:
  - ToolName:
    worker_name: ToolName            # Worker name
    job_name: tool_job              # Job name
    calculate_type: calculate       # Type: control/calculate/heavy_calculate
    cuda_visible_devices: "0,1"     # GPUs to use (optional)
    conda_env: "tool-server"        # Conda environment (optional)
    
    cmd:
    script-addr: ./online_workers/tool_worker.py
    port: 20XXX                    # Worker port
    host: "0.0.0.0"
    model_path: /path/to/model    # Model path
    load-4bit: yes                # 4-bit quantization
```



## Tool Manager Usage

Tool Manager is a unified interface for managing and invoking all tools. It handles:

1. **Tool Routing**: Automatically determines whether to call Online or Offline tools based on tool name
2. **Image Management**: Maintains image state within sessions
3. **Error Handling**: Unified error handling and return format
4. **Load Balancing**: Load balances requests for Online tools

### Initialize Tool Manager

```python
from tool_server.tool_workers.tool_manager.base_manager import ToolManager

# Initialize (specify Controller address to use Online tools)
tool_manager = ToolManager(
  controller_addr="http://localhost:20001"
)

# For Offline tools only, controller_addr is optional
tool_manager = ToolManager()
```

### Calling Online Tools

```python
# Call Point tool (locate points in image)
result = tool_manager.call_tool("Point", {
  "image": "img_1",           # Image ID
  "description": "the dog's nose"  # Point description
})

print(result)
# {
#     "tool_response_from": "Point",
#     "status": "success",
#     "points": [{"x": 123.5, "y": 456.7}],
#     "error_code": 0
# }

# Call OCR tool
result = tool_manager.call_tool("OCR", {
  "image": "img_1"
})

# Call Crop tool
result = tool_manager.call_tool("Crop", {
  "image": "img_1",
  "coordinates": "[100, 100, 300, 300]"
})
```

### Calling Offline Tools

```python
# Call A* pathfinding algorithm
result = tool_manager.call_tool("AStarWithPixelCoordinate", {
  "start": [100, 200],
  "goal": [300, 400],
  "obstacles": [[150, 150], [200, 250]]
})

print(result)
# {
#     "tool_response_from": "AStarWithPixelCoordinate",
#     "status": "success",
#     "path": "R,R,U,L,D,D",
#     "error_code": 0
# }

# Call path drawing tool
result = tool_manager.call_tool("Draw2DPath", {
  "image": "img_1",
  "start_point": [100, 100],
  "directions": "R,R,D,D,L,U",
  "step": 50,
  "line_width": 3,
  "line_color": "red"
})
```

### Direct Use of Offline Tool Functions

```python
from tool_server.tool_workers.offline_workers import get_tool_generate_fn, get_available_tools

# View all available Offline tools
available_tools = get_available_tools()
print(available_tools)

# Get tool function
crop_fn = get_tool_generate_fn("Crop")
astar_fn = get_tool_generate_fn("AStarWithPixelCoordinate")

# Direct invocation
result = crop_fn({
  "image": image_base64,
  "coordinates": "[100, 100, 300, 300]"
})
```

### Get Tool Instructions

```python
from tool_server.tool_workers.offline_workers import get_all_tool_instructions

# Get instructions for all tools (for prompt construction)
instructions = get_all_tool_instructions()

for tool_name, instruction in instructions.items():
  print(f"Tool: {tool_name}")
  print(f"Description: {instruction['function']['description']}")
  print(f"Parameters: {instruction['function']['parameters']}")
```

---

## Available Tools

### Online Workers

| Tool Name | Description | GPU Requirement | Model |
|-----------|-------------|-----------------|-------|
| **Point** | Point localization | 1 GPU (4bit) | Molmo-7B |
| **GroundingDINO** | Object detection | 1 GPU | DINO-SwinT |
| **SAM2** | Image segmentation | 1 GPU | SAM2-Hiera-L |
| **OCR** | Text recognition | 1 GPU | PaddleOCR |
| **Crop** | Image cropping | No GPU | - |
| **DrawShape** | Shape drawing | No GPU | - |
| **ZoomInSubfigure** | Subfigure selection | No GPU | - |

### Offline Workers

| Tool Name | Description | Parameters |
|-----------|-------------|------------|
| **AStarWithPixelCoordinate** | A* pathfinding algorithm | start, goal, obstacles |
| **Draw2DPath** | 2D path drawing | image, start_point, directions, step |
| **DetectBlackArea** | Black area detection | image, threshold, min_area |
| **InsertImage** | Image insertion | base_image, image_to_insert, coordinates |
| **RotateImage** | Image rotation | image, angle |
| **Crop** | Image cropping | image, coordinates |

---

## Custom Tool Development

### Developing Offline Workers

```python
# my_custom_tool.py
from tool_server.tool_workers.offline_workers.base_offline_worker import BaseOfflineWorker

class MyCustomTool(BaseOfflineWorker):
  def __init__(self):
    super().__init__(model_name="MyCustomTool")
    self.instruction = {
      "type": "function",
      "function": {
        "name": self.model_name,
        "description": "Description of my custom tool",
        "parameters": {
          "type": "object",
          "properties": {
            "param1": {
              "type": "string",
              "description": "Description of parameter 1"
            },
            "param2": {
              "type": "integer",
              "description": "Description of parameter 2"
            }
          },
          "required": ["param1"]
        }
      }
    }
  
  def _execute(self, params):
    """Implement core tool logic"""
    param1 = params["param1"]
    param2 = params.get("param2", 0)
    
    # Processing logic
    result = self.process(param1, param2)
    
    return {
      "status": "success",
      "result": result,
      "error_code": 0
    }
  
  def verify_tool_parameter(self, params):
    """Optional: Validate parameters"""
    try:
      # Validation logic
      return {
        "params_qualified_reward": 1,
        "params_qualified": True,
        "new_params": params
      }
    except Exception as e:
      return {
        "params_qualified_reward": 0,
        "params_qualified": False,
        "error_info": str(e),
        "new_params": None
      }
```

**Register Tool**:

```python
# Add to offline_workers/__init__.py
from .my_custom_tool import MyCustomTool

offline_tool_instances = {
  "MyCustomTool": MyCustomTool(),
  # ... other tools
}
```

### Developing Online Workers

```python
# my_online_tool.py
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.worker_arguments import WorkerArguments
from dataclasses import dataclass, field

@dataclass
class MyToolArguments(WorkerArguments):
  custom_param: str = field(default="default_value")

class MyOnlineToolWorker(BaseToolWorker):
  def __init__(self, worker_arguments: MyToolArguments = None):
    if worker_arguments and worker_arguments.model_name is None:
      worker_arguments.model_name = "MyOnlineTool"
    super().__init__(worker_arguments)
    
    self.instruction = {
      "type": "function",
      "function": {
        "name": self.model_name,
        "description": "Description of my online tool",
        "parameters": {
          "type": "object",
          "properties": {
            "image": {
              "type": "string",
              "description": "Image identifier"
            }
          },
          "required": ["image"]
        }
      }
    }
  
  def init_model(self):
    """Initialize model"""
    # Load model
    self.model = load_my_model(self.args.model_path)
  
  def generate(self, params):
    """Process request"""
    image = params["image"]
    
    # Model inference
    result = self.model.predict(image)
    
    return {
      "tool_response_from": self.model_name,
      "status": "success",
      "result": result,
      "error_code": 0
    }

if __name__ == "__main__":
  from transformers import HfArgumentParser
  parser = HfArgumentParser((MyToolArguments,))
  args, = parser.parse_args_into_dataclasses()
  
  worker = MyOnlineToolWorker(worker_arguments=args)
  worker.run()
```

---

## Troubleshooting

### 1. Controller Fails to Start

**Check**:
```bash
# Check if port is in use
netstat -tunlp | grep 20001

# View logs
tail -f logs/controller.log
```

**Solutions**:
- Modify port number in configuration
- Kill process using the port: `kill -9 <PID>`

### 2. Worker Registration Fails

**Check**:
```bash
# Test Controller connection
curl -X POST http://localhost:20001/list_models

# View Worker logs
tail -f logs/Point_worker.log
```

**Solutions**:
- Confirm Controller is running
- Check network connectivity
- Verify Controller address configuration

### 3. GPU Out of Memory

**Solutions**:
```yaml
# Enable 4-bit quantization
cmd:
  load-4bit: yes
  
# Or enable 8-bit quantization
cmd:
  load-8bit: yes
```

### 4. Check Service Status

```bash
# Check all services
curl -X POST http://localhost:20001/list_models

# Check specific Worker
curl -X POST http://localhost:20001/get_worker_address \
  -H "Content-Type: application/json" \
  -d '{"model": "Point"}'
```

### 5. Docker Cannot Access GPU

```bash
# Install nvidia-docker2
sudo apt-get install nvidia-docker2
sudo systemctl restart docker

# Verify GPU availability
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

---

## Advanced Configuration

### Multi-GPU Parallel

```yaml
- Point:
  cuda_visible_devices: "0,1,2,3"
  cmd:
    gpu_count: 4
```

### Load Balancing

Controller automatically supports load balancing for multiple instances of the same tool:

```yaml
- Point_1:
  worker_name: Point
  cuda_visible_devices: "0"
  cmd:
    port: 20027

- Point_2:
  worker_name: Point
  cuda_visible_devices: "1"
  cmd:
    port: 20028
```

---

## Related Resources

- [Full Documentation](https://github.com/ssmisya/AdaReasoner/tree/main/docs)
- [GitHub Issues](https://github.com/ssmisya/AdaReasoner/issues)
- [HuggingFace Models and Data](https://huggingface.co/collections/hitsmy/adareasoner)

---

For questions or assistance, please submit an Issue or participate in Discussions on GitHub.

