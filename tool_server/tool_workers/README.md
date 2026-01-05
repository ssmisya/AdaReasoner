# Tool Server Setup and Configuration Guide

This document provides comprehensive instructions on how to start and configure the OpenThinkIMG Tool Server, along with explanations of Online Workers and Offline Workers.

---

## 📚 Table of Contents

- Overview
- Understanding Tool Types
- [Deployment Methods](#deployment-methods)
  - Method 1: Docker Deployment
  - Method 2: SLURM Cluster Deployment
  - Method 3: Local Deployment
- Configuration Files
- Available Tools
- Troubleshooting

---

## Overview

The Tool Server is the core component of OpenThinkIMG, responsible for hosting and orchestrating all visual tools. It uses a distributed architecture with the following components:

```
Tool Server Architecture
├── Controller
│   └── Manages tool registration, request routing, and load balancing
├── Online Workers
│   ├── GPU-intensive tools requiring model inference
│   ├── Run as independent processes
│   └── Communicate via HTTP API
└── Offline Workers
    ├── Lightweight tools without GPU requirements
    ├── Direct function calls
    └── Managed by Tool Manager
```

---

## Understanding Tool Types

### 🌐 Online Workers

**Characteristics**:
- **GPU Required**: Used for deep learning model inference
- **Independent Processes**: Each tool runs in a separate service process
- **HTTP Communication**: Called through RESTful API
- **Resource Intensive**: Requires significant compute and memory resources
- **Asynchronous Processing**: Supports concurrent requests and load balancing

**Use Cases**:
- Tasks requiring deep learning models (e.g., object detection, segmentation)
- Compute-intensive operations
- Tasks requiring large VRAM

**Example Tools**:
```python
# Online Workers Examples
├── Point (Molmo-7B)           # Point localization
├── GroundingDINO             # Object detection
├── SAM2                      # Image segmentation
├── OCR (PaddleOCR)          # Text recognition
└── ...
```

**Workflow**:
```
1. Controller receives request
2. Finds Worker address based on tool name
3. Forwards request to Worker
4. Worker performs inference and returns result
5. Controller returns result to client
```

---

### 💾 Offline Workers

**Characteristics**:
- **No GPU Required**: Uses CPU computation or simple image processing
- **Direct Invocation**: Function calls without network communication
- **Lightweight**: Fast startup, minimal resource usage
- **Synchronous Execution**: Returns results immediately
- **Easy to Extend**: Inherit from `BaseOfflineWorker`

**Use Cases**:
- Image processing operations (crop, rotate, draw)
- Algorithm tools (A* pathfinding)
- Coordinate transformation, geometric calculations
- Simple data processing

**Example Tools**:
```python
# Offline Workers Examples
├── Crop                      # Image cropping
├── AStarWithPixelCoordinate # A* pathfinding algorithm
├── Draw2DPath               # Path drawing
├── DetectBlackArea          # Black area detection
├── InsertImage              # Image insertion
├── RotateImage              # Image rotation
└── ...
```

**Workflow**:
```
1. Tool Manager directly calls tool function
2. Tool executes computation/processing
3. Returns result (no network communication)
```

---

### Comparison Summary

| Feature | Online Workers | Offline Workers |
|---------|----------------|-----------------|
| **Resource Requirements** | High (GPU required) | Low (CPU only) |
| **Communication** | HTTP API | Direct function call |
| **Startup Time** | Slow (model loading) | Fast |
| **Memory Usage** | Large (model weights) | Small |
| **Extension Method** | Independent service | Inherit base class |
| **Concurrent Processing** | Load balancing supported | Controlled by caller |
| **Suitable Tasks** | Deep learning inference | Image processing/algorithms |

---

## Deployment Methods

### Method 1: Docker Deployment (Recommended)

#### Using Pre-built Images

**Pull the Image**:

```bash
# Pull from Aliyun (recommended for China)
docker pull crpi-fs6w5qkjtxy37mko.cn-shanghai.personal.cr.aliyuncs.com/hitsmy/tool_server:v0.1

# Or pull from Docker Hub
docker pull hitsmy/tool_server:v0.1
```

**Start the Service**:

```bash
docker run -it \
  --gpus all \
  -p 20001:20001 \
  -v /path/to/your/logdir:/log \
  --name tool_server \
  crpi-fs6w5qkjtxy37mko.cn-shanghai.personal.cr.aliyuncs.com/hitsmy/tool_server:v0.1 \
  bash -c "cd /app/OpenThinkIMG/tool_server/tool_workers/scripts/launch_scripts && \
  python start_server_local.py \
  --config /app/OpenThinkIMG/tool_server/tool_workers/scripts/launch_scripts/config/service_apptainer.yaml"
```

**Parameter Explanation**:
- `--gpus all`: Use all available GPUs
- `-p 20001:20001`: Map Controller port
- `-v /path/to/your/logdir:/log`: Mount log directory (optional)
- `service_apptainer.yaml`: Docker-specific configuration

**Test the Service**:

```bash
# Enter the container
docker exec -it tool_server bash

# Run tests
cd /app/OpenThinkIMG
python tool_server/tool_workers/online_workers/test_cases/worker_tests/test_all.py
```

#### Using Lightweight Images

If you need to customize model weights:

```bash
# Pull slim version
docker pull hitsmy/tool_server_slim:v0.1

# Start with mounted weights directory
docker run -it \
  --gpus all \
  -p 20001:20001 \
  -v /path/to/weights:/weights \
  -v /path/to/logdir:/log \
  hitsmy/tool_server_slim:v0.1
```

**Weights Directory Structure**:
```
weights/
├── Molmo-7B-D-0924/              # Molmo Point model
├── sam2-hiera-large/             # SAM2 model
├── groundingdino_swint_ogc.pth   # GroundingDINO weights
└── GroundingDINO_SwinT_OGC.py    # GroundingDINO config
```

#### Building Your Own Image

```bash
# Prepare weight files (see directory structure above)
cd OpenThinkIMG
docker build -f Dockerfile -t tool_server:custom ..
```

---

### Method 2: SLURM Cluster Deployment

Suitable for environments with SLURM cluster access.

#### Install Dependencies

```bash
# Create conda environment
conda create -n tool-server python=3.10
conda activate tool-server

# Install PyTorch
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install SAM2
pip install -e git+https://github.com/facebookresearch/sam2.git

# Install Tool Server
git clone https://github.com/OpenThinkIMG/OpenThinkIMG.git
pip install -e OpenThinkIMG
pip install -r OpenThinkIMG/requirements/requirements.txt
```

#### Configuration File

Create or modify `all_service_example.yaml`:

```yaml
base_dir: /path/to/OpenThinkIMG/tool_server/tool_workers
log_folder: /path/to/logs
partition: "your_slurm_partition"  # SLURM partition name
default_calculate_gpus: 1
default_calculate_cpus: 16
default_control_cpus: 2
default_control_gpus: 0
retry_interval: 1
request_timeout: 10
current_node: &node your-node-name  # Specify node

# Controller configuration
controller_config:
  worker_name: controller
  job_name: controller
  calculate_type: control
  conda_env: "tool-server"
  srun_kwargs:
    w: *node  # Run on specified node
  cmd:
    script-addr: ./online_workers/controller.py
    port: 20001
    host: "0.0.0.0"

# Tool Workers configuration
tool_worker_config:
  - Point:
      worker_name: Point
      job_name: point
      calculate_type: calculate
      conda_env: "tool-server"
      cuda_visible_devices: "0"  # Specify GPU
      srun_kwargs:
        w: *node
      cmd:
        script-addr: ./online_workers/molmo_point_worker.py
        port: 20027
        host: "0.0.0.0"
        model_path: /path/to/Molmo-7B-D-0924
        load-4bit: yes
        
  - GroundingDINO:
      worker_name: GroundingDINO
      job_name: groundingdino
      calculate_type: calculate
      conda_env: "tool-server"
      cuda_visible_devices: "1"
      srun_kwargs:
        w: *node
      cmd:
        script-addr: ./online_workers/grounding_dino_worker.py
        port: 20003
        host: "0.0.0.0"
        model-path: /path/to/groundingdino_swint_ogc.pth
        model-config: /path/to/GroundingDINO_SwinT_OGC.py
```

#### Start the Service

```bash
cd OpenThinkIMG/tool_server/tool_workers/scripts/launch_scripts

# Start all services
python start_server_config.py --config ./config/all_service_example.yaml

# View logs
tail -f /path/to/logs/controller.log
tail -f /path/to/logs/Point_worker.log
```

#### Stop the Service

Press `Ctrl+C` to automatically stop all SLURM jobs.

---

### Method 3: Local Deployment

Suitable for local development and testing.

#### Install Dependencies

```bash
# Same as SLURM installation steps
conda create -n tool-server python=3.10
conda activate tool-server
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -e git+https://github.com/facebookresearch/sam2.git
pip install -e OpenThinkIMG
pip install -r OpenThinkIMG/requirements/requirements.txt
```

#### Configuration File

Create local configuration `all_service_local.yaml`:

```yaml
base_dir: /path/to/OpenThinkIMG/tool_server/tool_workers
log_folder: ./logs
retry_interval: 1
request_timeout: 10

# Controller configuration
controller_config:
  worker_name: controller
  job_name: controller
  calculate_type: control
  controller_addr_location: ./controller_addr.json
  cmd:
    script-addr: ./online_workers/controller.py
    port: 20001
    host: "0.0.0.0"

# Tool Workers configuration
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

#### Start the Service

```bash
cd OpenThinkIMG/tool_server/tool_workers/scripts/launch_scripts

# Start all services
python start_server_local.py --config ./config/all_service_local.yaml
```

#### Stop the Service

Press `Ctrl+C` to automatically stop all processes.

---

## Configuration Files

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
  - ToolName:                          # Tool name
      worker_name: ToolName            # Worker name
      job_name: tool_job              # Job name
      calculate_type: calculate       # Type: control/calculate/heavy_calculate
      cuda_visible_devices: "0,1"     # GPUs to use (optional)
      conda_env: "tool-server"        # Conda environment (optional)
      
      # SLURM-specific (optional)
      srun_kwargs:
        w: node-name                   # Specify node
      
      cmd:
        script-addr: ./online_workers/tool_worker.py  # Worker script
        port: 20XXX                    # Worker port
        host: "0.0.0.0"
        # Tool-specific parameters
        model_path: /path/to/model
        load-4bit: yes
```

### Calculate Type Explanation

- **control**: Lightweight tools, no GPU or minimal GPU resources
  - CPU: 2 cores
  - GPU: 0
  
- **calculate**: Standard compute tools, moderate GPU resources
  - CPU: 16 cores
  - GPU: 1
  
- **heavy_calculate**: Heavy compute tools, large GPU resources
  - CPU: 32 cores
  - GPU: 4

---

## Available Tools

### Online Workers

| Tool Name | Description | GPU Required | Model |
|-----------|-------------|--------------|-------|
| **Point** | Point localization | 1 GPU (4bit) | Molmo-7B |
| **GroundingDINO** | Object detection | 1 GPU | DINO-SwinT |
| **SAM2** | Image segmentation | 1 GPU | SAM2-Hiera-L |
| **OCR** | Text recognition | 1 GPU | PaddleOCR |
| **Crop** | Image cropping | No GPU | - |
| **ZoomInSubfigure** | Subfigure selection | No GPU | - |
| **DrawHorizontalLineByY** | Draw horizontal line | No GPU | - |
| **DrawVerticalLineByX** | Draw vertical line | No GPU | - |

### Offline Workers

| Tool Name | Description |
|-----------|-------------|
| **AStarWithPixelCoordinate** | A* pathfinding algorithm |
| **Draw2DPath** | 2D path drawing |
| **DetectBlackArea** | Black area detection |
| **InsertImage** | Image insertion |
| **RotateImage** | Image rotation |
| **GetStartPoint** | Get start point |
| **GetEndPoint** | Get end point |
| **GetObstacles** | Get obstacles |
| **DrawDashLinePath** | Draw dashed path |
| **GetWeather** | Get weather (example) |

---

## Usage Examples

### Calling Online Workers

```python
from tool_server.tool_workers.tool_manager.base_manager import ToolManager

# Initialize Tool Manager
tool_manager = ToolManager(
    controller_addr="http://localhost:20001"
)

# Call Point tool
result = tool_manager.call_tool("Point", {
    "image": "img_1",  # Image ID
    "description": "the dog's nose"  # Point description
})

print(result)
# {
#     "tool_response_from": "Point",
#     "status": "success",
#     "points": [{"x": 123.5, "y": 456.7}],
#     "error_code": 0
# }
```

### Calling Offline Workers

```python
from tool_server.tool_workers.offline_workers import get_tool_generate_fn

# Get tool function
astar_fn = get_tool_generate_fn("AStarWithPixelCoordinate")

# Call tool
result = astar_fn({
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
```

---

## Troubleshooting

### 1. Controller Won't Start

**Problem**: Controller fails to start

**Check**:
```bash
# Check if port is occupied
netstat -tunlp | grep 20001

# View logs
tail -f logs/controller.log
```

**Solution**:
- Change port number in configuration
- Kill process occupying port: `kill -9 <PID>`

### 2. Worker Registration Failed

**Problem**: Worker cannot register with Controller

**Check**:
```bash
# Test Controller connection
curl -X POST http://localhost:20001/list_models

# View Worker logs
tail -f logs/Point_worker.log
```

**Solution**:
- Confirm Controller is running
- Check network connectivity
- Verify Controller address configuration

### 3. GPU Out of Memory

**Problem**: CUDA Out of Memory error

**Solution**:
```yaml
# Enable 4bit quantization
cmd:
  load-4bit: yes
  
# Or enable 8bit quantization
cmd:
  load-8bit: yes
```

### 4. Model Loading Failed

**Problem**: Cannot load model weights

**Check**:
```bash
# Verify model path
ls -lh /path/to/model

# Check permissions
chmod -R 755 /path/to/model
```

### 5. Docker Cannot Access GPU

**Problem**: GPU unavailable in Docker

**Solution**:
```bash
# Install nvidia-docker2
sudo apt-get install nvidia-docker2
sudo systemctl restart docker

# Verify GPU availability
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

### 6. Check Service Status

```bash
# Check all services
curl -X POST http://localhost:20001/list_models

# Check specific Worker
curl -X POST http://localhost:20001/get_worker_address \
  -H "Content-Type: application/json" \
  -d '{"model": "Point"}'
```

---

## Advanced Configuration

### Multi-GPU Parallel

```yaml
- Point:
    cuda_visible_devices: "0,1,2,3"  # Use 4 GPUs
    cmd:
      gpu_count: 4                   # Parallel count
```

### Load Balancing

Controller automatically supports load balancing when multiple instances of the same tool exist:

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

### Custom Offline Worker

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
                "description": "My custom tool description",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "Parameter 1"}
                    },
                    "required": ["param1"]
                }
            }
        }
    
    def _execute(self, params):
        # Implement tool logic
        result = self.process(params["param1"])
        return {
            "status": "success",
            "result": result,
            "error_code": 0
        }
```

Register the tool:

```python
# __init__.py
from .my_custom_tool import MyCustomTool

offline_tool_instances = {
    "MyCustomTool": MyCustomTool(),
    # ... other tools
}
```

---

## Best Practices

1. **Resource Allocation**:
   - Allocate dedicated GPUs for heavy models (Point, SAM2)
   - Lightweight tools (OCR, Crop) can share GPUs
   - Recommend running Controller on CPU

2. **Log Management**:
   - Regularly clean up log files
   - Use log rotation: `logrotate`

3. **Monitoring**:
   - Monitor GPU usage: `nvidia-smi`
   - Monitor process status: `ps aux | grep python`
   - Monitor ports: `netstat -tunlp`

4. **Performance Optimization**:
   - Use quantization (4bit/8bit) to reduce VRAM usage
   - Enable model concurrency: `limit_model_concurrency`
   - Use load balancing to distribute requests

---

## Related Resources

- Full Documentation
- API Reference
- Tool Development Guide
- [GitHub Issues](https://github.com/OpenThinkIMG/OpenThinkIMG/issues)

---

For questions or assistance, please submit an Issue or participate in Discussions on GitHub.