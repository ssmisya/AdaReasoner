import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser


logger = logging.getLogger(__name__)

@dataclass
class WorkerArguments:
    """
    Arguments for the worker setup
    """
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to bind the server"}
    )
    port: int = field(
        default=20009,
        metadata={"help": "Port to bind the server"}
    )
    worker_addr: str = field(
        default="auto",
        metadata={"help": "Worker address"}
    )
    controller_addr: str = field(
        default="http://localhost:20001",
        metadata={"help": "Controller address"}
    )
    limit_model_concurrency: int = field(
        default=5,
        metadata={"help": "Limit for model concurrency"}
    )
    no_register: bool = field(
        default=False,
        metadata={"help": "Do not register to controller"}
    )
    model_path: str = field(
        default=None,
        metadata={"help": "Path to the model"}
    )
    model_base: Optional[str] = field(
        default=None,
        metadata={"help": "Base model path"}
    )
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the model, derived from model_path if not provided"}
    )
    load_8bit: bool = field(
        default=False,
        metadata={"help": "Load model in 8-bit mode"}
    )
    load_4bit: bool = field(
        default=False,
        metadata={"help": "Load model in 4-bit mode"}
    )
    device: str = field(
        default="cuda",
        metadata={"help": "Device to use for model (cuda, cpu, etc.)"}
    )
    task_timeout: int = field(
        default=120,
        metadata={"help": "Timeout for tasks in seconds"}
    )
    wait_timeout: int = field(
        default=60,
        metadata={"help": "Timeout for semaphore waiting in seconds"}
    )
