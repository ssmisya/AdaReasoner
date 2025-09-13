import importlib
import os
import sys

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="WARNING")

AVAILABLE_MODELS = {
    "qwen2vl": "Qwen2VL",
    "gemini": "GeminiModels",
    "gemini_create_data": "Gemini_create_data_Models",
    "openai": "OpenaiModels",
    "llava_plus": "LLaVA_Plus",
    "lmdeploy_models": "LMDeployModels",
    "vllm_models": "VllmModels",
    "vllm_models_no_conversation": "VllmModelsNoConversation",
}


def get_model(model_name):
    print("运行到get_model这里了,model_name:", model_name)
    if model_name not in AVAILABLE_MODELS:
        print(f"Model {model_name} not found in available models.")
        raise ValueError(f"Model {model_name} not found in available models.")

    model_class = AVAILABLE_MODELS[model_name]
    print("运行到get_model这里了,model_class:", model_class)
    try:
        module = __import__(f"tool_server.tf_eval.models.{model_name}", fromlist=[model_class])
        return getattr(module, model_class)
    except Exception as e:
        logger.error(f"Failed to import {model_class} from {model_name}: {e}")
        raise
