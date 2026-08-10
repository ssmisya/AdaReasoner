import base64
import io
import os
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from urllib.parse import urlparse

from openai import OpenAI
from PIL import Image

from .abstract_model import tp_model
from ..tool_inferencer.dynamic_batch_manager import DynamicBatchItem
from ..utils.log_utils import get_logger
from ..utils.utils import load_image

inferencer_id = str(uuid.uuid4())[:6]
logger = get_logger(__name__)


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class OpenaiModels(tp_model):
    """OpenAI-compatible client used by the shared vLLM-server backend."""

    def __init__(
        self,
        pretrained: str = "gpt-4o",
        tensor_parallel: int = 1,
        limit_mm_per_prompt: int = 10,
        max_retry: int = 5,
        temperature: float = None,
        custom_system_prompt: str = None,
        seed: int = 0,
        request_concurrency: int = 8,
        request_timeout: float = 300,
        retry_base_seconds: float = 1,
        **kwargs,
    ):
        del tensor_parallel, custom_system_prompt, kwargs
        self.model_name = pretrained
        self.max_retry = max(1, _as_int(max_retry, 5))
        self.temperature = _as_float(temperature, 0.0) if temperature is not None else 0.0
        self.seed = _as_int(seed, 0)
        self.limit_mm_per_prompt = max(1, _as_int(limit_mm_per_prompt, 10))
        self.request_concurrency = max(1, _as_int(request_concurrency, 8))
        self.request_timeout = max(1.0, _as_float(request_timeout, 300.0))
        self.retry_base_seconds = max(0.0, _as_float(retry_base_seconds, 1.0))
        self.system_prompt = None

        vllm_base_url = os.environ.get("VLLM_BASE_URL")
        openai_base_url = os.environ.get("OPENAI_API_URL")
        self.base_url = vllm_base_url or openai_base_url or os.environ.get("BASE_URL")
        if not self.base_url:
            raise ValueError(
                "No API base URL found. Set VLLM_BASE_URL, OPENAI_API_URL, or BASE_URL."
            )
        self.base_url = self.base_url.rstrip("/")
        if self.base_url.endswith("/chat/completions"):
            self.base_url = self.base_url[: -len("/chat/completions")]
        if urlparse(self.base_url).hostname == "api.openai.com":
            raise ValueError(
                "The OpenAI official API is disabled for this evaluation. "
                "Use OPENAI_API_URL=https://yunwu.ai/v1 instead."
            )

        if vllm_base_url:
            # Never rotate the remote yunwu credential into local vLLM calls.
            candidate_keys = (
                os.environ.get("VLLM_API_KEY")
                or os.environ.get("OPENAI_API_KEY"),
            )
        else:
            candidate_keys = (os.environ.get("OPENAI_API_KEY"),)
        self.api_keys = list(dict.fromkeys(key for key in candidate_keys if key))
        if not self.api_keys:
            raise ValueError(
                "No API key found. Set VLLM_API_KEY for a local vLLM server or "
                "OPENAI_API_KEY for a remote OpenAI-compatible endpoint."
            )

        self._clients = [
            OpenAI(
                base_url=self.base_url,
                api_key=key,
                timeout=self.request_timeout,
                max_retries=0,
            )
            for key in self.api_keys
        ]
        self._client_index = 0
        self._client_lock = threading.Lock()
        self.generation_config = {
            "max_new_tokens": 2048,
            "temperature": self.temperature,
        }

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def _next_client(self):
        with self._client_lock:
            client = self._clients[self._client_index % len(self._clients)]
            self._client_index += 1
            return client

    @staticmethod
    def _encode_pil(image):
        image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _process_image(self, image):
        """Normalize supported image inputs to unprefixed, padded base64 JPEG."""
        try:
            if isinstance(image, Image.Image):
                return self._encode_pil(image)
            if isinstance(image, bytes):
                return self._encode_pil(Image.open(io.BytesIO(image)))
            if isinstance(image, str):
                if os.path.exists(image):
                    return self._encode_pil(Image.open(image))
                payload = image.split(",", 1)[1] if image.startswith("data:image/") else image
                payload = "".join(payload.split())
                payload += "=" * (-len(payload) % 4)
                decoded = base64.b64decode(payload, validate=True)
                return self._encode_pil(Image.open(io.BytesIO(decoded)))
            return self._encode_pil(load_image(image))
        except Exception as exc:
            raise ValueError(f"Unsupported or invalid image input ({type(image).__name__}): {exc}") from exc

    @staticmethod
    def _text_message(role, text):
        return {
            "role": role,
            "content": [{"type": "text", "text": text}],
        }

    def generate_conversation_fn(self, text, images, role="user", **kwargs):
        assert self.system_prompt, "System prompt must be set before generating conversation."
        user_content = [{"type": "text", "text": "Question: " + text}]
        for image in images or []:
            image_data = self._process_image(image)
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                }
            )

        messages = [self._text_message("system", self.system_prompt)]
        few_shot = kwargs.get("few_shot")
        if few_shot:
            if not isinstance(few_shot, str):
                raise ValueError("few_shot should be a string.")
            messages.append(self._text_message("user", few_shot))
        messages.append({"role": role, "content": user_content})
        return self.check_limit_mm_per_prompt(messages)

    def append_conversation_fn(self, conversation, text, image, role, **kwargs):
        del kwargs
        content = [{"type": "text", "text": text}]
        if image:
            image_data = self._process_image(image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                }
            )
        conversation.append({"role": role, "content": content})
        return self.check_limit_mm_per_prompt(conversation)

    def check_limit_mm_per_prompt(self, conversation: List[dict]):
        image_locations = []
        for message_index, message in enumerate(conversation):
            if message.get("role") == "system":
                continue
            for content_index, content in enumerate(message.get("content", [])):
                if content.get("type") == "image_url":
                    image_locations.append((message_index, content_index))

        if len(image_locations) <= self.limit_mm_per_prompt:
            return conversation

        # Match the native backend: retain the first image and the newest images.
        keep = {image_locations[0]}
        if self.limit_mm_per_prompt > 1:
            keep.update(image_locations[-(self.limit_mm_per_prompt - 1) :])
        for message_index, message in enumerate(conversation):
            message["content"] = [
                content
                for content_index, content in enumerate(message.get("content", []))
                if content.get("type") != "image_url"
                or (message_index, content_index) in keep
            ]
        return conversation

    def form_input_from_dynamic_batch(self, batch: List[DynamicBatchItem]):
        return [item.conversation for item in batch] if batch else []

    def _generate_one(self, conversation):
        max_new_tokens = _as_int(self.generation_config.get("max_new_tokens", 2048), 2048)
        temperature = _as_float(
            self.generation_config.get("temperature", self.temperature), self.temperature
        )
        top_p = _as_float(self.generation_config.get("top_p", 1.0), 1.0)
        last_error = None
        request_started = time.perf_counter()

        for attempt in range(self.max_retry):
            try:
                response = self._next_client().chat.completions.create(
                    model=self.model_name,
                    messages=conversation,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=self.seed,
                    timeout=self.request_timeout,
                )
                response_text = response.choices[0].message.content
                if not response_text:
                    raise RuntimeError("empty response content")
                metrics = {
                    "request_e2e_s": round(time.perf_counter() - request_started, 6),
                    "retry_count": attempt,
                    "backend": "openai_server",
                }
                usage = getattr(response, "usage", None)
                if usage is not None:
                    for source, target in (
                        ("prompt_tokens", "prompt_tokens"),
                        ("completion_tokens", "completion_tokens"),
                        ("total_tokens", "total_tokens"),
                    ):
                        value = getattr(usage, source, None)
                        if value is not None:
                            metrics[target] = int(value)
                return response_text, metrics
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retry:
                    delay = self.retry_base_seconds * min(2**attempt, 8)
                    logger.warning(
                        f"OpenAI-compatible request failed ({attempt + 1}/{self.max_retry}): "
                        f"{exc}; retrying in {delay:.1f}s"
                    )
                    if delay:
                        time.sleep(delay)

        raise RuntimeError(
            f"OpenAI-compatible request failed after {self.max_retry} attempts: {last_error}"
        )

    @staticmethod
    def generation_failure_result(exc):
        failure = (
            "<think> </think><response> Vllm failed to generate response due to "
            f"OpenAI-compatible request error: {exc}. </response>"
        )
        return (
            failure,
            {
                "request_e2e_s": None,
                "backend": "openai_server",
                "error": str(exc),
            },
        )

    def apply_generation_result(self, item, result):
        output_text, metrics = result
        item._backend_generation_metrics = metrics
        item.model_response.append(output_text)
        item.conversation = self.append_conversation_fn(
            item.conversation, output_text, None, "assistant"
        )

    def generate(self, batch):
        """Submit one request per active sample so vLLM can continuously batch them."""
        if not batch:
            return

        conversations = self.form_input_from_dynamic_batch(batch)
        worker_count = min(self.request_concurrency, len(conversations))
        outputs = [None] * len(conversations)

        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="vllm-request") as pool:
            futures = {
                pool.submit(self._generate_one, conversation): index
                for index, conversation in enumerate(conversations)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    outputs[index] = future.result()
                except Exception as exc:
                    logger.error(f"Error during OpenAI-compatible inference: {exc}")
                    outputs[index] = self.generation_failure_result(exc)

        for item, result in zip(batch, outputs):
            self.apply_generation_result(item, result)
