# Models

**AdaReasoner-7B** is a vision-language model trained with dynamic tool orchestration capabilities for iterative visual reasoning. d.


We provide three variants of AdaReasoner-7B, each optimized for different use cases:

| Model | Description | Hugging Face |
|------|-------------|--------------|
| **AdaReasoner-7B-Randomized** | Trained with the *adaptive learning* method, enabling strong generalization to **unseen tools and tasks**. Designed for open-ended and evolving tool environments where adaptability is required. | [🤗 Link](https://huggingface.co/AdaReasoner/AdaReasoner-7B-Randomized) |
| **AdaReasoner-7B-Non-Randomized** | Trained **without adaptive learning**, providing **more stable and reliable performance on known tools and tasks**, but limited generalization to unseen tools or task settings. | [🤗 Link](https://huggingface.co/AdaReasoner/AdaReasoner-7B-Non-Randomized) |
| **AdaReasoner-VSP-7B** | Task-specialized model trained **exclusively on the Visual Spatial Planning (VSP) task**, achieving strong performance on VSP benchmarks but not intended for cross-task generalization. | [🤗 Link](https://huggingface.co/AdaReasoner/AdaReasoner-VSP-7B) |

# Datasets

We provide the following datasets used for training and evaluating AdaReasoner-7B.
### Training Datasets 

| Dataset | Description | Hugging Face |
|-------|-------------|--------------|
| **AdaReasoner-TC** | Core **Tool-Calling (TC)** training dataset covering multiple visual reasoning tasks. Provides structured multi-turn tool-use trajectories for supervised and reinforcement learning of tool planning. | [🤗 Link](https://huggingface.co/datasets/hitsmy/AdaReasoner-TC) |
| **AdaReasoner-TC-Randomized** | Tool-calling dataset with **randomized tool definitions and invocation formats**, designed to support *adaptive learning* and improve generalization to unseen tools and task configurations. | [🤗 Link](https://huggingface.co/datasets/hitsmy/AdaReasoner-TC-Randomized) |
| **AdaReasoner-TC-VSP** | Task-specific TC dataset curated **exclusively for Visual Spatial Planning (VSP)**, focusing on navigation, verification, and spatial reasoning tool usage. | [🤗 Link](https://huggingface.co/datasets/hitsmy/AdaReasoner-TC-VSP) |
| **AdaReasoner-TC-VSP-Reflection** | An enhanced VSP TC dataset augmented with **reflection and self-correction trajectories**, encouraging models to evaluate intermediate tool outputs and revise planning decisions. | [🤗 Link](https://huggingface.co/datasets/hitsmy/AdaReasoner-TC-VSP-Reflection) |
| **AdaReasoner-TC-Jigsaw** | Tool-calling dataset tailored for the **Jigsaw visual reasoning task**, emphasizing trial, comparison, and correction behaviors during tool-based puzzle solving. | [🤗 Link](https://huggingface.co/datasets/hitsmy/AdaReasoner-TC-Jigsaw) |
| **AdaReasoner-TC-WebQA** | Tool-calling dataset designed for **WebQA-style visual question answering**, featuring coordinated use of perception tools such as cropping and OCR. | [🤗 Link](https://huggingface.co/datasets/hitsmy/AdaReasoner-TC-WebQA) |
| **AdaReasoner-TG-Data** | **Tool-GRPO (TG)** reinforcement learning dataset containing online interaction trajectories collected via the Tool Server, supporting long-horizon tool planning optimization. | [🤗 Link](https://huggingface.co/datasets/hitsmy/AdaReasoner-TG-Data) |
| **AdaReasoner-TG-Data-Randomized** | TG dataset with **randomized tool pools and invocation schemas**, specifically constructed to train tool-planning models with strong robustness and cross-tool generalization. | [🤗 Link](https://huggingface.co/datasets/hitsmy/AdaReasoner-TG-Data-Randomized) |


### Evaluation Datasets 

| Dataset | Hugging Face |
|-------|--------------|
| **AdaReasoner/AdaEval-VSPO** |  [🤗 Link](https://huggingface.co/datasets/AdaReasoner/AdaEval-VSPO) |
| **AdaReasoner/AdaEval-VSP** |  [🤗 Link](https://huggingface.co/datasets/AdaReasoner/AdaEval-VSP) |
| **AdaReasoner/AdaEval-Jigsaw-COCO** |  [🤗 Link](https://huggingface.co/datasets/AdaReasoner/AdaEval-Jigsaw-COCO) |