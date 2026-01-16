# Models

**AdaReasoner-7B** is a vision-language model trained with dynamic tool orchestration capabilities for iterative visual reasoning. d.


We provide three variants of AdaReasoner-7B, each optimized for different use cases:

| Model | Description | Hugging Face |
|------|-------------|--------------|
| **AdaReasoner-7B-Randomized** | Trained with the *adaptive learning* method, enabling strong generalization to **unseen tools and tasks**. Designed for open-ended and evolving tool environments where adaptability is required. | [🤗 Link](https://huggingface.co/AdaReasoner/AdaReasoner-7B-Randomized) |
| **AdaReasoner-7B-Non-Randomized** | Trained **without adaptive learning**, providing **more stable and reliable performance on known tools and tasks**, but limited generalization to unseen tools or task settings. | [🤗 Link](https://huggingface.co/AdaReasoner/AdaReasoner-7B-Non-Randomized) |
| **AdaReasoner-VSP-7B** | Task-specialized model trained **exclusively on the Visual Spatial Planning (VSP) task**, achieving strong performance on VSP benchmarks but not intended for cross-task generalization. | [🤗 Link](https://huggingface.co/AdaReasoner/AdaReasoner-VSP-7B) |

# Datasets