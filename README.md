# R1-V: Reinforcing Super Generalization Ability in Vision Langauge Models with Less Than $3



1. We firstly reveal that **Reinforcement Learning with Verifiable Rewards (RLVR)** outperforms chain-of-thought supervised fine-tuning (CoT-SFT) in both **effectiveness and out-of-distribution (OOD) robustness** for vision language models.

2. In our experiment, we **incentivize** VLMs to learn **generalizable** visual counting abilities, rather than overfitting to the training set.

3. The 2B model outperforms the 72B model in OOD tests within just **100** training steps.

4. The training was conducted on 8 A100 GPUs for **30 minutes, costing $2.62**.

5. Codes, models, datasets, more details and **all open-source** resources will be shared (within CNY holidays).

**Contributors:** Liang Chen · Lei Li · Haozhe Zhao · Yifan Song

[🤗 Train Datasets](https://huggingface.co/datasets/leonardPKU/clevr_cogen_a_train)

---





![image](./images/ood.png)

![image](./images/super_ood.png)

![image](./images/training.png)

## Acknowledgements

We sincerely thank DeepSeek, [Open-R1](https://github.com/huggingface/open-r1), [Open-R1-Multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal), [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/), [SuperCLEVR](https://github.com/Lizw14/Super-CLEVR) for providing open source resources for us to build the project.

## Citation

```bib
@misc{chen2025r1v,
  author       = {Chen, Liang and Li, Lei and Zhao, Haozhe and Song, Yifan},
  title        = {R1-V: Reinforcing Super Generalization Ability in Vision-Language Models with Less Than \$3},
  howpublished = {\url{https://github.com/Deep-Agent/R1-V}},
  note         = {Accessed: 2025-02-02},
  year         = {2025}
}
```


