# AdaReasoner 统一评测入口

## 实验组

- `no_tools/`：Qwen2.5-VL 3B、7B、32B、72B，不调用工具。
- `with_tools/`：AdaReasoner-7B-Randomized 跑一次，Qwen2.5-VL 7B+Tools 和 72B+Tools 各跑三个 seed。
- 每组默认运行八个主表 benchmark：VSP、VSPO、Jigsaw-COCO、BLINK-J、V*、GUIChat、WebMMU Functional、HRBench。
- Qwen 默认 seed 为 `42,1234,2026`，AdaReasoner 默认 seed 为 `42`。这些是固定 checkpoint 的受控推理 seed，不是独立训练 seed。当前只有 temperature > 0 的任务会产生采样方差；temperature=0 的任务主要得到重复测时时延方差，准确率方差理论上应接近 0。

## 运行

先预检：

```bash
bash /data/songmingyang/code/reasoning/AdaReasoner-rebuttal/.agent/ref/scripts/exps/validate.sh
```

分别运行：

```bash
bash /data/songmingyang/code/reasoning/AdaReasoner-rebuttal/.agent/ref/scripts/exps/no_tools/run_qwen25vl_3b_3seeds.sh
bash /data/songmingyang/code/reasoning/AdaReasoner-rebuttal/.agent/ref/scripts/exps/no_tools/run_qwen25vl_7b_3seeds.sh
bash /data/songmingyang/code/reasoning/AdaReasoner-rebuttal/.agent/ref/scripts/exps/no_tools/run_qwen25vl_32b_3seeds.sh
bash /data/songmingyang/code/reasoning/AdaReasoner-rebuttal/.agent/ref/scripts/exps/no_tools/run_qwen25vl_72b_3seeds.sh
bash /data/songmingyang/code/reasoning/AdaReasoner-rebuttal/.agent/ref/scripts/exps/with_tools/run_qwen25vl_7b_tools_3seeds.sh
bash /data/songmingyang/code/reasoning/AdaReasoner-rebuttal/.agent/ref/scripts/exps/with_tools/run_qwen25vl_72b_tools_3seeds.sh
```

旧 Qwen 全矩阵入口（不包含 AdaReasoner）：

```bash
bash /data/songmingyang/code/reasoning/AdaReasoner-rebuttal/.agent/ref/scripts/exps/run_all.sh
```

正式带工具实验分两步运行。先启动并检查工具：

```bash
bash /data/songmingyang/code/reasoning/AdaReasoner-rebuttal/.agent/ref/scripts/start_tools.sh
bash /data/songmingyang/code/reasoning/AdaReasoner-rebuttal/.agent/ref/scripts/exps/with_tools/check_tools.sh
```

再顺序运行 AdaReasoner 一次、Qwen 7B 三 seed、Qwen 72B 三 seed：

```bash
AUTO_START_TOOLS=0 bash /data/songmingyang/code/reasoning/AdaReasoner-rebuttal/.agent/ref/scripts/exps/with_tools/run_all_7b_72b.sh
```

每个模型都占用 GPU `0,3` 并使用 TP=2，因此三个模型必须串行；GPU `1,2` 始终保留给 Point/OCR 工具 worker。默认 `EVAL_BACKEND=native`，继续使用仓库原生 `VllmModels -> TFEvaluator -> ToolManager` 链路。

也可以选择共享常驻 vLLM 服务模式。先启动当前模型的服务，再运行同一个评测入口：

```bash
source /data/songmingyang/code/reasoning/AdaReasoner-rebuttal/.agent/ref/scripts/exps/with_tools/vllm_serve.sh
vllm_start /data/songmingyang/model/adareasoner/AdaReasoner-7B-Randomized 0,3 2 8000
vllm_wait 8000 600

EVAL_BACKEND=server VLLM_PORT=8000 VLLM_CONCURRENCY=32 TOOL_CONCURRENCY=16 \
  bash /data/songmingyang/code/reasoning/AdaReasoner-rebuttal/.agent/ref/scripts/exps/with_tools/run_one.sh \
  7B with_tools vsp 42
```

`server` 模式使用相同的 task loader、tool parser、checkpoint、结果目录、validator 和 `DONE.json` 协议；区别仅是模型请求并发提交到共享 OpenAI-compatible vLLM server，工具请求由有界线程池并发执行。服务必须与当前评测 checkpoint 对应；切换模型时先执行 `vllm_stop 8000`，再用新模型重新启动服务。完整多模型矩阵目前仍建议按模型启动服务并逐模型运行，避免误用错误 checkpoint。

只跑指定任务或修改 seed：

```bash
TASKS=vsp,jigsaw_coco SEEDS=42,1234,2026 \
  bash /data/songmingyang/code/reasoning/AdaReasoner-rebuttal/.agent/ref/scripts/exps/with_tools/run_qwen25vl_7b_tools_3seeds.sh
```

指定 GPU：

```bash
GPU_SINGLE=2 GPU_72B=0,3 \
  bash /data/songmingyang/code/reasoning/AdaReasoner-rebuttal/.agent/ref/scripts/exps/run_all.sh
```

带工具的 7B/72B 默认都使用模型 GPU `0,3`（TP=2），为已启动的 Point/OCR worker 保留 GPU `1,2`。3B/32B 默认使用单卡；32B 若单卡显存不足，可直接覆盖为双卡：

```bash
GPU_32B=0,3 TP_32B=2 \
  bash /data/songmingyang/code/reasoning/AdaReasoner-rebuttal/.agent/ref/scripts/exps/no_tools/run_qwen25vl_32b_3seeds.sh
```

同样支持 `GPU_3B`、`GPU_7B`、`GPU_72B` 和对应 `TP_*` 的逐规模覆盖。

## 工具服务

正式总入口会严格检查 `http://127.0.0.1:21112` 和六个服务端口，默认要求工具已提前通过：

```text
/data/songmingyang/code/reasoning/AdaReasoner-rebuttal/.agent/ref/scripts/start_tools.sh
```

临时需要入口代为启动时可设置 `AUTO_START_TOOLS=1`；正式实验建议保持默认 `0`，将工具启动日志和评测日志分开。注意 `.agent/ref/scripts/exps/run_all.sh` 是全实验入口，不是工具启动器。

## 结果与时延

默认输出：

```text
/data/songmingyang/code/reasoning/AdaReasoner-rebuttal/rebuttal_exps/qwen25vl_eval/
├── no_tools/qwen25vl_{3b,7b,32b,72b}/
└── with_tools/
    ├── adareasoner_randomized_7b/
    ├── qwen25vl_{7b,72b}/
    └── accuracy_latency/
        ├── accuracy_latency.csv
        ├── accuracy_latency.json
        └── accuracy_latency.svg
```

每次运行保存：

- `result.jsonl`：任务评分结果；
- `timing.json`：模型加载、模型 generation、纯评测 wall time、进程总时间；
- `stage_latency.json`：with-tools 的 tool/orchestration 分阶段时延；
- `ckpt.jsonl` / `latency.jsonl` / `latency_summary.json`：逐 instance 原始 latency、扁平记录和统计摘要；`instance_e2e_s` 从入队前到最后一轮被识别为完成，含排队、模型、工具和编排，不含最终序列化/checkpoint 写盘；
- `DONE.json`：样本数、任务、指标与元数据全部校验通过后的原子完成标志；
- `config.yaml`、`run_metadata.json`、`run.log`：协议和运行环境快照。

每个模型组自动生成 `summary.json` 和 `summary.csv`。正式三模型矩阵全部完成后还会生成：

- `accuracy_latency.csv`：逐 task 和 macro 的准确率、instance E2E latency、seed 标准差；
- `accuracy_latency.json`：相同数据及统计口径；
- `accuracy_latency.svg`：macro 与八个任务的准确率—时延对比曲线。

其中主指标以百分数报告。Qwen 的 `variance/std` 来自三个推理 seed；AdaReasoner 只跑一次，因此没有可解释的跨 seed 方差。GUIChat 的指标是阈值化 ANLS，其余任务按各自官方评分；macro 是八个 benchmark 主指标的非加权均值。该图连接的是同协议下三个模型配置，不等同于改变最大轮数得到的 matched-budget sweep。

## 注意事项

1. `VSP` 在这里强制恢复 `verify_test + navigation_test` 全量，不使用仓库当前 `num_sample: 4` 的 smoke 配置。
2. 生成长度统一通过实际生效的 `max_new_tokens` 配置。
3. `native` 与 `server` 模式共享 `ckpt.jsonl` 数据结构和结果协议，但正式复现应保持同一 job 的 backend、并发参数和 served checkpoint 不变。若要切换后端重跑，设置 `FORCE=1`，旧目录会先归档而非覆盖。
4. HRBench 当前评分器对不能直接解析的答案依赖 OpenAI API；没有 `OPENAI_API_KEY` 时仍保存推理输出，但只要产生 `Z`，脚本就拒绝写入正式 `DONE.json`，需要后续使用本地 72B judge 统一重判。
5. GUIChat 当前脚本使用仓库原生阈值化 ANLS 评分，不依赖 GPT judge；论文若采用另一套 72B judge 口径，必须另行离线统一重判并单独命名。
6. `with_tools` 的 wall time 会受在线工具排队影响。正式比较时不要并行跑多组模型，并保持相同工具 worker、GPU 和并发设置。
