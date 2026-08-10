# AdaReasoner 统一评测方法论

本文规定 `AdaReasoner-rebuttal` 中有工具（with-tools）和无工具（no-tools）评测的统一入口、可比性约束、并行调度、断点恢复、结果验证和 latency 口径。新实验应复用这里的脚本，不要另写临时推理命令。

## 1. 评测目标与基本单位

完整实验矩阵由以下维度组成：

- 模型：待评测模型；
- 模式：`no_tools` 或 `with_tools`；
- task：8 个固定任务；
- seed：Qwen 受控重复默认 `42,1234,2026`，最终 AdaReasoner 默认只跑 `42` 一次；
- instance：一个 task 中的一条样本；
- round：模型的一次生成；with-tools 中还可能包含一次工具调用；
- job：一个 `模型 × 模式 × task × seed`，只有全部样本推理、评分、校验完成并写出 `DONE.json` 才算完成。

因此 Qwen 单模型默认有 `8 tasks × 3 seeds = 24 jobs`，AdaReasoner 最终模型有 `8 tasks × 1 seed = 8 jobs`。面板中的 job 计数只表示完整 task×seed，不代表样本没有推进；必须同时查看样本级 checkpoint。

## 2. 固定实验契约

唯一任务清单与数据口径：

```text
.agent/ref/scripts/exps/shared/task_matrix.json
```

| task | 期望样本数 | 工具 |
|---|---:|---|
| `vsp` | 1100 | AStarWithPixelCoordinate, Draw2DPath, Point |
| `vspo` | 1670 | AStarWithPixelCoordinate, Draw2DPath, Point |
| `jigsaw_coco` | 1000 | DetectBlackArea, InsertImage |
| `jigsaw_blink` | 150 | DetectBlackArea, InsertImage |
| `vstar` | 191 | Point, OCR, Crop, AStarWithPixelCoordinate |
| `web_guichat` | 962 | OCR, Point, Crop |
| `webmmu` | 492 | OCR, Crop |
| `hrbench` | 800 | Point, OCR, Crop, AStarWithPixelCoordinate |

正式对比必须固定：

1. 相同 task、数据版本和期望样本数；
2. Qwen 7B/72B 使用相同三个 seed；AdaReasoner 按本轮实验设计只跑 seed 42 一次，因此只能报告单次点估计，不能声称其跨 seed 方差；
3. 相同 generation config；默认按 task 配置解码，temperature 大于 0 的任务会产生采样差异；
4. 相同上下文策略、工具集合、最大轮数与评分器；默认 `max_model_len=8192`，但 task 声明更高下限时自动提升（HRBench 为 32768）；
5. 相同 batch size、TP、模型 GPU、工具拓扑和并发方式，报告模型路径与 latency 口径；
6. 不得把缺少 `DONE.json` 的半成品计入正式结果；
7. AdaReasoner 与 Qwen 必须使用独立 `MODEL_SLUG`，禁止共用结果目录；
8. `no_tools` 和 `with_tools` 必须分目录汇总，不得用 wall time 代替逐 instance latency。

## 3. 统一调用链

```text
模型/模式入口
  -> shared/run_matrix.sh            # 遍历 task × seed
  -> shared/run_one.sh               # 锁、配置、恢复、执行、latency、校验
  -> shared/eval_entry.py
  -> tool_server.tf_eval.TFEvaluator
  -> BaseInferencer                   # no-tools
     或 BaseToolInferencer            # with-tools
  -> BaseEvalDataset.store_results    # 逐 instance 写 ckpt.jsonl
  -> summarize_latency.py             # latency.jsonl / latency_summary.json
  -> validate_run.py                  # 样本数、指标、模型、seed 校验
  -> DONE.json                        # 正式完成标志
  -> summarize.py                     # summary.json / summary.csv
```

`run_one.sh` 使用每 job 原子锁，避免两个进程同时覆盖同一个结果目录。默认 `RESUME=1`，已有 `ckpt.jsonl` 时只处理剩余 instance。

## 4. No-tools 评测

### 4.1 语义

- `script_args.if_use_tool=false`；
- 使用 `BaseInferencer`；
- 每个 instance 强制 `max_rounds=1`；
- 不访问 Controller，不调用 Point/OCR/Crop；
- 单轮仍可能生成较长回答，因此“单轮”不等于“单 token”。

### 4.2 四卡动态入口

```bash
bash /data/songmingyang/code/reasoning/AdaReasoner-rebuttal/.agent/ref/scripts/exps/no_tools/run_all_parallel.sh
```

重新进入：

```bash
tmux attach -t ada_eval_no_tools
```

布局原则：

- 72B 使用两卡 TP=2；
- 32B、7B、3B 使用单卡动态队列；
- 任一队列先结束后，空闲 GPU 自动加入剩余队列；
- worker 通过 job lock 抢任务，不会重复计算；
- dashboard 同时展示完整 job、当前 instance checkpoint、GPU 利用率和失败重试。

单模型调试入口位于：

```text
.agent/ref/scripts/exps/no_tools/run_qwen25vl_{3b,7b,32b,72b}_3seeds.sh
```

## 5. With-tools 评测

### 5.1 工具服务拓扑

固定布局如下：

- GPU 0、3：当前评测模型，`CUDA_VISIBLE_DEVICES=0,3`，TP=2；
- GPU 1：Point + OCR；
- GPU 2：Point + OCR；
- CPU：Crop；
- Controller：`http://127.0.0.1:21112`。

端口：Controller 21112，Point 50002/50003，OCR 50010/50011，Crop 50012。两个 Point/OCR endpoint 提供容量与冗余，实际 worker 选择由 Controller/ToolManager 决定；模型评测本身不会同时跑两个模型，因为 GPU 0、3 已被一个 TP=2 模型占满。

正式运行坚持“服务与评测分离”两步法。

第一步，启动并检查工具：

```bash
bash /data/songmingyang/code/reasoning/AdaReasoner-rebuttal/.agent/ref/scripts/start_tools.sh
bash /data/songmingyang/code/reasoning/AdaReasoner-rebuttal/.agent/ref/scripts/exps/with_tools/check_tools.sh
```

第二步，启动评测：

```bash
AUTO_START_TOOLS=0 bash /data/songmingyang/code/reasoning/AdaReasoner-rebuttal/.agent/ref/scripts/exps/with_tools/run_all_7b_72b.sh
```

注意：`.agent/ref/scripts/exps/run_all.sh` 是“no-tools + with-tools”的旧全实验串行入口，不是工具服务器启动器，不能代替 `start_tools.sh`。正式总入口默认 `AUTO_START_TOOLS=0`；临时环境需要自动拉起服务时才显式设为 `1`。

### 5.2 三模型正式矩阵

严格按以下顺序串行评测：

1. `AdaReasoner-7B-Randomized`：seed 42，一次；
2. `Qwen2.5-VL-7B-Instruct + Tools`：seed 42、1234、2026；
3. `Qwen2.5-VL-72B-Instruct + Tools`：seed 42、1234、2026。

这里需求中的 `Adareasoner0random-7b` 对应仓库实际模型 `/data/songmingyang/model/adareasoner/AdaReasoner-7B-Randomized`，结果 slug 为 `adareasoner_randomized_7b`；另外两个默认模型位于 `/data/songmingyang/models/baselines/Qwen2.5-VL-{7B,72B}-Instruct`。

所有模型都使用 GPU 0、3 和 TP=2，保证占满除工具卡之外的两张模型卡。三个模型不能并发，否则会争用同一对 GPU；模型侧并行来自 TP=2 和 vLLM 动态 batch，工具服务独立驻留在 GPU 1、2，而不是再并发启动多个评测模型。

正式入口会进行静态依赖、模型权重、GPU0/3 空闲显存、工具注册与六个端口检查，随后创建 tmux 四格界面：

1. 三模型 TP2 实时日志；
2. GPU、完整 job、当前样本和 latency 总览；
3. 工具健康与四卡利用率；
4. Controller、Point、OCR、Crop 实时日志。

重新进入：

```bash
tmux attach -t ada_eval_with_tools
```

单模型入口：

```text
with_tools/run_adareasoner_randomized_7b_once.sh
with_tools/run_adareasoner_randomized_7b_tools_3seeds.sh  # 仅额外消融时使用
with_tools/run_qwen25vl_7b_tools_3seeds.sh
with_tools/run_qwen25vl_72b_tools_3seeds.sh
```

### 5.3 工具调用语义

- `script_args.if_use_tool=true`；
- 最大 6 轮；
- 每轮最多解析并执行一个工具调用；
- 出现 `<response>...</response>` 时提前结束；
- 每个 task 只能看到 `task_matrix.json` 声明的工具；
- 默认 `IF_RANDOMIZE_TOOL=false`，即三个模型使用相同真实工具名，保证横向可比；
- 若研究工具名随机化，必须显式设置 `IF_RANDOMIZE_TOOL=true`，并作为独立实验报告，不得和默认结果混算。

## 6. Latency：每次评测必须记录

新评测默认 `REQUIRE_LATENCY=1`。缺少任一 instance latency 时，job 不生成正式 `DONE.json`。

### 6.1 记录口径

使用 `time.perf_counter_ns()` 单调高精度时钟：

- `instance_e2e_s`：instance 加入动态 batch、构建首轮对话之前，到最后一轮被识别为完成的客户端观测端到端时间；包含排队、模型生成、工具调用和主要框架编排，不包含最终结果序列化与 checkpoint 写盘；
- `round_e2e_s`：本轮模型生成开始，到本轮结束；若调用工具，则结束点是工具结果转换为下一轮输入之后；
- `generation_batch_wall_s`：一次同步模型 batch 的 wall time。该值由同一批请求共享，不能将它在 instance 间相加后解释为模型总计算时间；
- `backend_request_metrics.request_e2e_s`：vLLM 版本提供指标时记录的单请求后端 E2E；
- `backend_request_metrics.ttft_s`：vLLM 单请求首 token 时间；
- `backend_request_metrics.queue_s`：vLLM 内部排队时间；
- `tool_calls[].latency_s`：一次 `ToolManager.call_tool` RPC 的 wall time；
- `tool_latency_s`：该 instance 或 round 内工具 RPC latency 之和。

No-tools 的 `round_count` 固定为 1；with-tools 会保存每一轮的独立记录。

### 6.2 输出位置

每个 job 目录包含：

```text
ckpt.jsonl             # 逐 instance 保存；results.results.latency 为原始结构化 latency
latency.jsonl          # 提取后的逐 instance latency，便于分析
latency_summary.json   # mean/median/p90/p95/p99/min/max/std，含按 round 分组
result.jsonl           # task 最终评分结果
DONE.json              # 正式校验通过标志
```

每个模型目录另外生成 `summary.json` 和 `summary.csv`，汇总该模型的 task、seed、accuracy 和 latency。

恢复旧 checkpoint 时，如果旧记录没有 latency，严格模式会报告 coverage 不完整。正式 latency 实验应新建结果目录或归档旧结果，不得用少量新记录代表整个旧实验。

### 6.3 Accuracy–latency 曲线

三模型矩阵全部完成后，worker 自动运行：

```text
with_tools/plot_accuracy_latency.py
```

并在 `RESULT_ROOT/with_tools/accuracy_latency/` 生成：

```text
accuracy_latency.csv   # 逐 task 与 macro 的原始作图数据
accuracy_latency.json  # 统计定义、完整性状态和同一批数据
accuracy_latency.svg   # macro + 8 tasks 的九宫格曲线
```

横轴为客户端观测的 mean `instance_e2e_s`，纵轴为各任务官方主指标百分数；macro 对 accuracy 和 task-level latency 都取八任务非加权均值。Qwen 误差统计来自三个受控 seed，AdaReasoner 单次运行只能给点估计。曲线只读取有 `DONE.json` 且 latency coverage 完整的结果，缺任一正式 job 时不生成可提交曲线。

这张图连接的是**同协议下三个模型配置**，不是 max-round/tool-call budget sweep。若论文声称 matched-budget frontier，必须另设最大轮数或工具调用预算并分目录重跑，不能把本图包装成预算扫描。

## 7. 结果目录与完成判定

默认根目录：

```text
rebuttal_exps/qwen25vl_eval/
├── no_tools/<model_slug>/<task>/seed_<seed>/
└── with_tools/
    ├── adareasoner_randomized_7b/<task>/seed_42/
    ├── qwen25vl_7b/<task>/seed_<seed>/
    ├── qwen25vl_72b/<task>/seed_<seed>/
    └── accuracy_latency/
```

三类状态：

- `ckpt.jsonl` 增长：instance 正在完成；
- `result.jsonl` 存在：已评分，但不一定通过正式校验；
- `DONE.json` 且 `validated=true`：正式完成，可进入汇总。

不要仅根据进程退出码或 `result.jsonl` 判断成功。

## 8. 常用覆盖参数

```bash
TASKS=vsp,vspo
SEEDS=42,1234,2026
ADAREASONER_SEEDS=42
BATCH_ALL=64
MAX_ATTEMPTS=2
RESUME=1
REQUIRE_LATENCY=1
GENERATE_CURVE=1
AUTO_START_TOOLS=0
ADAREASONER_MODEL_PATH=/path/to/AdaReasoner-7B-Randomized
QWEN7B_MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct
QWEN72B_MODEL_PATH=/path/to/Qwen2.5-VL-72B-Instruct
```

只做小规模调试时可覆盖 task/seed，但正式结果必须恢复 Ada 一 seed、Qwen 三 seed 的默认矩阵。改变 batch size 会改变吞吐、排队及 latency，做 latency 横向比较时必须保持相同 batch size。模型路径覆盖会固化进本次 tmux 状态和 `run_metadata.json`，换机器时无需改 worker 源码。

## 9. 失败恢复

1. 使用 `Ctrl+B D` 分离 tmux，不要关闭评测进程；
2. 重新运行总入口时，活跃的同名 session 会直接进入，不会创建重复 worker；
3. 若上次 worker 已完成、失败或意外退出，入口会清理旧 tmux 界面并创建新 worker，`run_one.sh` 从 `ckpt.jsonl` 继续；
4. 单 job 默认最多尝试两次；
5. with-tools worker 每个 job 开始前检查工具健康；工具不完整时暂停等待，不会静默退化为 no-tools；
6. 只有所有 job 和 accuracy–latency 产物都完成才写 `ALL_DONE`；失败写 `FINISHED_WITH_FAILURES`；
7. `FORCE=1` 会归档旧运行后重算，只在确认需要重做时使用。

## 10. 常见问题

### 面板长期显示 `0/24`（AdaReasoner 为 `0/8`）

完整 job 尚未结束。查看 dashboard 的 `当前样本/期望样本` 和 `ckpt.jsonl` 更新时间。

### GPU 利用率高但 checkpoint 不增长

检查当前是否处于长输出 batch、模型加载、评分阶段，随后查看 `run.log` 是否有 OOM、EngineCore 或工具超时。

### with-tools 没有真正调用工具

检查：

1. `config.yaml` 中 `if_use_tool: true`；
2. `check_tools.sh` 全部通过；
3. `tool_cfg`、`tool_response` 非空；
4. `latency.jsonl` 的 round 中存在 `tool_calls`；
5. 工具 pane 中对应 worker 日志有请求。

### latency 为什么会随 batch 改变

这里报告客户端观测 latency，包含 vLLM 排队与同步 batch 等待。batch 越大，吞吐通常更高，但单 instance latency 不一定更低。比较 latency 时必须固定并报告 batch、TP、GPU 和并发方式。

### HRBench 校验失败

若评分存在需要外部判定的模糊答案，需提供有效 `OPENAI_API_KEY`；否则 `validate_run.py` 会拒绝将不完整评分写成正式 `DONE.json`。
