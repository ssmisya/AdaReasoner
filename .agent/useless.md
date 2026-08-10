# /data/songmingyang 无用数据分析报告

> 分析时间：2026-08-07
> 磁盘使用：33T / 35T（93%），剩余约 2.6T

---

## 一、确定可安全删除（约 375G）

### 1. 评测中断/失败备份目录 .bak_* — 约 78G

路径：`/data/songmingyang/code/reasoning/AdaReasoner-rebuttal/rebuttal_exps/qwen25vl_eval/`

共 **82 个** bak_* 目录，来自之前中断/失败的评测运行。主要有两类：

| 目录 | 大小 | 说明 |
|------|------|------|
| `with_tools/**/seed_*.bak_*` | ~59G | 带工具评测的 resume/protocol_mismatch 备份 |
| `no_tools/**/seed_*.bak_*` | ~19G | 无工具评测的备份 |

**删除命令参考：**
```bash
find /data/songmingyang/code/reasoning/AdaReasoner-rebuttal/rebuttal_exps/ -name "*.bak_*" -type d -exec rm -rf {} +
```

[!] 注意：如果当前评测还在运行中，建议等运行完再清理，避免误删正在使用的 checkpoint 备份。

---

### 2. 数据策展中间版本 — 约 272G

路径：`/data/songmingyang/code/safe/vllm_guard/data_curation/outputs/`

| 子目录 | 大小 | 说明 |
|--------|------|------|
| `v2.6/` | 66G | 中间版本 |
| `v2.8_withreason/` | 45G | 中间版本 |
| `v2.10_merged/` | 35G | 已合并，可删旧版本 |
| `v2.7_withreason/` | 34G | 中间版本 |
| `v2.5_withreason/` | 31G | 中间版本 |
| `v2.9_withreason_randompolicy/` | 17G | 中间版本 |
| `v2.9_withreason/` | 17G | 中间版本 |
| `v2.11/` | 12G | 较新版本 |
| `v2.3/` | 5.9G | 旧版本 |
| `v2.10_rl/` | 3.9G | RL 中间数据 |
| 其他 (v2.2, v2.4, v2.5 等) | ~5G | 旧版本 |

**建议：** 确认最终使用的版本后，只保留最终版本 + 最终合并版（如 v2.11 + v2.10_merged），删除其余所有中间版本。

---

### 3. 根目录临时日志/文件 — 约 2.7G

路径：`/data/songmingyang/`

| 文件 | 大小 | 说明 |
|------|------|------|
| `ada_deps_install.log` | 1.6G | conda 依赖安装日志 |
| `ada_hf_download.py` | 399M | HuggingFace 下载脚本 |
| `ada_paddle_install.log` | 195M | Paddle 安装日志 |
| `ada_vsp_download.log` | 142M | 下载日志 |
| `ada_shard_download.log` | 141M | 下载日志 |
| `ada_model_download.log` | 117M | 模型下载日志 |
| `ada_hf_download.log` | 75M | HF 下载日志 |
| `ada_editable_install.log` | 28M | 可编辑安装日志 |
| `vllm_install.log` | 14M | vLLM 安装日志 |
| `flash_attn_build.log` | 13M | flash attention 编译日志 |
| `flash_attn_install.log` | 10M | flash attention 安装日志 |
| `vllm_upgrade.log` | 2.1M | vLLM 升级日志 |
| `ada_hf_download.pid` | ~1K | PID 文件 |
| `vllm_install.pid` | ~1K | PID 文件 |
| `ada_hf_download.stdout` | ~1K | stdout 输出 |
| `provenance_audit_conclusion.md` | ~1K | 审计结论 |

全部可安全删除（脚本已执行完毕，日志无保留价值）。

**删除命令：**
```bash
rm -f /data/songmingyang/ada_*.log /data/songmingyang/ada_*.py /data/songmingyang/ada_*.pid /data/songmingyang/ada_*.stdout
rm -f /data/songmingyang/vllm_install.log /data/songmingyang/vllm_install.pid /data/songmingyang/vllm_upgrade.log
rm -f /data/songmingyang/flash_attn_build.log /data/songmingyang/flash_attn_install.log
rm -f /data/songmingyang/provenance_audit_conclusion.md
```

---

### 4. 评测日志 — 约 1G

| 路径 | 大小 |
|------|------|
| `rebuttal_exps/vllm_server_logs/` | 815M |
| `rebuttal_exps/toolserver_logs/` | 221M |

运行中的日志建议保留，历史日志可删除。

---

### 5. 历史运行状态目录 — 约 243M

路径：`/data/songmingyang/code/reasoning/AdaReasoner-rebuttal/rebuttal_exps/qwen25vl_eval/.with_tools_state/`

共 9 个状态目录，仅保留当前运行中的 `20260807_150537`，其余 8 个历史目录均可删除。

---

### 6. transfer_logs — 约 5.5M

`/data/songmingyang/transfer_logs/` 中的 `.complete` 标记文件和 `run_all_7b_72b.log` 是已完成任务的传输标记，没有保留价值。

---

### 7. wandb 运行日志 — 约 160M

路径：`/data/songmingyang/code/safe/vllm_guard/wandb/`

共 10+ 个 run 日志目录，每次 10-32M。训练已完成，可安全删除。

---

### 8. tmp 临时文件 — 约 39M

路径：`/data/songmingyang/tmp/`

---

### 9. uv pip 缓存 — 约 2.6G

路径：`/data/songmingyang/caches/uv/archive-v0/`

---

## 二、谨慎评估后可删除（约 673G+）

### 10. ablation_sft 消融实验模型 — 约 345G

路径：`/data/songmingyang/models/vllm_guard/safe/ablation_sft/`

共 **11 个**消融实验训练出的模型版本（3B 和 7B），命名如：
- `qwen25vl_3b_boundary_pair_sft_v211_...`
- `qwen25vl_7b_paircontrast_v211_...`

**建议：**
- 确认 railguard 论文/实验最终使用的是哪个模型
- 只保留最终选用版本，删除其余消融实验模型
- 如果模型已推送到 HuggingFace 或模型数据库，可全部删除

---

### 11. RL 训练中间 checkpoints — 约 275G

路径：`/data/songmingyang/checkpoints/vllm_guard/railguard_rl/`

| 目录 | 大小 | 说明 |
|------|------|------|
| `qwen25vl-7b-grpo-v210-prm-gpt54mini-r40/global_step_500/` | 124G | 7B GRPO，仅1步 |
| `qwen25vl-7b-grpo-v210-prm-gpt54mini-r40-v2/global_step_250/` | 93G | 7B GRPO v2，仅1步 |
| `qwen25vl-3b-grpo-v210-prm-gpt54mini-r40-20260423/global_step_850/` | 59G | 3B GRPO，仅1步 |

**建议：** 目前每个实验只保留了1个最终的 global_step，没有中间步骤。如果这些模型已完成评测且推送到 HF/model zoo，可全部删除。否则保留。

---

### 12. SFT 训练 checkpoint — 约 28G

路径：`/data/songmingyang/checkpoints/vllm_guard/qwen2.5-vl-3b-sft-v2.7-think-1ep/`

- `checkpoint-32/`: 21G
- 父目录（optimizer 等）: 7G

如果推送到 HF 后，可删除。

---

### 13. HuggingFace 模型缓存 — 约 23G

路径：`/data/songmingyang/caches/huggingface/hub/`

可能包含不再使用的模型临时下载缓存。清理前需要确认：
- 哪些模型还在 `model/` 中有 symlink 引用
- 哪些是纯缓存副本

---

### 14. model/adareasoner vs models/adareasoner — 需确认重复

| 路径 | 大小 |
|------|------|
| `model/adareasoner/AdaReasoner-7B-Randomized/` | 102G |
| `models/adareasoner/AdaReasoner-7B-Randomized/` | 6.1G |

同名但大小差异巨大，需确认：
- 102G 版本是否包含额外的训练文件/优化器状态
- 评测配置中使用的是哪个路径
- 如果 6.1G 版本已足够评测使用，可删除 102G 版本省出 ~96G

---

## 三、汇总表

| 类别 | 预估可释放 | 删除风险 |
|------|-----------|----------|
| ✅ 评测 bak_* 目录 | 78G | 无 |
| ✅ 数据策展中间版本 | 272G | 无（保留最终版本） |
| ✅ 根目录日志/临时文件 | 2.7G | 无 |
| ✅ 评测日志 | 1G | 无 |
| ✅ 历史状态目录 | 0.2G | 无 |
| ✅ wandb 日志 | 0.2G | 无 |
| ✅ tmp + transfer_logs | 0.05G | 无 |
| ✅ uv pip 缓存 | 2.6G | 低 |
| ⚠️ ablation SFT 模型 | 345G | 中（需确认最终选用版本） |
| ⚠️ RL checkpoints | 275G | 中（需确认是否已备份） |
| ⚠️ SFT checkpoint | 28G | 低 |
| ⚠️ HF 缓存 | 23G | 低（需确认依赖） |
| ⚠️ model/adareasoner 重复 | ~96G | 低（确认路径后删大版本） |
| **合计** | **~1.12T** | |

---

## 四、推荐的删除优先级

**第一优先级（立即可执行，无风险）：**
1. 评测 bak_* 目录：78G
2. 数据策展中间版本（保留 v2.11 + v2.10_merged）：~260G
3. 根目录日志/临时文件：2.7G
4. 评测日志 + 历史状态：~1.2G
5. wandb + tmp + transfer_logs：~0.2G
6. uv 缓存：2.6G

→ 可立即释放约 **345G**

**第二优先级（确认后执行）：**
1. ablation SFT 模型（留最终版）：~300G
2. RL checkpoints（确认模型已备份）：~275G
3. SFT checkpoint：28G
4. HF 缓存：23G
5. model/adareasoner 去重：~96G

→ 可额外释放约 **722G**

**总计最多可释放：约 1.07T**
