# DeepSeek V2 Lite 服务化支持 —— 第一阶段评审说明（中文详版）

本文逐项说明本分支的**每一处改动及其原因**，并完整记录三类问题：

1. 本次按要求修复的三个**测量口径缺陷**（已完成）；
2. 修复过程中**新发现的第四个缺陷**（高严重度，未修，需批准）；
3. 与上游 `main` 的**对账决策**（哪些保留、哪些丢弃、哪些重实现）。

- 基线：上游 `NetX-lab/Frontier` `main` = `d71ad80b0800880808a0857fd30477e6d96592c6`
- 分支：`review/deepseek-v2-lite-upstream`
- 提交数：8（5 个功能提交 + 2 个测量修复 + 1 个文档提交）

---

## 0. 本阶段交付内容

1. **目标正确的模型契约**：DeepSeek V2 Lite = 16 个 query head、`q_lora_rank=null`、
   第 0 层 dense FFN 宽度 10944、路由专家宽度 1408、2 个 shared expert 融合为 2816。
2. **目标专用 H100 profile**（vLLM 0.27.0 内核实测）：`attention.csv`、`linear_op.csv`、
   `moe.csv` 及其 `*_kernel_only.csv` 家族。
3. **FLASHMLA 六 scope MLA 导入器**：不再硬编码 128 query head，不再硬编码单一 backend。
4. **vLLM 0.27 兼容层**：RMSNorm CustomOp、RoPE `rope_parameters`、`fused_experts()` 生产内核路径。
5. **MLA 外部投影计入**：不再把 norm / 预投影 / RoPE / 输出投影静默记为 0。
6. **EP lane batch 的 decode CUDA Graph 元数据传播修复**。

---

## 1. 与上游 `main` 的对账（先读这一节）

本工作最初基于一个已被上游甩开 **178 个提交**的基线，而上游**独立实现了部分同样能力**。
因此本分支**不是**原始补丁的逐字重放：

| 原始改动 | 上游状态 | 本分支处置 |
|---|---|---|
| 模型配置新增 `dense_mlp_hidden_dim` | 上游已加入 `dense_mlp_hidden_dim` **与** `routed_mlp_hidden_dim`，并接入 `model_architectures.py`、`shared_prediction_model_manager.py`、`param_counter.py` | **丢弃我们的版本**，采用上游设计 |
| `profiling_plan.py` 用 `_ffn_construction_dim` 构造 dense FFN | 上游改为由 **typed layer contract** 推导 `padded_n_expanded_embd`，且对 `is_moe` producer 特意保持 routed 宽度 | **丢弃我们的版本**，并删除已成死代码的 `_ffn_construction_dim` |
| `linear_op/main.py`、`linear_op_wrapper.py` 的宽度元数据 | 上游重构为 `_build_profile_result(model_config=...)` | **丢弃我们的版本** |
| `base_cluster_scheduler.py` 中的 EP lane 元数据传播 | 上游**删除**了该段代码（文件从 9,890 行降到 1,871 行），EP lane 构造搬到 `frontier/scheduler/utils/` | **在新位置重实现**（`batch_builders.py`、`expert_parallel.py`）。`base_cluster_scheduler.py` 现与上游逐字节一致 |
| MLA q-head 目标派生、FLASHMLA、MLA 外部投影计入、vLLM 0.27 兼容层、DeepSeek profile | 上游仍缺（`families.py:198` 仍是 `expected_n_q_head=128`；`_predict_mla_attention_layer_time` 仍只返回六 scope） | **保留** |

两个必须明确的后果：

1. **本分支的 `linear_op.csv` 是 legacy（非 typed）CSV**，不含 `typed_operator_contracts` 列。
   上游的非 typed 路径会硬过滤 `n_expanded_embd == mlp_hidden_dim`，对混合 dense+MoE 模型会
   丢掉 10944 的 dense 行并抛 "No compute profiling rows remain"。因此本分支把该过滤放宽为
   `{mlp_hidden_dim} ∪ {dense_mlp_hidden_dim}`。**该改动只扩容、不缩容**：对上游所有既有非 typed
   CSV，其行宽全为 `mlp_hidden_dim`，实际过滤结果不变；typed 路径（存在 `typed_operator_contracts`
   列时）优先级更高且未被触碰。
2. **上游所有已入库的 MoE profile 都是 `vllm_fused`**（见第 2.3 节），因此第 2.3 节的修复会
   改变所有 MoE 模型的分层时间——这是**精度修正**，不是回退，但属于行为变化，必须显式声明。

---

## 2. 三个指定问题的核查与修复

### 2.1 问题一：`attn_pre_proj` 计时聚合错误

**核查结论：确认存在。**

`DeepseekV2MlaCausalSelfAttention` 中 `q_proj` 与 `kv_a_proj_with_mqa` 都被赋予
`linear_metric_name="attn_pre_proj"`。`TimerStatsStore.record_time()` 把同名样本**追加进同一个
列表**，`get_stats()` 对该列表取 **中位数**：

```python
# frontier/profiling/common/timer_stats_store.py
self.TIMING_STATS[name].append(time)
...
"median": np.median(times)
```

所以记录下来的值不是一次 forward 内 `q_proj + kv_a_proj` 的求和，而是**两个不同算子交错样本的
中位数**，合并投影成本被系统性低估。

**修复方式**：与仓库既有的 `Step3TextCausalSelfAttention` 保持同一模式——

- 两个子算子各自独立计时：`attn_q_proj`、`attn_kv_a_proj`；
- 保留 `precision_op_name="attn_pre_proj"`，量化契约仍绑定规范名；
- 新增**外层** `CudaTimer("attn_pre_proj")` 包裹整个预投影块，使规范指标等于
  **单次 forward 的求和**。

**GPU 实测验证**（10.96.11.9 / GPU0，TP1，H100）：

kernel-only 家族（`--profile_method record_function`）：

| num_tokens | attn_pre_proj | attn_q_proj | attn_kv_a_proj | q+kv | 残差（= kv_a_layernorm） |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.02517 | 0.00688 | 0.00557 | 0.01245 | 0.01272 |
| 64 | 0.02946 | 0.00733 | 0.00581 | 0.01314 | 0.01632 |
| 1024 | 0.04752 | 0.01933 | 0.00782 | 0.02715 | 0.02037 |

即 `attn_pre_proj` 现在**确实等于** `q_proj + kv_a_proj + kv_a_layernorm`。
残差的量级与第 3 节发现的 shim 膨胀一致（见 3.1）。

**必须并存的注意事项（CUDA_EVENT 家族）**：同一探测在 `--profile_method cuda_event` 下
@1 token 得到 `attn_pre_proj = 0.24749 ms`，而 `q_proj + kv_a_proj = 0.05667 ms`，
残差高达 **0.19 ms**。原因是外层 event 窗口跨越了三次串行算子的 **CPU 发射间隙**，
而内层 event 只覆盖各自内核窗口。这属于既有的 eager 族固有偏差，也正是本仓库设计
「双家族」的原因：

- prefill / eager 路径 → CUDA_EVENT 家族（含发射开销，与真实 eager 执行一致）；
- 纯 decode + CUDA Graph → KERNEL_ONLY 家族（仅内核时间）。

所以**小 batch 的 decode 不应使用 CUDA_EVENT 家族的 `attn_pre_proj`**。

**附带修正**：首版修复中我对 `kv[:, :512]` 调用了 `.contiguous()`，这会引入真实 vLLM
**不存在**的拷贝内核（vLLM 直接对 `split` 产生的视图做 RMSNorm）。已改为传入 strided 视图，
并用独立微基准确认 vLLM 的 `rms_norm` 对 strided 输入与 contiguous 输入耗时相同
（6.2–6.6 µs，差异在噪声内），即无需拷贝。

---

### 2.2 问题二：`kv_a_layernorm` 未执行

**核查结论：确认存在。**

真实 vLLM 0.27.0 的 `<container>/vllm/model_executor/layers/mla.py:189-190`：

```python
kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
kv_c_normed = self.kv_a_layernorm(kv_c)
```

而我们的 profiler 之前只取 rope 切片，**完全没有执行该 RMSNorm**。同时
`frozen_comparison_contract_v1.yaml` 与既往报告都声称
`attn_pre_proj = q_proj + kv_a_proj_with_mqa + kv_a_layernorm`——**文档与实现不一致**。

**修复方式**：在模块内新增 `RMSNorm(kv_lora_rank, norm_name=None)`（**不单独计时**，
成本由外层 `attn_pre_proj` 复合计时器承担，与 vLLM 中该 norm 夹在两个投影之间的位置一致），
并在 forward 内对 `kv_lora_rank` 切片执行归一化。

**验证**：见 2.1 表格残差项——残差非零且随 token 数增长，证明该算子在真实执行且被计入。

**注意**：本次仅修复「未执行」。该 norm 的**绝对测量值本身偏高**（第 3 节）。

---

### 2.3 问题三：`fused_experts()` 与 `moe_shuffling` 边界重叠

**核查结论：确认存在，且影响所有 `vllm_fused` profile。**

源码链（容器内 vLLM 0.27.0 实测确认）：

```
fused_experts()                                   # fused_moe.py:1587
  └─ torch.ops.vllm.fused_experts
       └─ fused_experts_op()                      # fused_moe.py:1448
            └─ fused_experts_impl()               # fused_moe.py:1650
                 └─ _prepare_expert_assignment()  # fused_moe.py:1538
                      └─ moe_align_block_size(...)  # fused_moe.py:1578
```

并且**真实运行时的模块化实现** `TritonExperts`（`experts/triton_moe.py:309-321`）同样调用
`_prepare_expert_assignment(...)`。

而 Frontier 的 MoE profiler 把 `moe_align_block_size` **单独**测成 `moe_shuffling`
（`moe_impl.py` 的 `MoEShuffling.forward`，计时器只包住 `moe_align_block_size`），
预测器又把两者相加：

```python
total_moe_time = mlp_norm_time + gating_time + shuffling_time + grouped_gemm_time + ...
```

因此在 `vllm_fused` 路径下，**token 分配/对齐步骤被计算了两次**。

**影响范围**：仓库中所有已入库 MoE profile 的 `moe_grouped_gemm_backend` 均为 `vllm_fused`：

```
mixtral_8x7b_moe                     : (无该列)
qwen2_moe_example                    : (无该列)
qwen3-a3b-30b-moe                    : vllm_fused
qwen3-next-80b-a3b-instruct-reduced-l2 / l20 : vllm_fused
Phi-tiny-MoE-instruct                : vllm_fused
Qwen3-30B-A3B-tiny                   : vllm_fused
Step2Mini-tiny                       : vllm_fused
step-moe-noquant-small               : vllm_fused
```

**修复方式**（预测器侧，无需重采数据）：

- 新增 `_moe_grouped_gemm_includes_assignment()`：读取 MoE profile 的
  `moe_grouped_gemm_backend` 列，当包含 `vllm_fused` 时判定 grouped-GEMM 项已内含对齐步骤；
- `_get_moe_shuffling_time()` 在该情形下返回 `0.0`，并输出 debug 日志说明原因；
- provenance 每个预测器**只解析一次并缓存**，避免长仿真中途口径翻转；
- `frontier_loop` 与不含该列的 legacy profile **保持原行为**（那里对齐不在 grouped-GEMM 测量内）。

**这是精度修正**：会改变所有 fused profile 的 MoE 分层时间。方向是去掉重复计数，
但属于上游行为的显式变更。

---

## 3. 修复过程中新发现的缺陷（高严重度，**未修**，需批准）

### 3.1 缺陷四：Frontier 的 RMSNorm shim 测出的 device 时间比 vLLM 原生算子高约 7–10 倍

**发现过程**：修完问题一/二后，用 kernel-only 模式验证 `attn_pre_proj` 的残差。理论上残差
应等于 512 宽 RMSNorm 的内核时间（数微秒），实测却是 12.7–20.4 µs。于是用同一套
`RecordFunctionTracer` 做了独立对照实验（10.96.11.9 / GPU0）：

| num_tokens | Frontier `RMSNorm` shim | vLLM 原生 `ops.rms_norm` | shim 多报 |
|---:|---:|---:|---:|
| 1 | 11.87 µs | 1.70 µs | **+10.17 µs** |
| 64 | 14.30 µs | 1.76 µs | **+12.54 µs** |
| 1024 | 18.56 µs | 2.46 µs | **+16.10 µs** |

并且已验证**不是** `hidden_size` 造成的（`VllmRMSNormClass(1)` 与 `VllmRMSNormClass(512)`
结果相同，分别 11.87 / 11.87 µs @1 token，18.34 / 18.45 µs @1024 token），
即多报来自 `torch.ops.vllm.rms_norm` CustomOp 派发路径本身。

**根因位置**：`frontier/profiling/common/layers/layernorm.py` 的 `_VllmRmsNormShim`。
它构造 `VllmRMSNormClass(1, eps=1e-6)` 并在每次调用时 `object.__setattr__` 绑定 weight，
再走 `self._op(x)`（即 CustomOp）。真实 vLLM 在该容器里 `rms_norm` 的优先级是 `native`，
实际走的是 `forward_native → ops.rms_norm`。shim 绕了一层 CustomOp 派发，导致
被计入 span 的 device 时间显著膨胀。

**影响范围（跨模型、既有）**：
`input_layernorm`、`post_attention_layernorm`、`attn_inter_norm`、以及本次新增的
`kv_a_layernorm` 等**所有经该 shim 测量的 norm 项**都被高估约 10–16 µs/次。
由于每层都有 2 个 norm，27 层、128 token 的 decode 会累积到毫秒量级。

**为什么本次不修**：
- 该 shim 被**所有模型**的 linear-op / MoE profiling 使用；
- 修它相当于改变全部既有 profile 的 norm 数值，按仓库开发准则必须有「显式批准的
  fidelity fix」；
- 修完还必须**重采全部既有 CSV**，否则新旧数据不可比。

**建议修复方式**（待批准）：让 shim 在可用时直接调用与真实 vLLM 相同的原生算子
（`ops.rms_norm` / `ops.fused_add_rms_norm`），而不是经 CustomOp；随后重采受影响的
`linear_op.csv` 家族。

**当前后果声明**：本分支重采后的 `attn_pre_proj` 在结构上已正确（= q + kv_a + norm 的求和），
但其中 norm 分量目前被上述缺陷放大约 10–16 µs。在缺陷修复并重采之前，
**`attn_pre_proj` 的绝对值不可用于跨实现对比**。

---

## 4. 逐文件改动说明（中文）

### 4.1 数据与模型配置

| 文件 | 原因 |
|---|---|
| `data/config/models/DeepSeek__DeepSeekV2-Lite.json` | 让 Frontier 正确解析模型结构：16 q head、`q_lora_rank=null`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、`v_head_dim=128`、`kv_lora_rank=512`、`intermediate_size=10944`、`moe_intermediate_size=1408`、64 路由 + 2 shared、top-6、`moe_layers_enum=1..26` |
| `.../h100/DeepSeek/DeepSeekV2-Lite/attention.csv` | FLASHMLA 六 scope CUDA-event 测量，TP1/2/4、H100、BF16、block 64 |
| `.../attention_kernel_only.csv` | 同上形状的 kernel-only 家族，供 piecewise CUDA Graph 下的纯 decode 使用 |
| `.../linear_op.csv` | MLA 外部投影 + `o_proj` + dense FFN 10944 + shared expert 2816，vLLM 0.27.0 内核测量，tokens 1..4096，TP1/2/4 |
| `.../linear_op_kernel_only.csv` | 同上 kernel-only 家族 |
| `.../moe.csv` | vLLM 0.27 `fused_experts()` 生产内核，TP×EP 网格，uniform 路由 |
| `.../moe_kernel_only.csv` | 同上 kernel-only 家族 |

> 数据状态：这些 CSV 由**修复前**的 profiler 生成，`attn_pre_proj` 仍是「同名中位数」口径，
> 且未包含 `kv_a_layernorm`。**必须重采**（见第 6 节）。

### 4.2 契约与导入

| 文件 | 原因 |
|---|---|
| `frontier/attention/families.py` | `expected_n_q_head=None`：query head 数是**目标模型**属性（本模型 16、DeepSeek V3 为 128），不应由家族全局固定。`expected_attention_backend=None`：backend 是**运行时**属性（旧 H800 探针用 FLASHINFER_MLA，vLLM 0.27.0 用 FLASHMLA） |
| `frontier/attention/ops.py` | `AttentionRuntimeMetaContract` 增加可选 `expected_attention_backend`，声明时校验非空 |
| `frontier/profiling/attention/vllm_mla_profile_importer.py` | 支持显式 `num_q_heads`（回退顺序：入参 → 行内 `n_q_head` → 家族契约）；受支持 backend 扩展为 `FLASHINFER_MLA`/`FLASHMLA` |
| `frontier/profiling/attention/main.py` | 新增 `--mla_num_q_heads`，`--attention_backend` 允许 `FLASHMLA`；未提供时从模型配置解析 |

### 4.3 模型配置与预测器

| 文件 | 原因 |
|---|---|
| `frontier/config/model_config.py` | 当无显式 `share_expert_dim` / `shared_expert_intermediate_size` 时，由 `n_shared_experts × moe_intermediate_size` 推断 shared expert 宽度（DeepSeek 把 2 个 shared expert 融合为一个 2816 的 FFN）。注：本文件中的 `dense_mlp_hidden_dim` / `routed_mlp_hidden_dim` 是**上游的**，我们的重复赋值已在对账中删除 |
| `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | ① MLA 外部投影计入 `_predict_mla_attention_layer_time`（见 4.4）；② legacy 非 typed 宽度过滤放宽为 `{mlp_hidden_dim} ∪ {dense_mlp_hidden_dim}`；③ 训练期对 `mlp_up_proj`/`mlp_act`/`mlp_down_proj` 仅取 dense 宽度行，避免 dense 模型学到专家形状数据 |
| `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` | ① DeepSeek 家族（`deepseek_v2`/`deepseek_v3`/`deepseek_mtp`）的 dense 层改走真实 `mlp_*` 模型，而非 Step2Mini/Step3 的 shared-expert 映射（后者会把 10944 错记为 2816）；② **本次新增**：`_moe_grouped_gemm_includes_assignment()` + `_get_moe_shuffling_time()` 去重（见 2.3） |

### 4.4 MLA 外部投影（本次两处修复所在）

文件：`frontier/profiling/linear_op/linear_op_impl.py`，类 `DeepseekV2MlaCausalSelfAttention`。

| 改动 | 原因 |
|---|---|
| `q_proj` 计时名改为 `attn_q_proj`，`kv_a_proj_with_mqa` 改为 `attn_kv_a_proj`，两者保留 `precision_op_name="attn_pre_proj"` | 避免同名样本被取中位数而非求和（问题一） |
| 新增外层 `CudaTimer("attn_pre_proj")` 包裹 q_proj + kv_a_proj + kv_a_layernorm | 使规范指标等于单次 forward 的求和 |
| 新增 `RMSNorm(kv_lora_rank, norm_name=None)` 并对 `kv_lora_rank` 切片执行 | 复现真实 vLLM 算子序列（问题二）；不再静默跳过 |
| 切片保持 strided 视图，不做 `.contiguous()` | vLLM 对 `split` 视图直接做 RMSNorm，不存在该拷贝 |

### 4.5 vLLM 0.27 兼容层

| 文件 | 原因 |
|---|---|
| `frontier/profiling/common/layers/layernorm.py` | vLLM 0.27 把 RMSNorm 改为 CustomOp，模块级 `rms_norm`/`fused_add_rms_norm` 函数消失。新增 shim 以保持旧签名（**该 shim 即第 3 节缺陷所在**） |
| `frontier/profiling/common/layers/rotary_embedding.py` | vLLM 0.27 用 `rope_parameters` 字典替代 `rotary_dim/base/is_neox_style/rope_scaling`，且 CustomOp 需要当前 vLLM config context。这是此前 RoPE 被迫走 Torch fallback（~140 µs vs 真实数微秒）的原因 |
| `frontier/profiling/moe/moe_impl.py` | vLLM 0.27 的 `fused_topk` 迁至 router 包、`get_config_dtype_str` 更名；新增 `ensure_vllm_profiling_runtime()` 初始化最小运行时（CustomOp 需要的 `VllmConfig`、`ReplicatedLinear` 需要的单进程 TP group），失败时静默以保持旧版行为 |
| `frontier/profiling/moe/moe_vllm_kernel.py` | vLLM ≥0.27 走 `fused_experts()`（与生产 `TritonExperts` 同入口）。此前退化为逐专家 Python 循环，1024 tokens 实测 9.33 ms vs 真实 0.588 ms，约 **19 倍**误差 |

### 4.6 调度器

| 文件 | 原因 |
|---|---|
| `frontier/scheduler/utils/batch_builders.py` | `build_ep_lane_batch()` 把源 batch 的 `decode_cuda_graph_metadata` 传播到 EP lane batch。EP lane 的合成请求会把路由 token 数记成 `num_prefill_tokens`，若无该元数据，measurement-family 选择器会短路到 eager 家族 |
| `frontier/scheduler/utils/expert_parallel.py` | `materialize_batch_group()` 同样向 EPBatchGroup 传播该元数据（上游 `create_ep_batch_group` 拿不到源 batch，只能在调用点传播） |
| `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py` | **本分支对该文件无净改动**（已与上游逐字节一致）；原补丁所针对的代码已被上游删除 |

### 4.7 测试

| 文件 | 原因 |
|---|---|
| `tests/unit/test_deepseek_mla_linear_op_accounting.py`（**新增**） | 守卫：子算子计时名互不相同、复合计时器持有规范名且未被禁用、`kv_a_layernorm` 存在且宽度为 `kv_lora_rank`、norm 不单独计时、builder 的路径选择与 `q_lora_rank` 拒绝逻辑 |
| `tests/unit/test_moe_fused_assignment_accounting.py`（**新增**） | 守卫：`vllm_fused` 判定、shuffling 抑制、`frontier_loop`/legacy/缺失 profile 保持原行为、provenance 只解析一次 |
| `mla_h800_fixture.py`、`test_attention_family_spec_data.py`、`test_mla_predictor_*`、`test_mla_vllm_profile_importer.py` | 家族契约由「固定 128 q head、单一 backend」变为「目标派生」，相应夹具与断言更新；`test_attention_family_spec_data.py` 现在显式断言契约为 `None`/`None` |

---

## 5. 验证结果

### 5.1 单元测试

```
702 passed, 1 skipped, 1 failed
```

覆盖 `test_mla_*.py`、`test_attention_*.py`、`test_profiling_*.py`、`test_operator_*.py`、
`test_linear_op_*.py`、`test_parallel_semantics.py`、`test_measurement_family_selector.py`、
`test_model_architecture_registry.py`、`test_ffn_memory_operator_families.py`、
`test_execution_time_op_times.py`、以及两个新增测试文件。

MoE 相关集合 `test_moe_*.py + test_typed_ep_*.py`：`422 passed, 6 failed`。

**唯一/全部失败项均在上游 `main`（`d71ad80`）的干净 worktree 中复现同样失败**，
属既有问题，**非本分支回归**：

- `test_mla_stage3_online_trace_builder.py::test_mla_stage3_cli_writes_trace_and_error_matrix`
- `test_moe_ep_baseline_replay.py` 的 5 项
- `test_typed_ep_trace_contract.py::test_ep_trace_helper_consumes_typed_lane_descriptor`

### 5.2 GPU 实测验证（10.96.11.9 / GPU0）

```
docker run --rm --gpus '"device=0"' ... ontos:vllm-0.27.0 \
  -m frontier.profiling.linear_op.main \
  --disable_ray --num_gpus 1 --device h100 --models DeepSeek/DeepSeekV2-Lite \
  --num_tensor_parallel_workers 1 --num_tokens_list 1 64 1024 \
  --profile_method record_function --is_moe --output_dir /out --yes
```

结果见 2.1 与 3.1 的表格。结论：

1. `attn_pre_proj` 已等于 `q_proj + kv_a_proj + kv_a_layernorm`（结构正确 ✅）；
2. `kv_a_layernorm` 确实执行（残差非零且随 token 增长 ✅）；
3. 但 norm 分量被 shim 放大约 7–10 倍（❌，见第 3 节）。

---

## 6. 数据重采状态与命令

`linear_op.csv` / `linear_op_kernel_only.csv` 目前仍是**修复前**口径（同名中位数、无
`kv_a_layernorm`），**必须重采**。但建议**在第 3 节 shim 缺陷修复之后再执行**，
否则会把已知偏差固化进数据。

重采网格（从现有 CSV 反推，共 259 个 token 点，与原数据逐点可比）：

```
1, 2, 4, 8, 16, 24, 32, ..., 1032   (步长 8)
1040, 1056, ..., 2080                (步长 16)
2080, 2112, ..., 4096                (步长 32)
```

即 `--num_tokens_list <259 个点>`；TP 覆盖 `--num_tensor_parallel_workers 1 2 4`；
两种 `--profile_method`（`cuda_event` 与 `record_function`）各跑一次，输出到
`data/profiling/compute/h100/DeepSeek/DeepSeekV2-Lite/`。

---

## 7. 环境与主机调度约束

按当前工作区规则：**1–2 卡实验用 `10.96.11.7`；≥4 卡实验在
`10.96.11.9 / .12 / .11 / .10` 中按 9 → 12 → 11 → 10 优先级选空闲节点。**

本次实测节点状态（2026-09-11 09:56）：

| 主机 | 负载 | 是否有 `ontos:vllm-0.27.0` | 空闲 GPU |
|---|---:|---|---|
| 10.96.11.7 (gpu007) | **204** | **无**（仅 0.26.0 / 0.17.1） | 6（GPU2–7） |
| 10.96.11.9 (gpu009，当前主机) | 39 | **有** | 8 |
| 10.96.11.10 (gpu010) | 110 | 无 | 8 |
| 10.96.11.11 (gpu011) | 43 | **有** | 8 |
| 10.96.11.12 (gpu012) | 49 | 无 | 8 |

**冲突与建议**：规则要求 1–2 卡任务放 `10.96.11.7`，但该节点**没有 0.27.0 镜像**
且负载已达 204。因此本次 1 卡验证**临时执行在 `10.96.11.9`**。若需继续在该节点跑
1–2 卡任务，建议先批准此偏离，或把 `ontos:vllm-0.27.0` 推送到 `.7`。

---

## 8. 遗留事项（按优先级）

1. **P0｜修复 RMSNorm shim 的 CustomOp 派发膨胀**（第 3 节），随后重采全部受影响 CSV。
   这是当前 `attn_pre_proj` 绝对精度不可用的直接原因。
2. **P0｜重采 `linear_op.csv` / `linear_op_kernel_only.csv`**（第 6 节），使数据与已修
   的 profiler 口径一致。
3. **P1｜迁移到 typed operator contract**：让 DeepSeek 的 `linear_op.csv` 携带
   `typed_operator_contracts` 列，从而删除本分支对 legacy 宽度过滤的放宽。
4. **P1｜确认 MoE 去重带来的既有基线变化**：第 2.3 节的修复会改变所有 `vllm_fused`
   profile 的 MoE 分层时间，需要同步更新受影响的期望值/回归基线。
5. **P2｜`_ffn_construction_dim` 已删除**，`profiling_plan.py` 完全交由上游 typed contract 驱动。

---

## 9. 变更摘要（对账视角）

- **保留**：目标派生的 MLA q-head 契约、FLASHMLA backend、MLA 外部投影计入、
  vLLM 0.27 兼容层（RMSNorm shim / RoPE / `fused_experts`）、DeepSeek profile、
  本次两个测量修复。
- **丢弃**：我们的 `dense_mlp_hidden_dim` 与 profiling-plan 宽度改动（让位给上游更完整的
  typed contract 设计），以及 `base_cluster_scheduler.py` 上的补丁（目标代码已被上游删除）。
- **重实现**：EP lane CUDA Graph 元数据传播（迁移到 `batch_builders.py` 与
  `expert_parallel.py`）。
