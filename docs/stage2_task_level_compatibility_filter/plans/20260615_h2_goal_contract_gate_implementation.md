# 2026-06-15 H2 Goal-Contract Memory Gate 实现文档

## 1. 目标

H2 实验目标是在保留 HiAgent 原始联合 `subgoal + action` prompt 结构的前提下，为 C1 GPT-OSS insight-only GMemory 注入增加一个轻量、确定性、agent-side 的 per-insight goal-contract gate。

H2 对应实验组：

```text
H0: HiAgent-only
H1: C1 GPT-OSS insight-only 原始注入
H2: C1 GPT-OSS insight-only + per-insight Goal-Contract Gate rule_v1
```

H2 只回答一个问题：

> 在不改变 GMemory server、不改变 HiAgent 主体 prompt 结构的前提下，per-insight rule gate 是否能减少与当前 task goal contract 不兼容的 retrieved insights，从而降低 injected-worse cases？

H2 不做：

```text
LLM judge
safe-hint rewrite
score threshold sweep
retrieval reranking
step-level retrieval
action-only split
GMemory server 改造
```

## 2. 任务范围

### 2.1 Codex 实现任务

1. 在 `GMemoryContextEfficientAgent` 中增加 H2 gate 开关和诊断字段。
2. 在 GMemory retrieve 后、prompt injection 前执行 per-insight keep/drop。
3. 实现 goal contract parser。
4. 实现 C1 insight-only prompt splitting。
5. 实现 rule_v1 风险判断。
6. 实现 kept insights prompt reconstruction。
7. 暴露 `get_diagnostics()`，让 ALFWorld 已有日志系统能写出 gate diagnostics。
8. 新增 H2 eval config，默认关闭 upload，固定 C1 insight-only 服务端 rendering。
9. 新增轻量离线/单元检查，覆盖 parser、split、gate、diagnostics 基本行为。

### 2.2 服务器参与任务

1. 启动或确认 GMemory API server 可用。
2. 启动或确认 executor LLM 服务可用，例如当前 Qwen3.5-4B vLLM server。
3. 使用固定 memory DB snapshot 运行 H1/H2 对比。
4. evaluation 阶段必须关闭 GMemory upload。
5. 汇总 H2 相对 H1/H0 的 SR、PR、grounding、injected-worse、drop reason 分布。

## 3. 需要修改的文件

### 3.1 必改文件

```text
agentboard/agents/gmemory_agent.py
```

修改点：

```text
GMemoryContextEfficientAgent.__init__()
  - 读取 gmemory.goal_contract_gate 配置
  - 初始化 self.gmemory_gate_diagnostics

GMemoryContextEfficientAgent.reset()
  - gate disabled 或配置缺失时：完全走旧逻辑 self._filter_gmemory_prompt(raw_prompt)
  - gate enabled 时：走 H2 专用 prepare/gate/limit 路径
  - H2 专用路径中按 memory_only 条件调用 _extract_memory_sections()
  - H2 gate 完成后再应用 max_context_chars 最终长度限制
  - 不直接重写 _filter_gmemory_prompt() 的现有行为

新增方法：
  - _goal_contract_gate_enabled()
  - _prepare_gmemory_prompt_for_gate(memory_prompt)
  - _limit_gmemory_prompt_chars(memory_prompt)
  - _parse_goal_contract(goal)
  - _split_insights(memory_prompt)
  - _assess_goal_contract_risk(contract, insight, initial_observation="")
  - _reconstruct_insight_prompt(kept_insights)
  - _gate_gmemory_prompt_per_insight(memory_prompt)
  - get_diagnostics()
```

`get_diagnostics()` 至少输出：

```json
{
  "agent_name": "GMemoryContextEfficientAgent",
  "gmemory_prompt_chars": 0,
  "memory_injected_to_prompt": false,
  "gmemory_gate": {
    "enabled": true,
    "mode": "per_insight_rule_v1",
    "task_decision": "inject | skip",
    "contract": {},
    "insight_count": 0,
    "kept_count": 0,
    "dropped_count": 0,
    "dropped_insights": []
  }
}
```

### 3.2 新增配置文件

建议新增：

```text
eval_configs/hiagent/alfworld_h2_goal_contract_gate_c1.yaml
```

配置要求：

```yaml
agent:
  name: GMemoryContextEfficientAgent
  gmemory:
    enabled: true
    base_url: http://127.0.0.1:8090
    task_type: alfworld
    recall_on_reset: true
    upload_on_finish: false
    memory_only: true
    max_context_chars: 5000
    timeout: 180.0
    experiment_render_label: C1_GPT_OSS_insight_only_server_side
    goal_contract_gate:
      enabled: true
      mode: per_insight_rule_v1
      split_mode: c1_insight_lines
      min_kept_insights: 1
      max_kept_insights: 3
```

默认原则：

```text
现有配置不应被 H2 改动影响。
如果 goal_contract_gate 缺失或 enabled=false，GMemoryContextEfficientAgent 行为必须与当前 H1 一致。
```

### 3.3 新增本地验收脚本

新增：

```text
scripts/check_goal_contract_gate.py
```

作用：

```text
本地快速工程验收 H2 gate 的核心逻辑。
不启动 GMemory server。
不启动 vLLM / OpenAI-compatible executor server。
不运行 ALFWorld。
不判断 H2 是否优于 H1。
```

用途：

```text
在 full-134 服务器实验前，用几秒钟确认：
1. goal contract parser 行为正确；
2. C1 insight-only prompt 能被拆分；
3. risk rules 能 drop 明显不兼容的 insight；
4. kept insights 能重建成合法 prompt；
5. kept_count=0 时会跳过注入；
6. diagnostics 可 JSON 序列化并包含 gate 结果。
```

实现方式采用 fake LLM 实例化 `GMemoryContextEfficientAgent`：

```python
class FakeLLM:
    engine = "fake"
    context_length = 32768
    max_tokens = 512

    def num_tokens_from_messages(self, messages):
        return sum(len(message.get("content", "")) for message in messages)
```

脚本直接调用 agent helper：

```text
_parse_goal_contract()
_split_insights()
_gate_gmemory_prompt_per_insight()
get_diagnostics()
```

定位：

```text
这是 Codex 部分验收工具，不是 H2 实验工具。
它只证明 H2 gate 的规则机器能按预期运行。
真实收益必须通过服务器参与的 H1/H2/H0 full-134 对比验证。
```

### 3.4 暂不修改文件

```text
agentboard/agents/split_action_agent.py
GMemory/*
agentboard/tasks/alfworld.py
agentboard/utils/logging/logger.py
```

说明：

```text
H2 主实验只作用于 GMemoryContextEfficientAgent。
ALFWorld logging 已经支持 agent_diagnostics；只要 agent 暴露 get_diagnostics()，不需要改 task/logger。
action-only agent 是另一条变量链，不进入 H2。
GMemory server/rendering 不进入 H2。
```

## 4. H2 Gate 规则

### 4.1 Goal Contract Parser

输入：

```text
goal
initial_observation
```

第一版只从 `goal` 抽取：

```json
{
  "count_constraint": "one | two | multiple | unknown",
  "state_requirement": "clean | hot | cool | none",
  "final_action": "put | examine | use | unknown",
  "target_object": "",
  "target_receptacle_or_tool": "",
  "needs_intermediate_state": false,
  "needs_finalization": true,
  "completion_pattern": "state_change_then_finalize | direct_place | multi_object_place | light_or_examine | unknown"
}
```

注意：

```text
目标是 goal contract，不是 ALFWorld task type label。
但 parser 可以从自然语言 goal 中识别 clean/hot/cool、two/all、put/examine/use。
```

### 4.2 Insight Split

支持 C1 insight-only 格式：

```text
## Key Insights from Related Tasks
1. ...
2. ...
3. ...
```

也支持：

```text
- ...
* ...
非空行 fallback
```

如果无法拆分：

```text
fallback to whole prompt as one insight
diagnostics.split_failed = true
```

### 4.3 Risk Rules

第一版 drop reasons：

```text
cardinality_mismatch
finalization_missing
over_verification_risk
stage_drift
```

H2 主线暂不加入：

```text
state_tool_conflict
cool -> fridge
heat -> microwave/stoveburner
clean -> sinkbasin
```

原因：

```text
H2 的主线目标是先验证抽象 goal-contract gate。
cool/tool semantic conflict 虽然是现有 flip analysis 中的重要负迁移来源，但它会引入更强的 ALFWorld domain-specific 语义映射。
该能力放入备选计划或后续 H2b/H3，不进入 H2 rule_v1 主实验。
```

### 4.4 Decision Policy

```text
if memory_prompt is empty:
  task_decision = skip

split memory_prompt into insights

for each insight:
  if any risk rule fires:
    drop with reasons
  else:
    keep

if kept_count >= min_kept_insights:
  reconstruct prompt from kept insights
  apply max_context_chars to reconstructed prompt
  task_decision = inject
else:
  final prompt = ""
  task_decision = skip
```

`max_kept_insights` 第一版使用原始顺序截断：

```text
kept_insights = kept_insights[:max_kept_insights]
```

不做 score 排序，避免引入新变量。

### 4.5 长度限制顺序

H2 正式实现不沿用“先按 `max_context_chars` 截断，再 split/gate”的顺序，但这个改变只允许发生在 gate enabled 的 H2 专用路径中。

Backward compatibility 是硬约束：

```text
gate disabled 或 goal_contract_gate 配置缺失时：
  必须完全调用现有 _filter_gmemory_prompt(raw_prompt)
  不改变现有 memory_only 行为
  不改变现有 max_context_chars 截断位置
  不改变现有 self.gmemory_prompt 结果
```

不要直接重写 `_filter_gmemory_prompt()` 让所有配置都变成新顺序。推荐保留旧函数不动，新增 H2 专用 helper：

```python
raw_prompt = response.get("memory_prompt", "")

if self._goal_contract_gate_enabled():
    prepared_prompt = self._prepare_gmemory_prompt_for_gate(raw_prompt)
    gated_prompt = self._gate_gmemory_prompt_per_insight(prepared_prompt)
    self.gmemory_prompt = self._limit_gmemory_prompt_chars(gated_prompt)
else:
    self.gmemory_prompt = self._filter_gmemory_prompt(raw_prompt)
```

其中 `_prepare_gmemory_prompt_for_gate()` 应按条件调用 `_extract_memory_sections()`：

```python
def _prepare_gmemory_prompt_for_gate(self, text: str) -> str:
    context = normalize_whitespace(text)
    if self.gmemory_memory_only:
        context = self._extract_memory_sections(context)
    return context
```

推荐顺序：

```text
raw memory_prompt
-> normalize whitespace
-> if memory_only: _extract_memory_sections()
-> _split_insights()
-> _assess_goal_contract_risk()
-> _reconstruct_insight_prompt()
-> apply max_context_chars to final reconstructed prompt
-> self.gmemory_prompt
```

原因：

```text
如果先截断，可能把最后一条 insight 切断，导致 split 或 risk 判断误判。
gate 应该评估完整 insight 单元；长度限制应作为最终 prompt budget 控制。
```

## 5. 验收标准

### 5.1 Codex 部分验收标准

Codex 部分只验证工程正确性，不验证 H2 实验收益。

必须满足：

```text
1. goal_contract_gate.enabled=false 或缺失时，必须完全走旧 _filter_gmemory_prompt(raw_prompt) 路径。
2. gate disabled 时，原 GMemory 注入行为保持不变，包括 memory_only 和 max_context_chars 的旧顺序。
3. gate enabled 时，走 H2 专用 prepare/gate/limit 路径，先按 memory_only 条件抽取 memory sections，再 split/gate/reconstruct，最后应用 max_context_chars。
4. 空 memory 不注入，并写出 diagnostics。
5. C1 numbered insights 能被拆成多条 insight。
6. 无法拆分时 fallback 为 whole prompt，并记录 split_failed。
7. put two / find two 目标能识别 count_constraint=two。
8. clean/hot/cool 目标能识别 state_requirement。
9. put/examine/use 目标能识别 final_action。
10. 命中 cardinality_mismatch 时 drop。
11. 命中 over_verification_risk 时 drop。
12. kept_count=0 时 self.gmemory_prompt=""，make_prompt 不注入 memory。
13. _extract_memory_sections() 只在 self.gmemory_memory_only=true 的 H2 专用 prepare 阶段调用。
14. get_diagnostics() 返回 gate diagnostics，并能被 ALFWorld task logger 接收。
15. Python 静态编译通过。
```

不要求 Codex 部分完成：

```text
full-134 evaluation
GMemory API connectivity
vLLM executor connectivity
真实 SR/PR 改善
```

### 5.2 服务器参与测试验收标准

服务器测试验证 H2 实验效果。

基础通过标准：

```text
H2 injected_worse < H1 injected_worse
H2 SR/PR >= H1
H2 grounding_acc 不明显低于 H1
insight_drop_rate > 0
insight_keep_rate > 0
```

强通过标准：

```text
H2 接近或超过 H0
task_injection_rate 非零
skipped_would_be_worse 明显多于 skipped_would_be_better
puttwo 的 injected-worse 明显减少
cool injected-worse 不因 gate 明显恶化
```

失败标准：

```text
H2 SR/PR 低于 H1
或 skipped_would_be_better 多于 skipped_would_be_worse
或 gate 几乎全部 drop，退化为 no-memory
或 diagnostics 缺失导致无法解释 keep/drop
```

## 6. 验收方法

### 6.1 Codex 部分验收方法

#### 6.1.1 静态检查

运行：

```bash
python -m py_compile agentboard/agents/gmemory_agent.py
```

以及：

```bash
python -m py_compile scripts/check_goal_contract_gate.py
```

#### 6.1.2 单元/离线样例检查

运行：

```bash
python scripts/check_goal_contract_gate.py
```

该脚本使用 fake LLM 实例化 `GMemoryContextEfficientAgent`，不访问任何服务器。

至少覆盖以下样例：

```text
Goal: put a clean plate in countertop.
Expected:
  count_constraint=one
  state_requirement=clean
  final_action=put
  completion_pattern=state_change_then_finalize
```

```text
Goal: put two cd in safe.
Expected:
  count_constraint=two
  state_requirement=none
  final_action=put
  completion_pattern=multi_object_place
```

```text
Goal: examine the alarmclock with the desklamp.
Expected:
  count_constraint=one
  final_action=examine
  completion_pattern=light_or_examine
```

```text
Memory:
## Key Insights from Related Tasks
1. Find and place the first object, because it completes the immediate goal.
2. Check inventory repeatedly to ensure progress, because observations can be ambiguous.

Goal:
put two cd in safe.

Expected:
  insight 1 dropped by cardinality_mismatch
  insight 2 dropped by over_verification_risk or cardinality_mismatch
  task_decision=skip if no kept insights
```

脚本成功输出建议：

```text
PASS goal contract parser
PASS insight split
PASS cardinality mismatch gate
PASS over verification gate
PASS reconstruction and diagnostics
```

脚本失败时应抛出 `AssertionError` 或返回非零退出码，方便 Codex/终端直接判断验收失败。

#### 6.1.3 Config backward compatibility

用现有 H1/C1 配置加载 agent，确认：

```text
goal_contract_gate 缺失时不会报错
gate diagnostics enabled=false
GMemory prompt filter/inject 行为保持原样，包括原有 max_context_chars 位置和结果
_filter_gmemory_prompt() 的输出与实现前一致
```

#### 6.1.4 Diagnostics 验收

无需服务器的最小检查：

```text
手工设置 agent.gmemory_prompt 和 agent.gmemory_gate_diagnostics
调用 get_diagnostics()
确认返回 dict 可 JSON 序列化
```

如果跑一个 mock retrieval，则确认样本日志中包含：

```json
"agent_diagnostics": {
  "gmemory_gate": {
    "enabled": true
  }
}
```

### 6.2 服务器参与测试验收方法

#### 6.2.1 前置条件

必须固定：

```text
same memory DB snapshot
same task order
same executor model
same C1 insight-only rendering
upload_on_finish=false
max_num_steps=30
num_exam=134
```

服务：

```text
GMemory API server: http://127.0.0.1:8090
executor LLM server: OPENAI_API_BASE 指向 vLLM/OpenAI-compatible endpoint
```

#### 6.2.2 H1 复跑或确认基线

如果已有 H1 可复用，确认它满足：

```text
Agent: GMemoryContextEfficientAgent
Rendering: C1_GPT_OSS_insight_only_server_side
upload_on_finish=false
num_exam=134
```

如果需要复跑：

```bash
export EVALTASK=alfworld
export STEP=30
python agentboard/eval_main.py \
  --cfg-path eval_configs/hiagent/alfworld_h1_c1_insight_only.yaml \
  --tasks alfworld \
  --model qwen3_5_4b_vllm_server \
  --log_path ./logs/alfworld/GMemoryContextEfficientAgent/test_H1_C1_fixed_qwen3_5_4b_vllm_server_GPT-OSS-120B_30 \
  --project_name none \
  --baseline_dir ./data/baseline_results \
  --max_num_steps 30 \
  --memory_size 100 \
  --agent GMemoryContextEfficientAgent
```

如果没有单独 H1 config，可从 H2 config 复制一份并设置：

```yaml
goal_contract_gate:
  enabled: false
```

#### 6.2.3 H2 运行

```bash
export EVALTASK=alfworld
export STEP=30
python agentboard/eval_main.py \
  --cfg-path eval_configs/hiagent/alfworld_h2_goal_contract_gate_c1.yaml \
  --tasks alfworld \
  --model qwen3_5_4b_vllm_server \
  --log_path ./logs/alfworld/GMemoryContextEfficientAgent/test_H2_goal_contract_gate_C1_qwen3_5_4b_vllm_server_GPT-OSS-120B_30 \
  --project_name none \
  --baseline_dir ./data/baseline_results \
  --max_num_steps 30 \
  --memory_size 100 \
  --agent GMemoryContextEfficientAgent
```

#### 6.2.4 结果汇总

从以下文件读取总体指标：

```text
logs/.../alfworld.txt
logs/.../logs/alfworld.jsonl
```

注意：

```text
当前 alfworld.jsonl 实际是 pretty-printed 多行 JSON 对象流，
不能简单按行 json.loads。
汇总脚本需要按 JSON object stream 解析。
```

需要汇总：

```text
Success Rate
Progress Rate
Grounding Accuracy
Average Steps
task_inject_count
task_skip_count
task_injection_rate
insight_total_count
insight_kept_count
insight_dropped_count
drop_reason_distribution
final_memory_avg_chars
```

与 H0/H1 对齐：

```text
injected_better
injected_worse
skipped_would_be_better
skipped_would_be_worse
cool injected_worse
puttwo injected_worse
```

说明：

```text
H2 主线不包含 state_tool_conflict，因此 cool 指标用于风险观察和后续备选计划判断；
不把 cool 显著改善作为 H2 主验收条件。
```

#### 6.2.5 人工 spot check

至少检查以下类型：

```text
cool worse ids:
  12, 13, 39, 61, 95, 103, 111, 116

puttwo worse ids:
  43, 53, 112

positive state-change ids:
  7, 14, 20, 31, 76, 83, 88, 121
```

检查内容：

```text
1. risky insights 是否被 drop。
2. useful insights 是否仍有部分 keep。
3. kept prompt 是否仍是合法 C1 insight-only block。
4. task_decision=skip 的任务是否符合风险解释。
5. H2 是否因为过度 drop 失去原本 H1 的正向收益。
```

## 7. 预期产物

Codex 完成后应产生：

```text
agentboard/agents/gmemory_agent.py 修改
eval_configs/hiagent/alfworld_h2_goal_contract_gate_c1.yaml 新增
scripts/check_goal_contract_gate.py 新增
```

服务器实验完成后应产生：

```text
logs/alfworld/GMemoryContextEfficientAgent/test_H2_goal_contract_gate_C1_.../
H2 vs H1/H0 汇总表
drop reason distribution
better/worse flip analysis
spot-check notes
```

## 8. 实施顺序

建议顺序：

```text
1. 实现 config parsing 和 disabled backward compatibility。
2. 实现 parser/split/reconstruct。
3. 实现 risk rules。
4. 接入 reset retrieve 流程。
5. 实现 get_diagnostics。
6. 新增 H2 config。
7. 跑 py_compile 和离线检查。
8. 服务器环境下跑 small smoke。
9. 服务器环境下跑 full-134。
10. 做 H2 vs H1/H0 汇总。
```

small smoke 不作为最终结论，只用于发现：

```text
服务连接问题
prompt reconstruction 格式问题
diagnostics 缺失
gate 全 drop 或全 keep 的明显异常
```
