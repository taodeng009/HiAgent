# 2026-06-28 Stage 3 Phase 2.3：Task-Type-Aware Memory Router 实现计划

## 目标

在同一次 ALFWorld full-134 run 中，以 GateV3 task-start persistent memory 为默认策略，并按 task type 应用已经验证的少量 override：

```yaml
task_type_policy:
  default_profile: GATEV3_PERSISTENT
  routes:
    place: MEMORY_OFF
    clean: NOTSV_STALE6
    heat: GATEV3_PERSISTENT
    cool: GATEV3_PERSISTENT
    puttwo: NOTSV_STALE6
    look: MEMORY_OFF
```

`heat` 与 `cool` 使用 default `GATEV3_PERSISTENT`。

## 与既有计划的关系

本计划是 Stage 3 总实现计划的 Phase 2.3 实现补充：

```text
docs/stage3_need_aware_memory_intervention/plans/20260622_need_aware_delayed_memory_injection_implementation_plan.md
```

原计划已经实现并验证：

```text
Phase 0：retrieve/cache 与 visible prompt 状态拆分
Phase 1/1.1：delayed injection、TTL、cooldown 与 stuck trigger
Phase 2：intervention diagnostics 与分析脚本
Phase 2.1：task-start short TTL + stuck reactivation
Phase 2.2：ineffective reactivation suppression 设计
```

本计划不重写上述机制，只在 episode reset 时增加一层：

```text
task type -> effective intervention profile
```

策略选择依据来自：

```text
docs/stage3_need_aware_memory_intervention/plans/20260627_task_type_aware_memory_intervention_validation_plan.md
```

该验证计划已经完成 cool/look 定向实验并冻结 routing。本实现计划负责把离线拼接策略变成一个真实、可回退、可诊断的 full-134 运行入口。

本阶段不涉及：

```text
GateV3 content filter 规则修改
GMemory memory construction / update 修改
stuck-time retrieval 或 subgoal retrieval
新的 TTL / stale threshold ablation
online credit assignment
```

## 当前代码基础与缺口

已有基础：

```text
1. ALFWorld 在 agent.reset() 前调用 set_current_task_type(task_type)。
2. GMemoryContextEfficientAgent 已支持 GateV3 persistent、delayed injection 和 task-start TTL mode。
3. cached/visible prompt、TTL、cooldown 和 injection diagnostics 已存在。
```

当前缺口：

```text
1. need_aware_intervention.mode 是全局配置，不能按 task type 切换。
2. 缺少显式 MEMORY_OFF mode。
3. diagnostics 未记录 selected policy 与 effective config。
4. 必须防止上一 episode 的 resolved policy 泄漏到下一 episode。
5. current_task_type 是跨 episode 持久字段，缺失 setter 调用时可能沿用上一任务类型。
6. persistent mode 当前不会累计 interaction/exposure，分析时会错误显示 exposure=0。
7. 现有 CLI 不支持嵌套 YAML override，必须把 state config 增量扩展为可直接运行的 mixed full-134 入口。
```

## 配置设计

在现有 `gmemory.need_aware_intervention` 下增加可选 router；router 关闭时完全保持现有行为：

```yaml
gmemory:
  goal_contract_gate:
    # Whether GateV3 content filtering is enabled.
    # Options: true, false.
    # This switch is independent of memory scheduling/routing below.
    enabled: true

    # Gate/filter implementation used before any profile controls visibility.
    # Selected value for this experiment: per_insight_task_type_rule_v3.
    mode: per_insight_task_type_rule_v3

  need_aware_intervention:
    # Master switch for Stage 3 intervention support.
    # Options: true, false.
    # false preserves the existing legacy/global behavior and ignores task_type_policy.
    enabled: true

    task_type_policy:
      # Enable per-task-type profile resolution.
      # Options: true, false.
      # false uses the existing global need_aware_intervention.mode/options unchanged.
      enabled: true

      # Fallback profile for an absent, unknown, or newly introduced task type.
      # Options: any key declared under profiles.
      # The frozen experiment uses GateV3 persistent memory as the safe/default path.
      default_profile: GATEV3_PERSISTENT

      # Explicit full mapping for the six ALFWorld task types.
      # Each value must name a profile declared below.
      # heat/cool equal the default but are listed for experiment readability.
      routes:
        place: MEMORY_OFF
        clean: NOTSV_STALE6
        heat: GATEV3_PERSISTENT
        cool: GATEV3_PERSISTENT
        puttwo: NOTSV_STALE6
        look: MEMORY_OFF

      profiles:
        GATEV3_PERSISTENT:
          # Memory is retrieved, GateV3-filtered, and visible from task start
          # for the whole episode, matching the existing GateV3 reuse behavior.
          mode: gatev3_persistent

        MEMORY_OFF:
          # Memory may still be retrieved/filtered/cached for diagnostics, but
          # is never visible in the prompt and never reactivated after stuck.
          mode: memory_off

        NOTSV_STALE6:
          # No task-start visibility. Cached GateV3 memory becomes visible only
          # after the stuck trigger and is then bounded by TTL/cooldown.
          mode: delayed_task_level_memory_injection

          # Minimum completed environment steps without progress before stuck.
          # Options: non-negative integer. Frozen value: 6.
          stale_steps_since_last_progress: 6

          # Default visibility window for delayed injection.
          # Options: positive integer counted in completed env.step actions.
          visibility_ttl: 3

          # Visibility window specifically for stuck reactivation.
          # Options: positive integer; defaults to visibility_ttl when omitted.
          stuck_visibility_ttl: 3

          # Steps blocking another injection after memory is cleared.
          # Options: non-negative integer; 0 disables cooldown.
          reentry_cooldown_after_clear: 3
```

配置约束：

```text
1. need_aware_intervention.enabled=true 是启用 router 的前置条件；否则完全忽略 task_type_policy。
2. base config 与 profile 必须使用 deepcopy 合成，禁止原地修改 YAML 解析得到的配置对象。
3. profile 只覆盖显式声明的字段，其余 trigger/suppression 字段继承 base config。
4. mixed-policy state config 必须完整冻结所有继承字段，确保 NOTSV_STALE6 与离线 composition 的来源 run 同口径。
5. task type 在解析前统一执行 strip().lower()；原始值和 normalized 值分别保留在 diagnostics 中。
```

配置语义：

```text
GATEV3_PERSISTENT：复用现有 router-disabled GateV3 task-start visible 行为。
MEMORY_OFF：允许 retrieve/filter/cache 供 diagnostics 使用，但 memory 永不进入 prompt，也不进行 stuck reactivation。
NOTSV_STALE6：复用已有 delayed_task_level_memory_injection，task-start 不可见，stale=6 后 TTL=3。
unknown task type：回退 default_profile，不允许静默回退到 MEMORY_OFF。
```

profile `mode` 的 router 可选值及含义：

| mode | task-start visible | stuck reactivation | 用途 |
|---|---|---|---|
| `gatev3_persistent` | 是，持续可见 | 不适用 | 复用 GateV3 default |
| `memory_off` | 否 | 否 | 完全禁止 prompt memory |
| `delayed_task_level_memory_injection` | 否 | 是 | NOTSV / stuck-only |
| `task_start_ttl_then_stuck_reactivation` | 是，短 TTL | 是 | TSV；本次冻结 routing 未使用 |

不支持的 mode 必须在 reset 时明确报错，不能按 `gatev3_persistent` 或 `memory_off` 猜测回退。

第一版保留 reset-time retrieve，避免同时改变 retrieval cost/timing。`MEMORY_OFF` 的 retrieve 可在后续独立优化中关闭，但不能混入本次行为验收。

## 实现步骤

### Step 1：Effective Policy Resolver

在 `GMemoryContextEfficientAgent` 中：

```text
1. 保存不可变的 base need_aware_intervention config。
2. 每次 reset 在任何 GMemory/client early return 和 retrieve try/except 之前，根据本 episode task type 解析 selected profile。
3. 用 base config + profile override 生成本 episode 的 effective config。
4. 所有 intervention helper 读取 effective config，而不是直接修改 base config。
5. 每次 reset 重新解析，禁止跨 episode 复用上一个 effective config。
6. policy/config 错误在 retrieve 异常边界之外抛出，不能被记录成普通 retrieve failure。
```

建议新增接口：

```python
_resolve_task_type_intervention_policy(task_type) -> dict
_activate_effective_intervention_policy(task_type) -> None
```

无效 profile、缺失 profile 或不支持的 mode 应在 reset 时明确报错，不应静默使用错误策略。

reset 顺序必须固定为：

```text
super().reset(...)
-> 清空上一 episode intervention runtime
-> 消费并规范化本 episode task type
-> resolve / validate / activate effective policy
-> 初始化包含 policy metadata 的 diagnostics
-> 检查 gmemory enabled / recall / client 并允许 early return
-> 仅对 retrieve / filter / cache 执行 try/except
```

这保证 retrieve 失败、空 memory、client 缺失时仍有 selected policy diagnostics，同时 invalid profile 会终止运行。

#### Task Type 生命周期

当前 ALFWorld 正常路径会在每次 `agent.reset()` 前调用 `set_current_task_type(task_type)`，正式 full-134 可以继续复用该路径。但 router 不能把持久的 `current_task_type` 直接视为本 episode 一定已设置的值。

第一版采用一次性 pending 值：

```text
set_current_task_type(task_type) 写入 pending_task_type；
reset 时读取并消费 pending_task_type；
未设置、None 或空字符串均按 missing task type 处理并选择 default_profile；
另存 active_task_type 供本 episode GateV3 contract/filter 与 diagnostics 使用；
下一次 reset 不得沿用上一 episode 的 active_task_type 进行 routing。
```

为兼容现有 GateV3 contract，可让 `current_task_type` 在 episode 内指向 normalized active task type，但 router 的下一次输入必须来自新的 pending 值。若实现中选择把 `task_type` 显式加入 `reset()` 参数，也必须保持上述 missing/fallback 语义，并相应修改 ALFWorld 调用点。

### Step 2：MEMORY_OFF Mode

新增显式判断：

```python
_memory_off_intervention_enabled() -> bool
```

行为：

```text
retrieve/filter 可以执行；
final memory 写入 cached_gmemory_prompt；
visible_gmemory_prompt 始终为空；
不创建 task-start injection event；
不响应 stuck trigger；
make_prompt() 不出现 memory heading。
```

这保证 place/look 的 prompt 行为与 HiAgent memory-off 对齐，同时保留统一 diagnostics schema。

#### 所有 profile 的 exposure 口径

`update_intervention_state()` 必须对所有 profile 统一累计：

```text
gmemory_interaction_steps
gmemory_exposure_steps_total（以该 env.step 执行前 memory 是否可见为准）
```

只有 TTL、cooldown、stuck trigger 和 reactivation 分支受 `_need_aware_prompt_scheduling_enabled()` 控制。不得在累计 exposure 之前因 persistent/memory-off mode 提前返回。

因此：

```text
GATEV3_PERSISTENT：每个实际看到 memory 的 interaction step 计为 exposure；
MEMORY_OFF：interaction 正常累计，exposure 始终为 0；
NOTSV_STALE6：仅 stuck injection 可见窗口计为 exposure。
```

persistent reset-time event 的 `phase` 继续使用 `task_start_persistent`，但分析脚本必须将 `task_start` 与 `task_start_persistent` 都识别为 task-start exposure/event；也可以统一事件 schema，但必须保持旧日志兼容。

### Step 3：Diagnostics

在 `gmemory_intervention` 中新增：

```text
task_type_policy_enabled
raw_task_type
detected_task_type                  # normalized active task type
selected_intervention_policy
default_intervention_policy
route_matched
selected_profile_differs_from_default
policy_resolution_source            # explicit_route / default_unknown / default_missing / router_disabled
effective_mode
effective_stale_steps_since_last_progress
effective_visibility_ttl
effective_stuck_visibility_ttl
effective_task_start_visibility_ttl
effective_reentry_cooldown_after_clear
effective_reentry_cooldown_after_task_start_clear
effective_failure_observation_threshold
effective_require_check_valid_actions_since_progress
effective_failure_signal_policy
effective_suppress_ineffective_reactivation
effective_reactivation_effect_window
effective_max_consecutive_ineffective_reactivations
effective_allow_late_reactivation_after_new_signal
effective_config                       # 清洗后的行为字段快照，或等价的 stable config hash
```

`selected_intervention_policy` 必须出现在每个 episode 的最终日志中，包括 retrieve 失败或 memory 为空的 episode。

`route_matched` 表示 routes 中存在该 normalized task type；`selected_profile_differs_from_default` 表示最终 profile 名称与 default 不同。这样 heat/cool 虽然显式列在 routes 中，仍可表达为 `route_matched=true`、`selected_profile_differs_from_default=false`，避免 `policy_was_override` 歧义。

### Step 4：Config 与分析脚本

直接在现有配置中增量加入 router schema，并将其作为 mixed-policy full-134 标准入口：

```text
eval_configs/hiagent/alfworld_h2_goal_contract_gate_c1_v3_state.yaml
```

保留现有全局 `need_aware_intervention.mode` 和 trigger/suppression 参数作为 base config；在其下新增并启用 `task_type_policy`。关闭 `task_type_policy.enabled` 时仍回退到该全局配置，不删除旧能力。

该配置需要同步修改：

```text
num_exam: 134
删除 start_index / end_index（推荐）；或显式设置 start_index: 0、end_index: 133
不设置 target_task_types / target_task_limits
upload_on_finish: false
启用 task_type_policy
完整写入冻结 routes、profiles 以及所有继承的 trigger/suppression 参数
使用与 GateV3 / NOTSV_STALE6 对齐的 model、split、GMemory snapshot 和 GateV3 filter
使用独立 log_path / project_name，禁止覆盖已有 baseline 日志
```

`end_index` 是闭区间，因此不能设置为 134；`0..134` 会尝试运行 135 个任务。为减少边界错误，本计划优先采用 `num_exam: 134` 且省略 `start_index/end_index`。

这是对现有 YAML 的增量扩展，不新增第二份配置。原 80-109 参数可保留为注释，便于人工恢复 targeted run；真正回退行为由 `task_type_policy.enabled=false` 保证。

分析脚本增加按 `selected_intervention_policy` 聚合：

```text
episode count
SR / PR
injection episode/event count
memory exposure
success_up / success_down（与对齐 baseline 比较时）
```

分析脚本 CLI 增加可选 baseline 输入，例如：

```text
--baseline-jsonl <HiAgent alfworld.jsonl>
```

并要求：

```text
1. 输出 overall、by_task_type、by_selected_intervention_policy 三层 SR / PR / exposure / injection 指标。
2. 使用全局 episode id + task_name 与 baseline 做一对一对齐。
3. baseline 或 mixed 日志存在重复、缺失、task_name 不一致时明确报错，不计算 success_up/down。
4. 报告 selected policy 缺失数、resolution source 分布和实际 route count。
5. 将 phase=task_start 与 phase=task_start_persistent 都计入 task-start event；保留旧日志兼容。
```

## 计划改动文件

```text
agentboard/agents/gmemory_agent.py
scripts/check_goal_contract_gate.py
scripts/analyze_stage3_need_aware_intervention.py
eval_configs/hiagent/alfworld_h2_goal_contract_gate_c1_v3_state.yaml
docs/stage3_need_aware_memory_intervention/plans/20260627_task_type_aware_memory_intervention_validation_plan.md
docs/stage3_need_aware_memory_intervention/plans/20260628_task_type_aware_memory_router_implementation_plan.md
```

预计无需修改：

```text
agentboard/tasks/alfworld.py
```

原因是它已经在 `agent.reset()` 前设置 `current_task_type`。如果采用上述 pending-task-type 方案，无需修改该文件；如果最终选择显式 `reset(..., task_type=...)` 接口，则必须同步修改该文件，不能再将它列为“无需修改”。

## Deterministic Checks

在 `scripts/check_goal_contract_gate.py` 增加：

```text
1. router disabled：行为与当前全局 GateV3 / NOTSV 配置一致。
2. cool / heat：选择 GATEV3_PERSISTENT，首个 prompt 包含 memory heading。
3. place / look：选择 MEMORY_OFF，cached 可存在但所有 prompt 均无 memory heading。
4. clean / puttwo：选择 NOTSV_STALE6，task-start prompt 无 memory；stale=6 trigger 后短期可见。
5. unknown task type：选择 default GATEV3_PERSISTENT。
6. invalid profile：明确失败，不静默 fallback。
7. 连续执行 place -> clean -> cool：三个 episode 的 effective config 不串联。
8. diagnostics 正确记录 task type、selected profile、effective mode 与 effective threshold。
9. MEMORY_OFF 不生成 injection event，也不响应人工构造的 stuck trigger。
10. GateV3 content filter diagnostics 在所有 profile 下保持存在。
11. known task type episode 后直接 reset、但不调用 setter：必须选择 default，不能沿用上一 episode route。
12. set_current_task_type(None) 与空字符串：选择 default，resolution source 为 default_missing。
13. task type 大小写与首尾空格规范化后正确命中 route。
14. retrieve failure、空 memory、gmemory client 缺失：仍记录 selected profile 与完整 effective diagnostics。
15. invalid profile / unsupported mode 的错误不能被 reset 中的 retrieve exception handler 吞掉。
16. GATEV3_PERSISTENT exposure 随可见 interaction steps 累计；MEMORY_OFF exposure 恒为 0；NOTSV 只累计 stuck visibility window。
17. persistent event 被分析脚本计入 task-start event，且旧日志仍可分析。
18. analyzer 按 selected profile 输出 SR/PR，并在 baseline 对齐缺失、重复或错配时明确失败。
19. 修改后的 state YAML 解析后确认 num_exam=134、无 range/target filter、router enabled、upload disabled。
```

本地验收命令：

```powershell
python -m py_compile agentboard/agents/gmemory_agent.py scripts/check_goal_contract_gate.py scripts/analyze_stage3_need_aware_intervention.py
python scripts/check_goal_contract_gate.py
```

若服务器标准入口使用 `py`，可等价替换；当前本地 Windows 环境应使用可解析到的 `python` 解释器。

## 服务器验证

本地 deterministic checks 通过后，只运行一组 mixed-policy full-134：

```text
default: GATEV3_PERSISTENT
place/look: MEMORY_OFF
clean/puttwo: NOTSV_STALE6
heat/cool: GATEV3_PERSISTENT
upload_on_finish: false
same model / split / GMemory snapshot / GateV3 filter
```

启动服务器 run 前增加 preflight：

```text
1. 从服务器实际 ALFWorld split/label 统计六类任务数，不只依赖文档中的历史计数。
2. 确认总数为 134，且 place/clean/heat/cool/puttwo/look 分别为 24/31/23/21/17/18。
3. dry-run 解析修改后的 state YAML，打印 default profile、六条 route 和完整 effective profile 参数。
4. 确认输出目录为空或为新的独立目录，避免混入已有 episode。
```

必须检查每种类型的实际路由数量：

```text
GATEV3_PERSISTENT: heat 23 + cool 21 = 44
MEMORY_OFF: place 24 + look 18 = 42
NOTSV_STALE6: clean 31 + puttwo 17 = 48
total: 134
```

## 验收条件

离线 composition 参考：

```text
SR = 0.4403
PR = 0.6250
GA = 0.8111
success_up vs HiAgent = 10
success_down vs HiAgent = 4
```

真实 full-134 的主要验收条件：

```text
1. 134 个 episode 全部记录 selected_intervention_policy，路由计数为 44 / 42 / 48。
2. place/look 的 memory exposure 与 injection event 均为 0。
3. clean/puttwo task-start exposure 为 0，只允许 stuck-time injection。
4. heat/cool 保持 GateV3 task-start visible 行为。
5. overall SR 不低于 GateV3 0.4030，PR 不低于 GateV3 0.5989。
6. success_down relative to HiAgent 低于 GateV3 的 13，目标不高于 NOTSV_STALE6 的 5。
7. 若结果明显偏离离线 composition，先检查路由、prompt visibility 与配置泄漏，不立即增加新 ablation。
8. persistent、memory-off、NOTSV 三种 profile 的 exposure 口径均通过 deterministic check，不能出现 persistent prompt 可见但 exposure=0。
9. mixed 日志与 HiAgent baseline 以 134 个全局 id + task_name 完整一对一对齐后，才报告 success_up / success_down。
```

离线 composition 来自不同已有 run 的逐任务拼接，是实现验收参考，不应被表述为真实 mixed-policy 实验结果。

## 回退条件

```text
task_type_policy.enabled=false
```

必须完全恢复现有单一全局 intervention 配置。若 router 启用导致未知 task type、retrieve failure 或空 memory 时行为不确定，则停止 full-134，先修复 resolver 与 diagnostics。

## 完成条件

```text
1. router、MEMORY_OFF 与 diagnostics 实现完成；
2. deterministic checks 全部通过；
3. mixed-policy full-134 完成；
4. 输出真实结果报告，并与离线 composition、GateV3、HiAgent、NOTSV_STALE6 对比；
5. 验收通过后，将该 routing 记为 Stage 3 当前默认候选，而不是立即替换通用 GateV3 默认配置。
```
