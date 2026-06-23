"""Local smoke checks for the H2 goal-contract memory gate.

This script does not contact GMemory, a model server, or ALFWorld. It only
validates the deterministic gate helpers on GMemoryContextEfficientAgent.
"""
from __future__ import annotations

import json
import importlib.util
import os
import sys
import types


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTBOARD_ROOT = os.path.join(PROJECT_ROOT, "agentboard")
if AGENTBOARD_ROOT not in sys.path:
    sys.path.insert(0, AGENTBOARD_ROOT)


def load_gmemory_agent_class():
    agents_pkg = types.ModuleType("agents")
    agents_pkg.__path__ = [os.path.join(AGENTBOARD_ROOT, "agents")]
    sys.modules.setdefault("agents", agents_pkg)
    module_path = os.path.join(AGENTBOARD_ROOT, "agents", "gmemory_agent.py")
    spec = importlib.util.spec_from_file_location("agents.gmemory_agent", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["agents.gmemory_agent"] = module
    spec.loader.exec_module(module)
    return module.GMemoryContextEfficientAgent


def load_stage3_analysis_module():
    module_path = os.path.join(PROJECT_ROOT, "scripts", "analyze_stage3_need_aware_intervention.py")
    spec = importlib.util.spec_from_file_location("analyze_stage3_need_aware_intervention", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_stage3_need_aware_intervention"] = module
    spec.loader.exec_module(module)
    return module


GMemoryContextEfficientAgent = load_gmemory_agent_class()
stage3_analysis = load_stage3_analysis_module()


class FakeLLM:
    engine = "fake"
    context_length = 32768
    max_tokens = 512

    def num_tokens_from_messages(self, messages):
        return sum(len(message.get("content", "")) for message in messages)


class FakeGMemoryClient:
    def __init__(self, memory_prompt):
        self.memory_prompt = memory_prompt

    def retrieve(self, **kwargs):
        return {"memory_prompt": self.memory_prompt}


def make_agent(
    max_context_chars=5000,
    mode="per_insight_rule_v2",
    diagnostics_only=False,
    gate_overrides=None,
    need_aware_intervention=None,
):
    goal_contract_gate = {
        "enabled": True,
        "mode": mode,
        "split_mode": "c1_insight_lines",
        "min_kept_insights": 1,
        "max_kept_insights": 3,
        "diagnostics_only": diagnostics_only,
    }
    if gate_overrides:
        goal_contract_gate.update(gate_overrides)
    return GMemoryContextEfficientAgent(
        llm_model=FakeLLM(),
        need_goal=True,
        gmemory={
            "enabled": True,
            "base_url": "http://127.0.0.1:8090",
            "memory_only": True,
            "max_context_chars": max_context_chars,
            "goal_contract_gate": goal_contract_gate,
            "need_aware_intervention": need_aware_intervention or {},
        },
    )


def make_gate_disabled_agent(max_context_chars=80):
    return GMemoryContextEfficientAgent(
        llm_model=FakeLLM(),
        need_goal=True,
        gmemory={
            "enabled": True,
            "base_url": "http://127.0.0.1:8090",
            "memory_only": True,
            "max_context_chars": max_context_chars,
        },
    )


def check_disabled_path_uses_legacy_filter():
    agent = make_gate_disabled_agent(max_context_chars=90)
    raw_prompt = """Intro text that should be removed when memory_only is true.

## Key Insights from Related Tasks
1. This is a useful insight that should be truncated by the legacy filter when the gate is disabled.
2. This second insight should not be reached with a small context limit.

## Other Section
Ignore this section.
"""
    legacy_prompt = agent._filter_gmemory_prompt(raw_prompt)
    assert not agent._goal_contract_gate_enabled()
    assert legacy_prompt.startswith("## Key Insights from Related Tasks")
    assert len(legacy_prompt) <= 90
    print("PASS disabled path legacy filter")


def check_phase0_intervention_diagnostics_and_compat_prompt_state():
    agent = make_agent(need_aware_intervention={"enabled": False, "mode": "disabled"})
    prompt = "## Key Insights from Related Tasks\n1. Pick up the object before placing it."
    agent._set_gmemory_prompt_state(cached_prompt=prompt, visible_prompt=prompt, retrieve_step=0, reset_events=True)
    diagnostics = agent.get_diagnostics()
    intervention = diagnostics["gmemory_intervention"]
    assert agent.cached_gmemory_prompt == prompt
    assert agent.visible_gmemory_prompt == prompt
    assert agent.gmemory_prompt == prompt
    assert diagnostics["gmemory_prompt_chars"] == len(prompt)
    assert diagnostics["memory_injected_to_prompt"] is True
    assert intervention["enabled"] is False
    assert intervention["mode"] == "disabled"
    assert intervention["retrieved"] is True
    assert intervention["cached"] is True
    assert intervention["injected"] is True
    assert intervention["visible"] is True
    assert intervention["retrieved_but_not_injected"] is False
    assert intervention["retrieve_step"] == 0
    assert intervention["injection_events"][0]["reason"] == "reset_time_immediate"

    agent._set_gmemory_prompt_state(cached_prompt=prompt, visible_prompt="", retrieve_step=2, reset_events=True)
    diagnostics = agent.get_diagnostics()
    intervention = diagnostics["gmemory_intervention"]
    assert agent.cached_gmemory_prompt == prompt
    assert agent.visible_gmemory_prompt == ""
    assert agent.gmemory_prompt == ""
    assert diagnostics["memory_injected_to_prompt"] is False
    assert intervention["retrieved_but_not_injected"] is True
    assert intervention["injection_events"] == []
    json.dumps(diagnostics)
    print("PASS phase0 intervention diagnostics and compat prompt state")


def make_delayed_agent(**overrides):
    config = {
        "enabled": True,
        "mode": "delayed_task_level_memory_injection",
        "visibility_ttl": 2,
        "reentry_cooldown_after_clear": 2,
        "stale_steps_since_last_progress": 2,
        "failure_observation_threshold": 2,
        "require_check_valid_actions_since_progress": True,
        "failure_signal_policy": "nothing_happens_or_query_action_loop",
        "query_action_loop_count_threshold": 2,
        "query_action_loop_ratio_threshold": 0.6,
        "refresh_ttl_on_progress": False,
        "clear_on_progress": False,
    }
    config.update(overrides)
    return make_agent(mode="per_insight_task_type_rule_v3", need_aware_intervention=config)


def make_task_start_ttl_agent(**overrides):
    config = {
        "enabled": True,
        "mode": "task_start_ttl_then_stuck_reactivation",
        "task_start_visibility_ttl": 2,
        "visibility_ttl": 2,
        "stuck_visibility_ttl": 2,
        "reentry_cooldown_after_task_start_clear": 2,
        "reentry_cooldown_after_clear": 2,
        "stale_steps_since_last_progress": 2,
        "failure_observation_threshold": 2,
        "require_check_valid_actions_since_progress": True,
        "failure_signal_policy": "nothing_happens_or_query_action_loop",
        "query_action_loop_count_threshold": 2,
        "query_action_loop_ratio_threshold": 0.6,
        "refresh_ttl_on_progress": False,
        "clear_on_progress": False,
    }
    config.update(overrides)
    return make_agent(mode="per_insight_task_type_rule_v3", need_aware_intervention=config)


def drive_no_progress_step(agent, step, action="go to desk 1", observation="Nothing happens."):
    agent.update_intervention_state(
        step_id=step,
        executed_action=action,
        observation=observation,
        progress_rate=0.0,
        previous_progress_rate=0.0,
        is_valid_action=True,
        nothing_happens=(observation.strip() == "Nothing happens."),
        is_check_valid_actions=(action == "check valid actions"),
    )


def check_phase1_delayed_reset_and_prompt_visibility():
    os.environ.setdefault("EVALTASK", "alfworld")
    agent = make_delayed_agent()
    agent.set_current_task_type("place")
    agent.gmemory_client = FakeGMemoryClient(
        "## Key Insights from Related Tasks\n"
        "1. Find the plate, pick it up, and put it on the countertop."
    )
    agent.reset(goal="put a plate in countertop.", init_obs="You are in the middle of a room.")
    assert agent.cached_gmemory_prompt.startswith("## Key Insights from Related Tasks")
    assert agent.visible_gmemory_prompt == ""
    assert agent.gmemory_prompt == ""
    diagnostics = agent.get_diagnostics()
    assert diagnostics["memory_injected_to_prompt"] is False
    assert diagnostics["gmemory_intervention"]["retrieved_but_not_injected"] is True

    prompt = agent.make_prompt(need_goal=True, check_actions="check valid actions", check_inventory="inventory")
    assert "## Key Insights from Related Tasks" not in prompt
    print("PASS phase1 delayed reset and prompt visibility")


def check_phase1_trigger_ttl_cooldown_and_progress_delta():
    agent = make_delayed_agent()
    memory_prompt = "## Key Insights from Related Tasks\n1. Use cached guidance only when stuck."
    agent._set_gmemory_prompt_state(cached_prompt=memory_prompt, visible_prompt="", retrieve_step=0, reset_events=True)

    drive_no_progress_step(agent, 0)
    assert agent.visible_gmemory_prompt == ""
    assert agent.get_diagnostics()["gmemory_intervention"]["last_decision"]["trigger_condition_satisfied"] is False

    drive_no_progress_step(agent, 1, action="check valid actions")
    diagnostics = agent.get_diagnostics()["gmemory_intervention"]
    assert agent.visible_gmemory_prompt == memory_prompt
    assert diagnostics["last_decision"]["trigger_condition_satisfied"] is True
    assert diagnostics["last_decision"]["intervention_allowed"] is True
    assert diagnostics["last_decision"]["skip_reason"] == "none"
    assert diagnostics["current_visible_ttl_remaining"] == 2
    assert diagnostics["injection_events"][-1]["step"] == 1

    agent.update_intervention_state(
        step_id=2,
        executed_action="go to desk 1",
        observation="Nothing happens.",
        progress_rate=0.25,
        previous_progress_rate=0.0,
        is_valid_action=True,
        nothing_happens=True,
        is_check_valid_actions=False,
    )
    diagnostics = agent.get_diagnostics()["gmemory_intervention"]
    assert diagnostics["stale_steps_since_last_progress"] == 0
    assert diagnostics["current_visible_ttl_remaining"] == 1
    assert diagnostics["injection_events"][-1]["post_injection_progress_delta"] == 0.25
    assert diagnostics["last_decision"]["skip_reason"] == "memory_visible"

    agent.update_intervention_state(
        step_id=3,
        executed_action="go to desk 1",
        observation="Nothing happens.",
        progress_rate=0.25,
        previous_progress_rate=0.25,
        is_valid_action=True,
        nothing_happens=True,
        is_check_valid_actions=False,
    )
    diagnostics = agent.get_diagnostics()["gmemory_intervention"]
    assert agent.visible_gmemory_prompt == ""
    assert diagnostics["current_visible_ttl_remaining"] is None
    assert diagnostics["current_cooldown_remaining"] == 2
    assert diagnostics["clear_events"][-1]["reason"] == "ttl_expired"

    drive_no_progress_step(agent, 4, action="check valid actions")
    diagnostics = agent.get_diagnostics()["gmemory_intervention"]
    assert diagnostics["last_decision"]["would_trigger"] is True
    assert diagnostics["last_decision"]["skip_reason"] == "reentry_cooldown"
    assert diagnostics["current_cooldown_remaining"] == 1
    print("PASS phase1 trigger ttl cooldown and progress delta")


def check_phase1_trigger_skip_reasons():
    no_cached_agent = make_delayed_agent(stale_steps_since_last_progress=1, failure_observation_threshold=1)
    drive_no_progress_step(no_cached_agent, 0, action="check valid actions")
    diagnostics = no_cached_agent.get_diagnostics()["gmemory_intervention"]
    assert diagnostics["last_decision"]["trigger_condition_satisfied"] is True
    assert diagnostics["last_decision"]["skip_reason"] == "no_cached_memory"

    no_usable_agent = make_delayed_agent(stale_steps_since_last_progress=1, failure_observation_threshold=1)
    no_usable_agent._set_gmemory_prompt_state(cached_prompt="", visible_prompt="", retrieve_step=0, reset_events=True)
    drive_no_progress_step(no_usable_agent, 0, action="check valid actions")
    diagnostics = no_usable_agent.get_diagnostics()["gmemory_intervention"]
    assert diagnostics["last_decision"]["trigger_condition_satisfied"] is True
    assert diagnostics["last_decision"]["skip_reason"] == "no_usable_memory"
    print("PASS phase1 trigger skip reasons")


def check_phase11_query_action_loop_trigger():
    memory_prompt = "## Key Insights from Related Tasks\n1. Query-loop recovery guidance."
    agent = make_delayed_agent(stale_steps_since_last_progress=2)
    agent._set_gmemory_prompt_state(cached_prompt=memory_prompt, visible_prompt="", retrieve_step=0, reset_events=True)

    drive_no_progress_step(agent, 0, action="check valid actions", observation="Choose an action from these valid actions: inventory")
    drive_no_progress_step(agent, 1, action="inventory", observation="You are not carrying anything.")
    diagnostics = agent.get_diagnostics()["gmemory_intervention"]
    assert diagnostics["trigger_reason"] == "query_action_loop"
    assert diagnostics["query_action_count_since_last_progress"] == 2
    assert diagnostics["inventory_count_since_last_progress"] == 1
    assert diagnostics["query_action_ratio_since_last_progress"] == 1.0
    assert diagnostics["last_decision"]["trigger_condition_satisfied"] is True
    assert diagnostics["last_decision"]["trigger_reason"] == "query_action_loop"
    assert diagnostics["injection_events"][-1]["trigger_reason"] == "query_action_loop"
    assert agent.visible_gmemory_prompt == memory_prompt

    rollback_agent = make_delayed_agent(
        failure_signal_policy="nothing_happens_and_check",
        stale_steps_since_last_progress=2,
    )
    rollback_agent._set_gmemory_prompt_state(cached_prompt=memory_prompt, visible_prompt="", retrieve_step=0, reset_events=True)
    drive_no_progress_step(rollback_agent, 0, action="check valid actions", observation="Choose an action from these valid actions: inventory")
    drive_no_progress_step(rollback_agent, 1, action="inventory", observation="You are not carrying anything.")
    rollback_diag = rollback_agent.get_diagnostics()["gmemory_intervention"]
    assert rollback_diag["query_action_loop_trigger"] is True
    assert rollback_diag["last_decision"]["trigger_condition_satisfied"] is False
    assert rollback_agent.visible_gmemory_prompt == ""

    ratio_agent = make_delayed_agent(
        stale_steps_since_last_progress=4,
        query_action_loop_count_threshold=2,
        query_action_loop_ratio_threshold=0.75,
    )
    ratio_agent._set_gmemory_prompt_state(cached_prompt=memory_prompt, visible_prompt="", retrieve_step=0, reset_events=True)
    drive_no_progress_step(ratio_agent, 0, action="go to fridge 1", observation="The fridge 1 is closed.")
    drive_no_progress_step(ratio_agent, 1, action="check valid actions", observation="Choose an action from these valid actions: inventory")
    drive_no_progress_step(ratio_agent, 2, action="go to fridge 1", observation="The fridge 1 is closed.")
    drive_no_progress_step(ratio_agent, 3, action="inventory", observation="You are not carrying anything.")
    ratio_diag = ratio_agent.get_diagnostics()["gmemory_intervention"]
    assert ratio_diag["query_action_count_since_last_progress"] == 2
    assert ratio_diag["query_action_ratio_since_last_progress"] == 0.5
    assert ratio_diag["query_action_loop_trigger"] is False
    assert ratio_diag["last_decision"]["trigger_condition_satisfied"] is False
    print("PASS phase1.1 query action loop trigger")


def check_phase21_task_start_ttl_then_stuck_reactivation():
    os.environ.setdefault("EVALTASK", "alfworld")
    memory_prompt = (
        "## Key Insights from Related Tasks\n"
        "1. Find the plate, pick it up, and put it on the countertop."
    )
    agent = make_task_start_ttl_agent()
    agent.set_current_task_type("place")
    agent.gmemory_client = FakeGMemoryClient(memory_prompt)
    agent.reset(goal="put a plate in countertop.", init_obs="You are in the middle of a room.")

    diagnostics = agent.get_diagnostics()["gmemory_intervention"]
    assert agent.cached_gmemory_prompt.startswith("## Key Insights from Related Tasks")
    assert agent.visible_gmemory_prompt == agent.cached_gmemory_prompt
    assert diagnostics["injection_events"][-1]["reason"] == "task_start_ttl"
    assert diagnostics["injection_events"][-1]["phase"] == "task_start"
    assert diagnostics["current_visible_ttl_remaining"] == 2
    assert diagnostics["metrics"]["task_start_injection_count"] == 1

    agent.update_intervention_state(
        step_id=0,
        executed_action="go to countertop 1",
        observation="On the countertop 1, you see a plate 1.",
        progress_rate=0.25,
        previous_progress_rate=0.0,
        is_valid_action=True,
        nothing_happens=False,
        is_check_valid_actions=False,
    )
    diagnostics = agent.get_diagnostics()["gmemory_intervention"]
    assert diagnostics["current_visible_ttl_remaining"] == 1
    assert diagnostics["metrics"]["post_task_start_progress_delta"] == 0.25
    assert diagnostics["metrics"]["memory_exposure_steps_total"] == 1

    agent.update_intervention_state(
        step_id=1,
        executed_action="take plate 1 from countertop 1",
        observation="You pick up the plate 1 from the countertop 1.",
        progress_rate=0.5,
        previous_progress_rate=0.25,
        is_valid_action=True,
        nothing_happens=False,
        is_check_valid_actions=False,
    )
    diagnostics = agent.get_diagnostics()["gmemory_intervention"]
    assert agent.visible_gmemory_prompt == ""
    assert diagnostics["clear_events"][-1]["reason"] == "task_start_ttl_expired"
    assert diagnostics["clear_events"][-1]["phase"] == "task_start"
    assert diagnostics["current_cooldown_remaining"] == 2
    assert diagnostics["metrics"]["task_start_ttl_expired_count"] == 1
    assert diagnostics["metrics"]["memory_exposure_steps_total"] == 2

    drive_no_progress_step(agent, 2, action="check valid actions")
    diagnostics = agent.get_diagnostics()["gmemory_intervention"]
    assert diagnostics["last_decision"]["skip_reason"] == "reentry_cooldown"
    assert diagnostics["last_decision"]["would_trigger"] is False
    assert agent.visible_gmemory_prompt == ""

    drive_no_progress_step(agent, 3, action="check valid actions")
    diagnostics = agent.get_diagnostics()["gmemory_intervention"]
    assert diagnostics["last_decision"]["skip_reason"] == "reentry_cooldown"
    assert diagnostics["last_decision"]["would_trigger"] is True
    assert diagnostics["current_cooldown_remaining"] == 0
    assert agent.visible_gmemory_prompt == ""

    drive_no_progress_step(agent, 4, action="check valid actions")
    diagnostics = agent.get_diagnostics()["gmemory_intervention"]
    assert agent.visible_gmemory_prompt == agent.cached_gmemory_prompt
    assert diagnostics["injection_events"][-1]["reason"] == "stuck_reactivation"
    assert diagnostics["injection_events"][-1]["phase"] == "stuck"
    assert diagnostics["metrics"]["stuck_reactivation_count"] == 1
    assert diagnostics["metrics"]["first_stuck_reactivation_step"] == 4
    print("PASS phase2.1 task-start ttl then stuck reactivation")


def check_phase21_no_usable_memory_skips_task_start_visibility():
    agent = make_task_start_ttl_agent()
    agent.set_current_task_type("puttwo")
    agent.gmemory_client = FakeGMemoryClient(
        "## Key Insights from Related Tasks\n"
        "1. Find one object and put it in the target, because this completes the immediate goal."
    )
    agent.reset(goal="put two cd in safe.", init_obs="You are in the middle of a room.")
    diagnostics = agent.get_diagnostics()["gmemory_intervention"]
    assert agent.cached_gmemory_prompt == ""
    assert agent.visible_gmemory_prompt == ""
    assert diagnostics["cached"] is False
    assert diagnostics["visible"] is False
    assert diagnostics["injection_events"] == []
    print("PASS phase2.1 no usable memory skips task-start visibility")


def check_phase2_stage3_analysis_metrics():
    fake_records = [
        {
            "id": 0,
            "task_name": "pick_clean_then_place_in_recep-Plate-None-CounterTop-10",
            "is_done": False,
            "progress_rate": 0.5,
            "agent_diagnostics": {
                "gmemory_intervention": {
                    "retrieved": True,
                    "cached": True,
                    "visible": False,
                    "retrieved_but_not_injected": False,
                    "injection_events": [
                        {
                            "step": 2,
                            "phase": "stuck",
                            "trigger_reason": "nothing_happens",
                            "post_injection_progress_delta": 0.5,
                            "delta_within_ttl": 0.25,
                            "delta_within_ttl_plus_3": 0.5,
                            "eventual_delta_after_injection": 0.5,
                        }
                    ],
                    "clear_events": [{"step": 4, "reason": "ttl_expired"}],
                    "metrics": {
                        "cooldown_block_count": 1,
                        "trigger_no_cached_memory_count": 0,
                        "trigger_no_usable_memory_count": 0,
                        "stuck_trigger_count": 2,
                    },
                },
                "alfworld_action_stats": {"check_valid_actions_count": 3, "nothing_happens_count": 2},
            },
        },
        {
            "id": 1,
            "task_name": "pick_cool_then_place_in_recep-Lettuce-None-CounterTop-10",
            "is_done": False,
            "progress_rate": 0.0,
            "agent_diagnostics": {
                "gmemory_intervention": {
                    "retrieved": True,
                    "cached": True,
                    "visible": False,
                    "retrieved_but_not_injected": True,
                    "injection_events": [],
                    "clear_events": [],
                    "metrics": {
                        "cooldown_block_count": 0,
                        "trigger_no_cached_memory_count": 0,
                        "trigger_no_usable_memory_count": 0,
                        "stuck_trigger_count": 0,
                    },
                },
                "alfworld_action_stats": {"check_valid_actions_count": 5, "nothing_happens_count": 0},
            },
        },
        {
            "id": 2,
            "task_name": "pick_and_place_simple-Plate-None-CounterTop-10",
            "is_done": True,
            "progress_rate": 1.0,
            "agent_diagnostics": {
                "gmemory_intervention": {
                    "retrieved": True,
                    "cached": True,
                    "visible": False,
                    "retrieved_but_not_injected": False,
                    "injection_events": [
                        {
                            "step": -1,
                            "phase": "task_start",
                            "reason": "task_start_ttl",
                            "post_injection_progress_delta": 1.0,
                            "delta_within_ttl": 0.5,
                            "delta_within_ttl_plus_3": 1.0,
                            "eventual_delta_after_injection": 1.0,
                        }
                    ],
                    "clear_events": [{"step": 1, "reason": "task_start_ttl_expired", "phase": "task_start"}],
                    "metrics": {
                        "cooldown_block_count": 0,
                        "trigger_no_cached_memory_count": 0,
                        "trigger_no_usable_memory_count": 0,
                        "stuck_trigger_count": 0,
                        "memory_exposure_steps_total": 2,
                        "memory_exposure_rate": 0.5,
                    },
                },
                "alfworld_action_stats": {"check_valid_actions_count": 0, "nothing_happens_count": 0},
            },
        },
    ]
    analysis = stage3_analysis.analyze_records(fake_records)
    overall = analysis["overall"]
    assert overall["episode_count"] == 3
    assert overall["injected_episode_count"] == 2
    assert overall["task_start_injected_episode_count"] == 1
    assert overall["stuck_reactivation_episode_count"] == 1
    assert overall["memory_injection_rate"] == 2 / 3
    assert overall["recovery_success_rate"] == 1.0
    assert overall["retrieved_but_not_visible_count"] == 1
    assert overall["avg_delta_within_ttl"] == 0.375
    assert overall["avg_post_task_start_progress_delta"] == 1.0
    assert overall["avg_post_stuck_reactivation_progress_delta"] == 0.5
    assert overall["task_start_ttl_expired_count"] == 1
    assert overall["memory_exposure_steps_total"] == 2
    assert overall["avg_memory_exposure_rate"] == 1 / 6
    assert analysis["by_task_type"]["clean"]["injected_episode_count"] == 1
    assert analysis["by_task_type"]["cool"]["injected_episode_count"] == 0
    assert analysis["by_task_type"]["place"]["task_start_injected_episode_count"] == 1
    print("PASS phase2 stage3 analysis metrics")


def check_goal_contract_parser():
    agent = make_agent()

    agent.set_current_task_type("clean")
    contract = agent._parse_goal_contract("put a clean plate in countertop.")
    assert contract["task_type"] == "clean"
    assert contract["object"] == "plate"
    assert contract["target"] == "countertop"
    assert contract["count"] == "one"
    assert contract["required_state"] == "clean"
    assert contract["final_action"] == "put"
    assert "count_constraint" not in contract
    assert "state_requirement" not in contract
    assert "target_object" not in contract
    assert "target_receptacle_or_tool" not in contract
    assert "needs_intermediate_state" not in contract
    assert "needs_finalization" not in contract
    assert "completion_pattern" not in contract

    agent.set_current_task_type("puttwo")
    contract = agent._parse_goal_contract("put two cd in safe.")
    assert contract["task_type"] == "puttwo"
    assert contract["object"] == "cd"
    assert contract["target"] == "safe"
    assert contract["count"] == "two"
    assert contract["required_state"] == "none"
    assert contract["final_action"] == "put"

    agent.set_current_task_type("look")
    contract = agent._parse_goal_contract("examine the alarmclock with the desklamp.")
    assert contract["task_type"] == "look"
    assert contract["object"] == "alarmclock"
    assert contract["target"] == "desklamp"
    assert contract["count"] == "one"
    assert contract["required_state"] == "none"
    assert contract["final_action"] == "examine"
    print("PASS goal contract parser")


def check_insight_split():
    agent = make_agent()
    prompt = """## Key Insights from Related Tasks
1. Find the object first, because search must precede placement.
2. Place it in the target receptacle to finish.
"""
    insights = agent._split_insights(prompt)
    assert len(insights) == 2
    assert insights[0].startswith("Find the object")
    assert insights[1].startswith("Place it")
    print("PASS insight split")


def check_instruction_preamble_filtered():
    agent = make_agent()
    agent.goal = "put a clean plate in countertop."
    prompt = """## Key Insights from Related Tasks
The following are insights gathered during the execution of similar tasks. You may refer to them during your task execution to improve problem-solving accuracy.

1. Confirm that the intended operation is suitable for the target object before executing, because applying an inappropriate action wastes steps and fails the task.
2. After an object has been processed (e.g., cleaned), confirm it is in the correct state **and that the processing action succeeded**, because proceeding without verification can leave the object unchanged and the task incomplete.
3. Open containers before attempting to retrieve items, because items inside are inaccessible while the container is closed.
---
"""
    insights = agent._split_insights(prompt)
    assert len(insights) == 3
    assert all("following are insights gathered" not in insight for insight in insights)
    final_prompt = agent._gate_gmemory_prompt_per_insight(prompt)
    diagnostics = agent.get_diagnostics()["gmemory_gate"]
    assert "following are insights gathered" in final_prompt
    assert final_prompt.rstrip().endswith("---")
    assert diagnostics["original_memory_prompt"] == prompt
    assert diagnostics["final_memory_prompt"] == final_prompt
    assert diagnostics["instruction_preamble_count"] == 1
    assert diagnostics["instruction_preamble_lines"] == [
        "The following are insights gathered during the execution of similar tasks. You may refer to them during your task execution to improve problem-solving accuracy."
    ]
    assert diagnostics["end_delimiter"] == "---"
    assert diagnostics["insight_count"] == 3
    print("PASS instruction preamble and delimiter preservation")


def check_cardinality_mismatch_gate():
    agent = make_agent()
    agent.goal = "put two cd in safe."
    agent.init_obs = "You are in the middle of a room."
    prompt = """## Key Insights from Related Tasks
1. Find one object and put it in the target, because this completes the immediate goal.
2. Repeat for the second object and count both placements.
"""
    final_prompt = agent._gate_gmemory_prompt_per_insight(prompt)
    diagnostics = agent.get_diagnostics()["gmemory_gate"]
    assert "Find one object" not in final_prompt
    assert "Repeat for the second object" in final_prompt
    assert diagnostics["kept_count"] == 1
    assert diagnostics["dropped_count"] == 1
    assert diagnostics["dropped_insights"][0]["reasons"] == ["cardinality_mismatch"]
    print("PASS cardinality mismatch gate")


def check_over_verification_gate():
    agent = make_agent()
    agent.goal = "put a plate in countertop."
    agent.init_obs = "You are in the middle of a room."
    prompt = """## Key Insights from Related Tasks
1. Check inventory, examine the object, verify the state, confirm the location, and check again because nothing happens can be ambiguous.
2. Put the object in the target receptacle to finish the task.
"""
    final_prompt = agent._gate_gmemory_prompt_per_insight(prompt)
    diagnostics = agent.get_diagnostics()["gmemory_gate"]
    assert "Check inventory" not in final_prompt
    assert "Put the object" in final_prompt
    assert diagnostics["kept_count"] == 1
    assert diagnostics["dropped_count"] == 1
    assert "over_verification_risk" in diagnostics["dropped_insights"][0]["reasons"]
    print("PASS over verification gate")


def check_v1_mode_still_uses_original_rule():
    agent = make_agent(mode="per_insight_rule_v1")
    contract = agent._parse_goal_contract("put a clean plate in countertop.")
    risk = agent._assess_goal_contract_risk(
        contract,
        "Ensure the target object is in your inventory before issuing a clean action, because cleaning requires holding it.",
    )
    assert risk["drop"]
    assert "over_verification_risk" in risk["reasons"]
    print("PASS v1 mode preserves original rule")


def check_v2_over_verification_is_tighter():
    agent = make_agent(mode="per_insight_rule_v2")
    contract = agent._parse_goal_contract("put a clean plate in countertop.")

    risk = agent._assess_goal_contract_risk(
        contract,
        "Ensure the target object is in your inventory before issuing a clean action, because cleaning requires holding it.",
    )
    assert not risk["drop"]
    assert risk["reasons"] == []

    risk = agent._assess_goal_contract_risk(
        contract,
        "Verify the object is held before cleaning it.",
    )
    assert not risk["drop"]

    risk = agent._assess_goal_contract_risk(
        contract,
        "Repeatedly check and examine the object again when nothing happens.",
    )
    assert risk["drop"]
    assert "over_verification_risk" in risk["reasons"]

    risk = agent._assess_goal_contract_risk(
        contract,
        "Check the object, then check it again before every step.",
    )
    assert risk["drop"]
    assert "over_verification_risk" in risk["reasons"]
    print("PASS v2 over verification tightening")


def check_v2_finalization_precondition_and_final_terms():
    agent = make_agent(mode="per_insight_rule_v2")
    contract = agent._parse_goal_contract("put a hot cup in cabinet.")

    risk = agent._assess_goal_contract_risk(
        contract,
        "Ensure the target object is in your inventory before issuing a heat action, because heating requires holding the object first.",
    )
    assert not risk["drop"]

    risk = agent._assess_goal_contract_risk(
        contract,
        "Heat the object, then place it in the target receptacle to complete the goal.",
    )
    assert not risk["drop"]

    risk = agent._assess_goal_contract_risk(
        contract,
        "Keep checking whether the object is hot before doing anything else.",
    )
    assert risk["drop"]
    assert "finalization_missing" in risk["reasons"]
    print("PASS v2 finalization tightening")


def check_v2_stage_drift_is_diagnostic_only():
    agent = make_agent(mode="per_insight_rule_v2")
    contract = {
        "task_type": "place",
        "object": "plate",
        "target": "countertop",
        "count": "one",
        "required_state": "none",
        "final_action": "unknown",
    }
    drift_insight = (
        "Open the fridge, go to the microwave, check the cabinet, then return to the fridge "
        "while searching for a different location."
    )
    assert agent._has_stage_drift(contract, drift_insight.lower())
    risk = agent._assess_goal_contract_risk(contract, drift_insight)
    assert not risk["drop"]
    assert risk["reasons"] == []
    assert risk["diagnostic_reasons"] == ["stage_drift"]
    print("PASS v2 stage drift diagnostic-only")


def check_reconstruction_skip_limit_and_diagnostics():
    agent = make_agent(max_context_chars=70)
    agent.goal = "put two cd in safe."
    agent.init_obs = "You are in the middle of a room."
    prompt = """## Key Insights from Related Tasks
1. Find one object and put it in the safe.
2. Move the item to the target.
"""
    final_prompt = agent._gate_gmemory_prompt_per_insight(prompt)
    limited_prompt = agent._limit_gmemory_prompt_chars(final_prompt)
    diagnostics = agent.get_diagnostics()
    json.dumps(diagnostics)
    assert final_prompt == ""
    assert limited_prompt == ""
    assert diagnostics["gmemory_gate"]["task_decision"] == "skip"
    assert diagnostics["gmemory_gate"]["kept_count"] == 0
    assert diagnostics["gmemory_gate"]["original_memory_prompt"] == prompt
    assert diagnostics["gmemory_gate"]["final_memory_prompt"] == ""

    agent.goal = "put a plate in countertop."
    long_prompt = """## Key Insights from Related Tasks
1. Put the object in the target receptacle to complete the task after locating it carefully.
"""
    gated_prompt = agent._gate_gmemory_prompt_per_insight(long_prompt)
    limited_prompt = agent._limit_gmemory_prompt_chars(gated_prompt)
    agent.gmemory_gate_diagnostics["final_memory_prompt"] = limited_prompt
    agent.gmemory_gate_diagnostics["final_memory_chars"] = len(limited_prompt)
    assert gated_prompt.startswith("## Key Insights from Related Tasks")
    assert len(limited_prompt) <= 70
    assert agent.get_diagnostics()["gmemory_gate"]["final_memory_prompt"] == limited_prompt
    print("PASS reconstruction and diagnostics")


def check_diagnostics_only_preserves_prompt_exactly():
    agent = make_agent(mode="per_insight_rule_v2", diagnostics_only=True)
    agent.set_current_task_type("puttwo")
    agent.goal = "put two cd in safe."
    agent.init_obs = "You are in the middle of a room."
    prompt = """## Key Insights from Related Tasks
The following are insights gathered during the execution of similar tasks.

1. Find one object and put it in the target, because this completes the immediate goal.
2. Repeat for the second object and count both placements.
---
"""
    returned_prompt = agent._diagnose_gmemory_prompt_per_insight(prompt)
    diagnostics = agent.get_diagnostics()["gmemory_gate"]
    assert returned_prompt == prompt
    assert diagnostics["diagnostics_only"] is True
    assert diagnostics["task_decision"] == "diagnostics_only"
    assert diagnostics["contract"]["task_type"] == "puttwo"
    assert diagnostics["original_memory_prompt"] == prompt
    assert diagnostics["final_memory_prompt"] == prompt
    assert diagnostics["dropped_count"] == 1
    assert diagnostics["dropped_insights"][0]["reasons"] == ["cardinality_mismatch"]
    print("PASS diagnostics-only preserves prompt exactly")


def check_v3_place_state_workflow_pollution():
    agent = make_agent(mode="per_insight_task_type_rule_v3")
    contract = {
        "task_type": "place",
        "object": "plate",
        "target": "countertop",
        "count": "one",
        "required_state": "none",
        "final_action": "put",
    }

    risk = agent._assess_goal_contract_risk(
        contract,
        "Use the fridge or microwave to process the object, then verify device readiness.",
    )
    assert risk["drop"]
    assert risk["reasons"] == ["place_state_workflow_pollution"]

    risk = agent._assess_goal_contract_risk(
        contract,
        "Find the plate, pick it up, and put it on the countertop.",
    )
    assert not risk["drop"]
    print("PASS v3 place state workflow pollution")


def check_v3_puttwo_single_object_completion_boundary():
    agent = make_agent(mode="per_insight_task_type_rule_v3")
    contract = {
        "task_type": "puttwo",
        "object": "soapbar",
        "target": "cabinet",
        "count": "two",
        "required_state": "none",
        "final_action": "put",
    }

    risk = agent._assess_goal_contract_risk(
        contract,
        "Find one object and put it in the target, because this completes the immediate goal.",
    )
    assert risk["drop"]
    assert risk["reasons"] == ["puttwo_cardinality_mismatch"]

    risk = agent._assess_goal_contract_risk(
        contract,
        "Always return to the target receptacle after picking up an object.",
    )
    assert not risk["drop"]

    risk = agent._assess_goal_contract_risk(
        contract,
        "After placing the first object, search for the second remaining object and return to the target container.",
    )
    assert not risk["drop"]
    print("PASS v3 puttwo single-object completion boundary")


def check_v3_puttwo_cardinality_tightening():
    agent = make_agent(mode="per_insight_task_type_rule_v3")
    contract = {
        "task_type": "puttwo",
        "object": "soapbar",
        "target": "cabinet",
        "count": "two",
        "required_state": "none",
        "final_action": "put",
    }

    risk = agent._assess_goal_contract_risk(
        contract,
        "Verify that the object you intend to clean matches the required type, because cleaning the wrong item fails.",
    )
    assert risk["drop"]
    assert "puttwo_state_workflow_pollution" in risk["reasons"]

    risk = agent._assess_goal_contract_risk(
        contract,
        "Identify and acquire the exact target object before performing any actions.",
    )
    assert risk["drop"]
    assert risk["reasons"] == ["puttwo_weak_cardinality_signal"]

    risk = agent._assess_goal_contract_risk(
        contract,
        "Open a container only when you need to retrieve or store an item.",
    )
    assert risk["drop"]
    assert risk["reasons"] == ["puttwo_weak_cardinality_signal"]

    risk = agent._assess_goal_contract_risk(
        contract,
        "Always return to the target receptacle after picking up an object.",
    )
    assert not risk["drop"]

    risk = agent._assess_goal_contract_risk(
        contract,
        "After placing one object, continue searching for the remaining object.",
    )
    assert not risk["drop"]
    print("PASS v3 puttwo cardinality tightening")


def check_v3_obvious_verification_loop_risk():
    agent = make_agent(mode="per_insight_task_type_rule_v3")
    contract = {
        "task_type": "place",
        "object": "plate",
        "target": "countertop",
        "count": "one",
        "required_state": "none",
        "final_action": "put",
    }

    risk = agent._assess_goal_contract_risk(
        contract,
        "Repeatedly check and examine the object again when nothing happens.",
    )
    assert risk["drop"]
    assert risk["reasons"] == ["obvious_verification_loop_risk"]
    print("PASS v3 obvious verification loop risk")


def check_v3_broad_over_verification_gate():
    agent = make_agent(mode="per_insight_task_type_rule_v3")
    contract = {
        "task_type": "place",
        "object": "plate",
        "target": "countertop",
        "count": "one",
        "required_state": "none",
        "final_action": "put",
    }

    risk = agent._assess_goal_contract_risk(
        contract,
        "Before each action, verify the object location, readiness, and state.",
    )
    assert risk["drop"]
    assert risk["reasons"] == ["broad_over_verification_workflow_pollution"]

    risk = agent._assess_goal_contract_risk(
        contract,
        "After every step, confirm the state and check preconditions.",
    )
    assert risk["drop"]
    assert risk["reasons"] == ["broad_over_verification_workflow_pollution"]

    risk = agent._assess_goal_contract_risk(
        contract,
        "Before each action, verify preconditions, then take the object and put it in the target.",
    )
    assert not risk["drop"]
    assert risk["diagnostic_reasons"] == ["broad_over_verification_workflow_pollution"]

    risk = agent._assess_goal_contract_risk(
        contract,
        "After each action, confirm success, then return to the target receptacle.",
    )
    assert not risk["drop"]
    assert risk["diagnostic_reasons"] == ["broad_over_verification_workflow_pollution"]

    risk = agent._assess_goal_contract_risk(
        {
            "task_type": "other",
            "object": "plate",
            "target": "desklamp",
            "count": "one",
            "required_state": "none",
            "final_action": "examine",
        },
        (
            "Ensure each action uses the exact valid-action phrasing, selects the correct object, "
            "applies any required transformation at the appropriate appliance, verifies the resulting "
            "property, and then moves the object to its final location, confirming the state change "
            "after every step."
        ),
    )
    assert not risk["drop"]
    assert risk["diagnostic_reasons"] == ["broad_over_verification_workflow_pollution"]

    risk = agent._assess_goal_contract_risk(
        contract,
        "Verify that the object is soapbar, not soapbottle.",
    )
    assert not risk["drop"]
    assert risk["diagnostic_reasons"] == []

    risk = agent._assess_goal_contract_risk(
        {
            "task_type": "heat",
            "object": "mug",
            "target": "countertop",
            "count": "one",
            "required_state": "hot",
            "final_action": "put",
        },
        "Close the microwave before heating.",
    )
    assert not risk["drop"]
    assert risk["diagnostic_reasons"] == []

    risk = agent._assess_goal_contract_risk(
        contract,
        "Check valid actions after one failed action.",
    )
    assert not risk["drop"]
    assert risk["diagnostic_reasons"] == []
    print("PASS v3 broad over-verification gate")


def check_v3_look_specific_refinement():
    agent = make_agent(mode="per_insight_task_type_rule_v3")
    contract = {
        "task_type": "look",
        "object": "bowl",
        "target": "desklamp",
        "count": "one",
        "required_state": "none",
        "final_action": "examine",
    }

    risk = agent._assess_goal_contract_risk(
        contract,
        "Locate the bowl and desklamp, then use the exact command examine bowl with desklamp.",
    )
    assert not risk["drop"]
    assert risk["reasons"] == []

    risk = agent._assess_goal_contract_risk(
        contract,
        "Search the room and navigate to the location containing both objects before acting.",
    )
    assert not risk["drop"]
    assert risk["reasons"] == []

    risk = agent._assess_goal_contract_risk(
        contract,
        "Find the object and put it in the target receptacle to complete the task.",
    )
    assert risk["drop"]
    assert risk["reasons"] == ["look_workflow_pollution"]

    risk = agent._assess_goal_contract_risk(
        contract,
        "Clean the object using the sinkbasin before placing it in the final location.",
    )
    assert risk["drop"]
    assert risk["reasons"] == ["look_workflow_pollution"]
    print("PASS v3 look-specific refinement")


def check_v3_state_finalization_is_diagnostic_only():
    agent = make_agent(mode="per_insight_task_type_rule_v3")
    contract = {
        "task_type": "clean",
        "object": "bowl",
        "target": "cabinet",
        "count": "one",
        "required_state": "clean",
        "final_action": "put",
    }

    risk = agent._assess_goal_contract_risk(
        contract,
        "Keep checking whether the bowl is clean before doing anything else.",
    )
    assert not risk["drop"]
    assert risk["reasons"] == []
    assert risk["diagnostic_reasons"] == ["missing_finalization_signal"]
    print("PASS v3 state finalization diagnostic-only")


def check_v3_state_action_refinement():
    agent = make_agent(mode="per_insight_task_type_rule_v3")

    clean_contract = {
        "task_type": "clean",
        "object": "bowl",
        "target": "cabinet",
        "count": "one",
        "required_state": "clean",
        "final_action": "put",
    }
    risk = agent._assess_goal_contract_risk(
        clean_contract,
        "Clean the bowl with the sinkbasin after taking it.",
    )
    assert not risk["drop"]
    assert risk["diagnostic_reasons"] == []

    heat_contract = {
        "task_type": "heat",
        "object": "mug",
        "target": "countertop",
        "count": "one",
        "required_state": "hot",
        "final_action": "put",
    }
    risk = agent._assess_goal_contract_risk(
        heat_contract,
        "Heat the mug using the microwave once it is available.",
    )
    assert not risk["drop"]
    assert risk["diagnostic_reasons"] == []

    cool_contract = {
        "task_type": "cool",
        "object": "apple",
        "target": "table",
        "count": "one",
        "required_state": "cool",
        "final_action": "put",
    }
    risk = agent._assess_goal_contract_risk(
        cool_contract,
        "Cool the apple in the fridge before moving on.",
    )
    assert not risk["drop"]
    assert risk["diagnostic_reasons"] == []

    risk = agent._assess_goal_contract_risk(
        clean_contract,
        "Before cleaning the bowl, first take it so the object is in inventory.",
    )
    assert not risk["drop"]
    assert risk["diagnostic_reasons"] == []

    drop_agent = make_agent(
        mode="per_insight_task_type_rule_v3",
        gate_overrides={"state_finalization_missing_action": "drop"},
    )
    risk = drop_agent._assess_goal_contract_risk(
        clean_contract,
        "Keep checking whether the bowl is clean before doing anything else.",
    )
    assert risk["drop"]
    assert risk["reasons"] == ["missing_finalization_signal"]
    assert risk["diagnostic_reasons"] == []
    print("PASS v3 state action refinement")


def main():
    check_disabled_path_uses_legacy_filter()
    check_phase0_intervention_diagnostics_and_compat_prompt_state()
    check_phase1_delayed_reset_and_prompt_visibility()
    check_phase1_trigger_ttl_cooldown_and_progress_delta()
    check_phase1_trigger_skip_reasons()
    check_phase11_query_action_loop_trigger()
    check_phase21_task_start_ttl_then_stuck_reactivation()
    check_phase21_no_usable_memory_skips_task_start_visibility()
    check_phase2_stage3_analysis_metrics()
    check_goal_contract_parser()
    check_insight_split()
    check_instruction_preamble_filtered()
    check_cardinality_mismatch_gate()
    check_over_verification_gate()
    check_v1_mode_still_uses_original_rule()
    check_v2_over_verification_is_tighter()
    check_v2_finalization_precondition_and_final_terms()
    check_v2_stage_drift_is_diagnostic_only()
    check_reconstruction_skip_limit_and_diagnostics()
    check_diagnostics_only_preserves_prompt_exactly()
    check_v3_place_state_workflow_pollution()
    check_v3_puttwo_single_object_completion_boundary()
    check_v3_puttwo_cardinality_tightening()
    check_v3_obvious_verification_loop_risk()
    check_v3_broad_over_verification_gate()
    check_v3_look_specific_refinement()
    check_v3_state_finalization_is_diagnostic_only()
    check_v3_state_action_refinement()


if __name__ == "__main__":
    main()
