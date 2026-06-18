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


GMemoryContextEfficientAgent = load_gmemory_agent_class()


class FakeLLM:
    engine = "fake"
    context_length = 32768
    max_tokens = 512

    def num_tokens_from_messages(self, messages):
        return sum(len(message.get("content", "")) for message in messages)


def make_agent(max_context_chars=5000, mode="per_insight_rule_v2", diagnostics_only=False):
    return GMemoryContextEfficientAgent(
        llm_model=FakeLLM(),
        need_goal=True,
        gmemory={
            "enabled": True,
            "base_url": "http://127.0.0.1:8090",
            "memory_only": True,
            "max_context_chars": max_context_chars,
            "goal_contract_gate": {
                "enabled": True,
                "mode": mode,
                "split_mode": "c1_insight_lines",
                "min_kept_insights": 1,
                "max_kept_insights": 3,
                "diagnostics_only": diagnostics_only,
            },
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


def check_goal_contract_parser():
    agent = make_agent()

    agent.set_current_task_type("clean")
    contract = agent._parse_goal_contract("put a clean plate in countertop.")
    assert contract["task_type"] == "clean"
    assert contract["count_constraint"] == "one"
    assert contract["state_requirement"] == "clean"
    assert contract["final_action"] == "put"
    assert contract["completion_pattern"] == "state_change_then_finalize"
    assert contract["target_object"] == "plate"
    assert contract["target_receptacle_or_tool"] == "countertop"

    agent.set_current_task_type("puttwo")
    contract = agent._parse_goal_contract("put two cd in safe.")
    assert contract["task_type"] == "puttwo"
    assert contract["count_constraint"] == "two"
    assert contract["state_requirement"] == "none"
    assert contract["final_action"] == "put"
    assert contract["completion_pattern"] == "multi_object_place"
    assert contract["target_object"] == "cd"
    assert contract["target_receptacle_or_tool"] == "safe"

    agent.set_current_task_type("look")
    contract = agent._parse_goal_contract("examine the alarmclock with the desklamp.")
    assert contract["task_type"] == "look"
    assert contract["count_constraint"] == "one"
    assert contract["final_action"] == "examine"
    assert contract["completion_pattern"] == "light_or_examine"
    assert contract["target_object"] == "alarmclock"
    assert contract["target_receptacle_or_tool"] == "desklamp"
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
        "count_constraint": "one",
        "state_requirement": "none",
        "final_action": "unknown",
        "target_object": "plate",
        "target_receptacle_or_tool": "countertop",
        "needs_intermediate_state": False,
        "needs_finalization": False,
        "completion_pattern": "unknown",
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


def main():
    check_disabled_path_uses_legacy_filter()
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


if __name__ == "__main__":
    main()
