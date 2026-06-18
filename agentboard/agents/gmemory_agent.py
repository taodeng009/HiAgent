"""Context-efficient HiAgent variant with GMemory retrieval hooks."""
from __future__ import annotations

import os
import re
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Dict, List, Optional

from common.registry import registry
from utils.logging.agent_logger import AgentLogger

from .cme_final import ContextEfficientAgentV2
from .gmemory_client import GMemoryClient


logger = AgentLogger(__name__)


@registry.register_agent("GMemoryContextEfficientAgent")
class GMemoryContextEfficientAgent(ContextEfficientAgentV2):
    def __init__(
        self,
        llm_model,
        memory_size=100,
        examples=[],
        instruction="",
        init_prompt_path=None,
        system_message="You are a helpful assistant.",
        need_goal=False,
        check_actions=None,
        check_inventory=None,
        use_parser=True,
        enable_retrieve_instruction=True,
        check_actions_prompt_mode="strict",
        gmemory: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            llm_model,
            memory_size,
            examples,
            instruction,
            init_prompt_path,
            system_message,
            need_goal,
            check_actions,
            check_inventory,
            use_parser,
            enable_retrieve_instruction,
            check_actions_prompt_mode,
        )
        self.gmemory_config = gmemory or {}
        self.gmemory_enabled = bool(self.gmemory_config.get("enabled", False))
        self.gmemory_recall_on_reset = bool(self.gmemory_config.get("recall_on_reset", True))
        self.gmemory_upload_on_finish = bool(self.gmemory_config.get("upload_on_finish", True))
        self.gmemory_max_context_chars = int(self.gmemory_config.get("max_context_chars", 1000))
        self.gmemory_memory_only = bool(self.gmemory_config.get("memory_only", False))
        self.gmemory_goal_contract_gate_config = self.gmemory_config.get("goal_contract_gate", {}) or {}
        self.current_task_type = ""
        self.gmemory_prompt = ""
        self.gmemory_gate_diagnostics = self._empty_gate_diagnostics()
        self.gmemory_client = self._build_gmemory_client()

    def set_current_task_type(self, task_type: Optional[str]) -> None:
        self.current_task_type = str(task_type or "").strip()

    def _build_gmemory_client(self) -> Optional[GMemoryClient]:
        if not self.gmemory_enabled:
            return None
        base_url = self.gmemory_config.get("base_url")
        if not base_url:
            logger.warning("GMemory is enabled but base_url is incomplete")
            return None
        timeout = self.gmemory_config.get("timeout", 10.0)
        return GMemoryClient(base_url=base_url, timeout=timeout)

    def reset(self, goal, init_obs, init_act=None):
        super().reset(goal, init_obs, init_act)
        self.gmemory_prompt = ""
        self.gmemory_gate_diagnostics = self._empty_gate_diagnostics()
        if not self.gmemory_enabled or not self.gmemory_recall_on_reset or self.gmemory_client is None:
            return
        try:
            gate_enabled = self._goal_contract_gate_enabled()
            response = self.gmemory_client.retrieve(
                task_type=self.gmemory_config.get("task_type") or os.environ.get("EVALTASK", ""),
                goal=goal,
                initial_observation=init_obs,
                query=self.gmemory_config.get("query"),
                max_chars=None if gate_enabled else self.gmemory_max_context_chars,
            )
            raw_prompt = response.get("memory_prompt", "")
            if gate_enabled:
                prepared_prompt = self._prepare_gmemory_prompt_for_gate(raw_prompt)
                if self._goal_contract_gate_diagnostics_only():
                    self._diagnose_gmemory_prompt_per_insight(prepared_prompt)
                    self.gmemory_prompt = self._limit_gmemory_prompt_chars(prepared_prompt)
                else:
                    gated_prompt = self._gate_gmemory_prompt_per_insight(prepared_prompt)
                    self.gmemory_prompt = self._limit_gmemory_prompt_chars(gated_prompt)
                self.gmemory_gate_diagnostics["final_memory_chars"] = len(self.gmemory_prompt)
                self.gmemory_gate_diagnostics["final_memory_prompt"] = self.gmemory_prompt
                self.gmemory_gate_diagnostics["memory_injected"] = bool(self.gmemory_prompt)
            else:
                self.gmemory_prompt = self._filter_gmemory_prompt(raw_prompt)
            logger.info(
                "GMemory retrieve completed: memory_prompt_chars=%s",
                len(self.gmemory_prompt),
            )
        except Exception as exc:
            self.gmemory_prompt = ""
            logger.warning("GMemory retrieve failed: %s", exc)

    def make_prompt(self, need_goal=False, check_actions="check valid actions", check_inventory="inventory", system_message=''):
        with redirect_stdout(StringIO()):
            prompt = super().make_prompt(
                need_goal=need_goal,
                check_actions=check_actions,
                check_inventory=check_inventory,
                system_message=system_message,
            )
        if self.gmemory_prompt:
            prompt = self._inject_gmemory_prompt(prompt)
        print(f'------------[Prompt Start]-----------\n{prompt}\n----------[Prompt END]------------')
        return prompt

    def _inject_gmemory_prompt(self, prompt: str) -> str:
        block = f"{self.gmemory_prompt}\n\n"
        marker = self._current_history_marker()
        if marker:
            idx = prompt.find(marker)
            if idx >= 0:
                return prompt[:idx] + block + prompt[idx:]
        return block + prompt

    def _filter_gmemory_prompt(self, text: str) -> str:
        if not text:
            return ""
        context = re.sub(r"\n{3,}", "\n\n", str(text).strip())
        if self.gmemory_memory_only:
            context = self._extract_memory_sections(context)
        max_chars = max(0, self.gmemory_max_context_chars)
        if max_chars and len(context) > max_chars:
            context = context[:max_chars].rstrip()
        return context

    def _extract_memory_sections(self, context: str) -> str:
        target_headings = [
            "## Your Own Past Successes (Execution Patterns)",
            "## Key Insights from Related Tasks",
        ]
        sections = []
        for heading in target_headings:
            start = context.find(heading)
            if start < 0:
                continue
            next_heading = context.find("\n## ", start + len(heading))
            end = next_heading if next_heading >= 0 else len(context)
            section = context[start:end].strip()
            if section:
                sections.append(section)
        return "\n\n".join(sections) if sections else context

    def _goal_contract_gate_enabled(self) -> bool:
        return bool(self.gmemory_goal_contract_gate_config.get("enabled", False))

    def _goal_contract_gate_mode(self) -> str:
        return str(self.gmemory_goal_contract_gate_config.get("mode", "per_insight_rule_v1"))

    def _goal_contract_gate_diagnostics_only(self) -> bool:
        return bool(self.gmemory_goal_contract_gate_config.get("diagnostics_only", False))

    def _empty_gate_diagnostics(self) -> Dict[str, Any]:
        return {
            "enabled": self._goal_contract_gate_enabled() if hasattr(self, "gmemory_goal_contract_gate_config") else False,
            "mode": self._goal_contract_gate_mode()
            if hasattr(self, "gmemory_goal_contract_gate_config")
            else "per_insight_rule_v1",
            "diagnostics_only": self._goal_contract_gate_diagnostics_only()
            if hasattr(self, "gmemory_goal_contract_gate_config")
            else False,
            "goal": getattr(self, "goal", None),
            "contract": {},
            "task_decision": "disabled",
            "split_failed": False,
            "instruction_preamble_count": 0,
            "instruction_preamble_lines": [],
            "end_delimiter": "",
            "insight_count": 0,
            "kept_count": 0,
            "dropped_count": 0,
            "kept_insights": [],
            "dropped_insights": [],
            "diagnostic_insights": [],
            "original_memory_chars": 0,
            "final_memory_chars": 0,
            "original_memory_prompt": "",
            "final_memory_prompt": "",
            "memory_injected": False,
        }

    def _prepare_gmemory_prompt_for_gate(self, text: str) -> str:
        if not text:
            return ""
        context = re.sub(r"\n{3,}", "\n\n", str(text).strip())
        if self.gmemory_memory_only:
            context = self._extract_memory_sections(context)
        return context

    def _limit_gmemory_prompt_chars(self, memory_prompt: str) -> str:
        if not memory_prompt:
            return ""
        max_chars = max(0, self.gmemory_max_context_chars)
        if max_chars and len(memory_prompt) > max_chars:
            return memory_prompt[:max_chars].rstrip()
        return memory_prompt

    def _parse_goal_contract(self, goal: str) -> Dict[str, Any]:
        goal_text = (goal or "").strip()
        lower_goal = goal_text.lower()

        if re.search(r"\btwo\b", lower_goal):
            count = "two"
        elif re.search(r"\b(all|multiple)\b", lower_goal):
            count = "multiple"
        else:
            count = "one" if lower_goal else "unknown"

        if re.search(r"\bclean\b", lower_goal):
            required_state = "clean"
        elif re.search(r"\b(hot|heat|heated)\b", lower_goal):
            required_state = "hot"
        elif re.search(r"\b(cool|cooled)\b", lower_goal):
            required_state = "cool"
        else:
            required_state = "none"

        if re.search(r"\b(put|place)\b", lower_goal):
            final_action = "put"
        elif re.search(r"\b(examine|look at|look)\b", lower_goal):
            final_action = "examine"
        elif re.search(r"\buse\b", lower_goal):
            final_action = "use"
        else:
            final_action = "unknown"

        target_object = self._parse_target_object(lower_goal)
        target_receptacle_or_tool = self._parse_target_receptacle_or_tool(lower_goal, final_action)
        task_type = str(getattr(self, "current_task_type", "") or "").strip()

        return {
            "task_type": task_type,
            "object": target_object,
            "target": target_receptacle_or_tool,
            "count": count,
            "required_state": required_state,
            "final_action": final_action,
        }

    def _parse_target_object(self, lower_goal: str) -> str:
        patterns = [
            r"\bput\s+(?:a|an|some|the|two|all|multiple)?\s*(?:clean|hot|cool|heated|cooled)?\s*([a-z0-9]+)",
            r"\b(?:clean|heat|cool)\s+(?:a|an|some|the)?\s*([a-z0-9]+)",
            r"\b(?:examine|look at|look)\s+(?:a|an|some|the)?\s*([a-z0-9]+)",
            r"\buse\s+(?:a|an|some|the)?\s*([a-z0-9]+)",
        ]
        skip_words = {"and", "in", "on", "with", "under", "it", "them"}
        for pattern in patterns:
            match = re.search(pattern, lower_goal)
            if match:
                candidate = match.group(1).strip(" .")
                if candidate and candidate not in skip_words:
                    return candidate
        return ""

    def _parse_target_receptacle_or_tool(self, lower_goal: str, final_action: str) -> str:
        if final_action == "put":
            match = re.search(r"\b(?:in|on)\s+([a-z0-9]+)", lower_goal)
            return match.group(1).strip(" .") if match else ""
        match = re.search(r"\b(?:with|under)\s+(?:a|an|some|the)?\s*([a-z0-9]+)", lower_goal)
        return match.group(1).strip(" .") if match else ""

    def _split_insights(self, memory_prompt: str) -> List[str]:
        if not memory_prompt:
            return []

        context = memory_prompt.strip()
        heading = "## Key Insights from Related Tasks"
        heading_index = context.lower().find(heading.lower())
        if heading_index >= 0:
            section = context[heading_index + len(heading):].strip()
            next_heading = re.search(r"\n##\s+", section)
            if next_heading:
                section = section[:next_heading.start()].strip()
        else:
            section = context

        lines = [line.strip() for line in section.splitlines()]
        insights = []
        current = []
        numbered_or_bulleted = False
        preamble_lines = []
        end_delimiter = ""
        item_pattern = re.compile(r"^(?:\d+[\.\)]|[-*])\s+(.*)$")

        for line in lines:
            if not line:
                continue
            if line == "---":
                end_delimiter = line
                break
            match = item_pattern.match(line)
            if match:
                numbered_or_bulleted = True
                if current:
                    insights.append(" ".join(current).strip())
                current = [match.group(1).strip()]
            elif numbered_or_bulleted and current:
                current.append(line)
            elif not numbered_or_bulleted:
                preamble_lines.append(line)

        if current:
            insights.append(" ".join(current).strip())

        insights = [insight for insight in insights if insight]
        self.gmemory_gate_diagnostics["instruction_preamble_count"] = len(preamble_lines)
        self.gmemory_gate_diagnostics["instruction_preamble_lines"] = preamble_lines
        self.gmemory_gate_diagnostics["end_delimiter"] = end_delimiter
        if not insights and context:
            self.gmemory_gate_diagnostics["split_failed"] = True
            return [context]
        if len(insights) == 1 and insights[0] == context and heading_index >= 0:
            self.gmemory_gate_diagnostics["split_failed"] = True
        return insights

    def _assess_goal_contract_risk(
        self,
        contract: Dict[str, Any],
        insight: str,
        initial_observation: str = "",
    ) -> Dict[str, Any]:
        mode = self._goal_contract_gate_mode()
        if mode == "per_insight_task_type_rule_v3":
            return self._assess_goal_contract_risk_v3(contract, insight, initial_observation)
        if mode == "per_insight_rule_v2":
            return self._assess_goal_contract_risk_v2(contract, insight, initial_observation)
        return self._assess_goal_contract_risk_v1(contract, insight, initial_observation)

    def _assess_goal_contract_risk_v1(
        self,
        contract: Dict[str, Any],
        insight: str,
        initial_observation: str = "",
    ) -> Dict[str, Any]:
        text = (insight or "").lower()
        reasons = []

        cardinality_terms = [
            "two",
            "both",
            "second",
            "another",
            "repeat",
            "all",
            "remaining",
            "until all",
            "count",
        ]
        if contract.get("count") in {"two", "multiple"} and not self._contains_any(text, cardinality_terms):
            reasons.append("cardinality_mismatch")

        intermediate_terms = [
            "clean",
            "heat",
            "hot",
            "cool",
            "fridge",
            "microwave",
            "sinkbasin",
            "check",
            "verify",
            "ensure",
            "examine",
        ]
        finalization_terms = [
            "put",
            "place",
            "final",
            "target",
            "complete",
            "finish",
            "use",
            "examine",
        ]
        has_finalization = self._contains_any(text, finalization_terms) or bool(re.search(r"\b(in|on)\b", text))
        if (
            self._contract_needs_finalization(contract)
            and self._contains_any(text, intermediate_terms)
            and not has_finalization
        ):
            reasons.append("finalization_missing")

        verification_terms = [
            "check",
            "examine",
            "verify",
            "ensure",
            "because",
            "confirm",
            "inventory",
            "nothing happens",
        ]
        verification_count = sum(len(re.findall(r"\b" + re.escape(term) + r"\b", text)) for term in verification_terms)
        line_count = max(1, len([line for line in (insight or "").splitlines() if line.strip()]))
        if verification_count >= 4 or verification_count / line_count >= 3:
            reasons.append("over_verification_risk")

        if self._has_stage_drift(contract, text):
            reasons.append("stage_drift")

        return {"drop": bool(reasons), "reasons": reasons, "diagnostic_reasons": []}

    def _assess_goal_contract_risk_v2(
        self,
        contract: Dict[str, Any],
        insight: str,
        initial_observation: str = "",
    ) -> Dict[str, Any]:
        text = (insight or "").lower()
        reasons = []
        diagnostic_reasons = []

        cardinality_terms = [
            "two",
            "both",
            "second",
            "another",
            "repeat",
            "all",
            "remaining",
            "until all",
            "count",
        ]
        if contract.get("count") in {"two", "multiple"} and not self._contains_any(text, cardinality_terms):
            reasons.append("cardinality_mismatch")

        intermediate_terms = [
            "clean",
            "heat",
            "hot",
            "cool",
            "fridge",
            "microwave",
            "sinkbasin",
            "check",
            "verify",
            "ensure",
            "examine",
        ]
        if (
            self._contract_needs_finalization(contract)
            and self._contains_any(text, intermediate_terms)
            and not self._has_final_action_terms(text)
            and not self._has_state_action_precondition(text, contract)
        ):
            reasons.append("finalization_missing")

        if self._has_over_verification_hard_signal(text) or self._has_repeated_probe_risk(text):
            reasons.append("over_verification_risk")

        if self._has_stage_drift(contract, text):
            diagnostic_reasons.append("stage_drift")

        return {"drop": bool(reasons), "reasons": reasons, "diagnostic_reasons": diagnostic_reasons}

    def _assess_goal_contract_risk_v3(
        self,
        contract: Dict[str, Any],
        insight: str,
        initial_observation: str = "",
    ) -> Dict[str, Any]:
        text = (insight or "").lower()
        reasons = []
        diagnostic_reasons = []

        if self._has_place_state_workflow_pollution(contract, text):
            reasons.append("place_state_workflow_pollution")

        if self._has_puttwo_single_object_completion(contract, text):
            reasons.append("puttwo_cardinality_mismatch")

        if self._has_obvious_verification_loop_risk(text):
            reasons.append("obvious_verification_loop_risk")

        if self._has_clean_heat_cool_missing_finalization_diagnostic(contract, text):
            diagnostic_reasons.append("missing_finalization_signal")

        return {"drop": bool(reasons), "reasons": reasons, "diagnostic_reasons": diagnostic_reasons}

    def _has_place_state_workflow_pollution(self, contract: Dict[str, Any], text: str) -> bool:
        if contract.get("task_type") != "place":
            return False
        state_workflow_terms = [
            "clean",
            "cleaning",
            "heat",
            "heating",
            "hot",
            "cool",
            "cooling",
            "cooled",
            "fridge",
            "microwave",
            "sinkbasin",
            "state change",
            "processed",
            "processing",
            "device readiness",
            "appliance",
        ]
        if not self._contains_any(text, state_workflow_terms):
            return False
        target_object = (contract.get("object") or "").lower()
        target_receptacle = (contract.get("target") or "").lower()
        supports_current_place_task = bool(target_object and target_object in text) or bool(
            target_receptacle and target_receptacle in text
        )
        return not supports_current_place_task

    def _has_puttwo_single_object_completion(self, contract: Dict[str, Any], text: str) -> bool:
        if contract.get("task_type") != "puttwo" and contract.get("count") not in {"two", "multiple"}:
            return False
        single_object_markers = [
            r"\bone object\b",
            r"\ba single object\b",
            r"\bthe object\b",
            r"\bthe item\b",
            r"\bone item\b",
            r"\bit\b",
        ]
        completion_markers = [
            r"\btask is complete\b",
            r"\bcompletes? the (?:task|goal|immediate goal)\b",
            r"\bfinish(?:es)? the (?:task|goal)\b",
            r"\bto complete the (?:task|goal|immediate goal)\b",
            r"\bthis completes\b",
            r"\bthis finishes\b",
        ]
        placement_markers = [
            r"\bput\b",
            r"\bplace\b",
            r"\bplacing\b",
            r"\bplaced\b",
            r"\bin the target\b",
            r"\bon the target\b",
            r"\btarget receptacle\b",
        ]
        has_single_object = any(re.search(pattern, text) for pattern in single_object_markers)
        has_completion = any(re.search(pattern, text) for pattern in completion_markers)
        has_placement = any(re.search(pattern, text) for pattern in placement_markers)
        return has_single_object and has_completion and has_placement

    def _has_obvious_verification_loop_risk(self, text: str) -> bool:
        if self._contains_any(text, ["repeated", "repeatedly", "again", "loop", "nothing happens"]):
            return True
        probe_terms = ["check", "examine", "inventory"]
        return any(len(re.findall(r"\b" + re.escape(term) + r"\b", text)) >= 2 for term in probe_terms)

    def _has_clean_heat_cool_missing_finalization_diagnostic(self, contract: Dict[str, Any], text: str) -> bool:
        if contract.get("task_type") not in {"clean", "heat", "cool"}:
            return False
        intermediate_terms = [
            "clean",
            "cleaning",
            "heat",
            "heating",
            "hot",
            "cool",
            "cooling",
            "fridge",
            "microwave",
            "sinkbasin",
            "check",
            "verify",
            "ensure",
        ]
        return (
            self._contains_any(text, intermediate_terms)
            and not self._has_final_action_terms(text)
            and not self._has_state_action_precondition(text, contract)
        )

    def _has_final_action_terms(self, text: str) -> bool:
        finalization_terms = [
            "put",
            "place",
            "final",
            "target",
            "complete",
            "finish",
            "use",
            "examine",
        ]
        return self._contains_any(text, finalization_terms) or bool(re.search(r"\b(in|on)\b", text))

    def _has_state_action_precondition(self, text: str, contract: Dict[str, Any]) -> bool:
        required_state = contract.get("required_state")
        if required_state == "clean":
            action_terms = ["clean", "cleaning"]
        elif required_state == "hot":
            action_terms = ["heat", "heating"]
        elif required_state == "cool":
            action_terms = ["cool", "cooling"]
        else:
            action_terms = ["clean", "cleaning", "heat", "heating", "cool", "cooling"]

        object_available_terms = [
            "inventory",
            "in hand",
            "held",
            "hold",
            "holding",
            "take",
            "pick up",
            "pickup",
            "grab",
        ]
        precondition_terms = ["before", "requires", "must", "need", "first"]
        return (
            self._contains_any(text, action_terms)
            and self._contains_any(text, object_available_terms)
            and self._contains_any(text, precondition_terms)
        )

    def _has_over_verification_hard_signal(self, text: str) -> bool:
        return self._contains_any(text, ["repeated", "again", "loop", "nothing happens"])

    def _has_repeated_probe_risk(self, text: str) -> bool:
        probe_terms = ["check", "examine", "inventory"]
        return any(len(re.findall(r"\b" + re.escape(term) + r"\b", text)) >= 2 for term in probe_terms)

    def _contract_needs_finalization(self, contract: Dict[str, Any]) -> bool:
        return contract.get("final_action") in {"put", "examine", "use"}

    def _has_stage_drift(self, contract: Dict[str, Any], text: str) -> bool:
        target = (contract.get("target") or "").lower()
        target_absent = bool(target) and target not in text
        tool_location_terms = [
            "fridge",
            "microwave",
            "sinkbasin",
            "stoveburner",
            "toaster",
            "coffeemachine",
            "countertop",
            "cabinet",
            "drawer",
            "safe",
            "shelf",
            "desk",
        ]
        action_terms = [
            "open",
            "close",
            "take",
            "put",
            "clean",
            "cool",
            "heat",
            "check",
            "examine",
            "inventory",
            "go to",
        ]
        location_hits = [term for term in tool_location_terms if re.search(r"\b" + re.escape(term) + r"\b", text)]
        action_hits = [term for term in action_terms if re.search(r"\b" + re.escape(term) + r"\b", text)]
        repeated_location = any(len(re.findall(r"\b" + re.escape(term) + r"\b", text)) >= 2 for term in tool_location_terms)
        return target_absent and len(location_hits) >= 3 and len(action_hits) >= 3 and repeated_location

    def _contains_any(self, text: str, terms: List[str]) -> bool:
        for term in terms:
            if " " in term:
                if term in text:
                    return True
            elif re.search(r"\b" + re.escape(term) + r"\b", text):
                return True
        return False

    def _reconstruct_insight_prompt(self, kept_insights: List[str]) -> str:
        if not kept_insights:
            return ""
        lines = ["## Key Insights from Related Tasks"]
        preamble_lines = self.gmemory_gate_diagnostics.get("instruction_preamble_lines") or []
        if preamble_lines:
            lines.extend(preamble_lines)
            lines.append("")
        lines.extend(f"{idx}. {insight}" for idx, insight in enumerate(kept_insights, start=1))
        end_delimiter = self.gmemory_gate_diagnostics.get("end_delimiter")
        if end_delimiter:
            lines.append(end_delimiter)
        return "\n".join(lines)

    def _diagnose_gmemory_prompt_per_insight(self, memory_prompt: str) -> str:
        self.gmemory_gate_diagnostics = self._empty_gate_diagnostics()
        self.gmemory_gate_diagnostics.update(
            {
                "enabled": True,
                "mode": self._goal_contract_gate_mode(),
                "diagnostics_only": True,
                "goal": getattr(self, "goal", None),
                "original_memory_chars": len(memory_prompt or ""),
                "original_memory_prompt": memory_prompt or "",
            }
        )

        if not memory_prompt:
            self.gmemory_gate_diagnostics["task_decision"] = "skip"
            self.gmemory_gate_diagnostics["final_memory_prompt"] = ""
            return ""

        contract = self._parse_goal_contract(getattr(self, "goal", "") or "")
        insights = self._split_insights(memory_prompt)
        kept_insights = []
        dropped_insights = []
        diagnostic_insights = []

        for insight in insights:
            risk = self._assess_goal_contract_risk(
                contract=contract,
                insight=insight,
                initial_observation=getattr(self, "init_obs", "") or "",
            )
            if risk.get("diagnostic_reasons"):
                diagnostic_insights.append({"text": insight, "reasons": risk["diagnostic_reasons"]})
            if risk["drop"]:
                dropped_insights.append(
                    {
                        "text": insight,
                        "reasons": risk["reasons"],
                        "diagnostic_only_reasons": risk.get("diagnostic_reasons", []),
                    }
                )
            else:
                kept_insights.append(insight)

        self.gmemory_gate_diagnostics.update(
            {
                "contract": contract,
                "task_decision": "diagnostics_only",
                "insight_count": len(insights),
                "kept_count": len(kept_insights),
                "dropped_count": len(dropped_insights),
                "kept_insights": list(kept_insights),
                "dropped_insights": dropped_insights,
                "diagnostic_insights": diagnostic_insights,
                "final_memory_chars": len(memory_prompt),
                "final_memory_prompt": memory_prompt,
                "memory_injected": bool(memory_prompt),
            }
        )
        return memory_prompt

    def _gate_gmemory_prompt_per_insight(self, memory_prompt: str) -> str:
        self.gmemory_gate_diagnostics = self._empty_gate_diagnostics()
        self.gmemory_gate_diagnostics.update(
            {
                "enabled": True,
                "mode": self._goal_contract_gate_mode(),
                "goal": getattr(self, "goal", None),
                "original_memory_chars": len(memory_prompt or ""),
                "original_memory_prompt": memory_prompt or "",
            }
        )

        if not memory_prompt:
            self.gmemory_gate_diagnostics["task_decision"] = "skip"
            self.gmemory_gate_diagnostics["final_memory_prompt"] = ""
            return ""

        contract = self._parse_goal_contract(getattr(self, "goal", "") or "")
        insights = self._split_insights(memory_prompt)
        max_kept = int(self.gmemory_goal_contract_gate_config.get("max_kept_insights", 3))
        min_kept = int(self.gmemory_goal_contract_gate_config.get("min_kept_insights", 1))
        kept_insights = []
        dropped_insights = []
        diagnostic_insights = []

        for insight in insights:
            risk = self._assess_goal_contract_risk(
                contract=contract,
                insight=insight,
                initial_observation=getattr(self, "init_obs", "") or "",
            )
            if risk.get("diagnostic_reasons"):
                diagnostic_insights.append({"text": insight, "reasons": risk["diagnostic_reasons"]})
            if risk["drop"]:
                dropped_insights.append(
                    {
                        "text": insight,
                        "reasons": risk["reasons"],
                        "diagnostic_only_reasons": risk.get("diagnostic_reasons", []),
                    }
                )
            else:
                kept_insights.append(insight)

        if max_kept > 0:
            kept_insights = kept_insights[:max_kept]

        final_prompt = self._reconstruct_insight_prompt(kept_insights) if len(kept_insights) >= min_kept else ""
        task_decision = "inject" if final_prompt else "skip"
        self.gmemory_gate_diagnostics.update(
            {
                "contract": contract,
                "task_decision": task_decision,
                "insight_count": len(insights),
                "kept_count": len(kept_insights),
                "dropped_count": len(dropped_insights),
                "kept_insights": list(kept_insights),
                "dropped_insights": dropped_insights,
                "diagnostic_insights": diagnostic_insights,
                "final_memory_chars": len(final_prompt),
                "final_memory_prompt": final_prompt,
                "memory_injected": bool(final_prompt),
            }
        )
        return final_prompt

    def get_diagnostics(self):
        return {
            "agent_name": self.__class__.__name__,
            "gmemory_prompt_chars": len(getattr(self, "gmemory_prompt", "") or ""),
            "memory_injected_to_prompt": bool(getattr(self, "gmemory_prompt", "") or ""),
            "gmemory_gate": self.gmemory_gate_diagnostics,
        }

    def _current_history_marker(self) -> Optional[str]:
        history = getattr(self, "memory", [])[-self.memory_size:]
        if not history or not history[0]:
            return None
        key, value = history[0][0]
        marker = f"{key}: {value}"
        return marker if value is not None else f"{key}: "

    def remember_current_task(
        self,
        task_type: str = "",
        success: Optional[bool] = None,
        progress_rate: Optional[float] = None,
        score_change_record=None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not self.gmemory_enabled or not self.gmemory_upload_on_finish or self.gmemory_client is None:
            return None
        if success is None:
            logger.warning("GMemory episode upload skipped: success is missing")
            return None
        episode = self._memory_to_gmemory_episode(
            task_type=task_type,
            success=success,
            progress_rate=progress_rate,
            score_change_record=score_change_record,
            metadata=metadata,
        )
        if episode is None:
            return None
        try:
            response = self.gmemory_client.save_episode(**episode)
            logger.info(
                "GMemory episode upload completed: stored=%s episode_id=%s",
                response.get("stored"),
                response.get("episode_id"),
            )
            if response.get("stored") is False:
                logger.warning("GMemory episode upload was not stored: %s", response)
            return response
        except Exception as exc:
            logger.warning("GMemory episode upload failed: %s", exc)
            return None

    def _memory_to_gmemory_episode(
        self,
        task_type: str,
        success: bool,
        progress_rate: Optional[float],
        score_change_record=None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        memory = getattr(self, "memory", [])
        initial_observation = getattr(self, "init_obs", None)
        reward_by_step = self._score_change_record_to_dict(score_change_record)
        steps = []
        current_subgoal = None
        action_step_id = 0

        for item in memory:
            fields = dict(item)
            if "Subgoal" in fields:
                current_subgoal = fields.get("Subgoal")
                continue
            action = fields.get("Action")
            observation = fields.get("Observation")
            if initial_observation is None and observation is not None:
                initial_observation = observation
            if action is None:
                continue
            step = {
                "subgoal": current_subgoal,
                "action": str(action),
                "observation": str(observation or ""),
            }
            if action_step_id in reward_by_step:
                step["reward"] = reward_by_step[action_step_id]
            steps.append(step)
            action_step_id += 1

        if not steps:
            logger.warning("GMemory episode upload skipped: no action steps recorded")
            return None

        episode_metadata = {
            "agent": "GMemoryContextEfficientAgent",
            "score_change_record": self._normalise_score_change_record(score_change_record),
            "step_count": len(steps),
        }
        if metadata:
            episode_metadata.update(metadata)

        return {
            "task_type": task_type or self.gmemory_config.get("task_type") or os.environ.get("EVALTASK", ""),
            "goal": getattr(self, "goal", None),
            "initial_observation": str(initial_observation or ""),
            "success": bool(success),
            "progress_rate": self._json_scalar(progress_rate),
            "steps": steps,
            "metadata": episode_metadata,
        }

    def _score_change_record_to_dict(self, score_change_record) -> Dict[int, Any]:
        if not score_change_record:
            return {}
        rewards = {}
        for item in score_change_record:
            try:
                step_id, reward = item
                rewards[int(step_id)] = self._json_scalar(reward)
            except (TypeError, ValueError):
                continue
        return rewards

    def _normalise_score_change_record(self, score_change_record):
        if not score_change_record:
            return []
        normalised = []
        for item in score_change_record:
            try:
                step_id, reward = item
                normalised.append([int(step_id), self._json_scalar(reward)])
            except (TypeError, ValueError):
                continue
        return normalised

    def _json_scalar(self, value):
        if value is None:
            return None
        try:
            if hasattr(value, "item"):
                return value.item()
            return float(value)
        except (TypeError, ValueError):
            return value

    @classmethod
    def from_config(cls, llm_model, config):
        memory_size = config.get("memory_size", 100)
        instruction = config.get("instruction", "")
        examples = config.get("examples", [])
        init_prompt_path = config.get("init_prompt_path", None)
        system_message = config.get("system_message", "You are a helpful assistant.")
        check_actions = config.get("check_actions", None)
        check_inventory = config.get("check_inventory", None)
        use_parser = config.get("use_parser", True)
        need_goal = config.get("need_goal", False)
        enable_retrieve_instruction = config.get("enable_retrieve_instruction", True)
        check_actions_prompt_mode = config.get("check_actions_prompt_mode", "strict")
        gmemory = config.get("gmemory", {})
        return cls(
            llm_model,
            memory_size,
            examples,
            instruction,
            init_prompt_path,
            system_message,
            need_goal,
            check_actions,
            check_inventory,
            use_parser,
            enable_retrieve_instruction,
            check_actions_prompt_mode,
            gmemory,
        )
