"""Context-efficient HiAgent variant with GMemory retrieval hooks."""
from __future__ import annotations

import copy
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
        self.gmemory_base_need_aware_intervention_config = copy.deepcopy(
            self.gmemory_config.get("need_aware_intervention", {}) or {}
        )
        self.gmemory_need_aware_intervention_config = copy.deepcopy(
            self.gmemory_base_need_aware_intervention_config
        )
        self.pending_task_type = None
        self.raw_current_task_type = ""
        self.current_task_type = ""
        self.gmemory_task_type_policy_metadata = self._default_task_type_policy_metadata()
        self.cached_gmemory_prompt = ""
        self.visible_gmemory_prompt = ""
        self.gmemory_prompt = ""
        self.gmemory_retrieve_step = None
        self.gmemory_visible_ttl_remaining = None
        self.gmemory_cooldown_remaining = 0
        self.gmemory_stale_steps_since_last_progress = 0
        self.gmemory_nothing_happens_since_last_progress = 0
        self.gmemory_check_valid_actions_since_last_progress = 0
        self.gmemory_inventory_since_last_progress = 0
        self.gmemory_last_intervention_decision = {}
        self.gmemory_injection_events = []
        self.gmemory_clear_events = []
        self.gmemory_active_injection_index = None
        self.gmemory_interaction_steps = 0
        self.gmemory_exposure_steps_total = 0
        self.gmemory_stuck_trigger_count = 0
        self.gmemory_cooldown_block_count = 0
        self.gmemory_trigger_no_cached_memory_count = 0
        self.gmemory_trigger_no_usable_memory_count = 0
        self.gmemory_ineffective_reactivation_streak = 0
        self.gmemory_effective_reactivation_count = 0
        self.gmemory_suppressed_reactivation_count = 0
        self.gmemory_last_ineffective_reactivation_signal = None
        self.gmemory_last_suppression_reason = ""
        self.gmemory_gate_diagnostics = self._empty_gate_diagnostics()
        self.gmemory_intervention_diagnostics = self._empty_intervention_diagnostics()
        self.gmemory_client = self._build_gmemory_client()

    def set_current_task_type(self, task_type: Optional[str]) -> None:
        raw_task_type = str(task_type or "")
        self.pending_task_type = raw_task_type
        # Preserve the legacy behavior for helpers called before reset().
        self.raw_current_task_type = raw_task_type
        self.current_task_type = self._normalize_task_type(raw_task_type)

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
        self._reset_intervention_runtime_state()
        raw_task_type = self._consume_pending_task_type()
        self._activate_effective_intervention_policy(raw_task_type)
        self._set_gmemory_prompt_state(cached_prompt="", visible_prompt="", retrieve_step=None, reset_events=True)
        self.gmemory_gate_diagnostics = self._empty_gate_diagnostics()
        self.gmemory_intervention_diagnostics = self._empty_intervention_diagnostics()
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
                    final_prompt = self._limit_gmemory_prompt_chars(prepared_prompt)
                else:
                    gated_prompt = self._gate_gmemory_prompt_per_insight(prepared_prompt)
                    final_prompt = self._limit_gmemory_prompt_chars(gated_prompt)
                if self._need_aware_task_start_ttl_enabled():
                    self._set_gmemory_prompt_state(cached_prompt=final_prompt, visible_prompt="", retrieve_step=0)
                    if final_prompt:
                        self._inject_cached_gmemory(
                            step_id=-1,
                            progress_rate=0.0,
                            trigger_condition_satisfied=False,
                            reason="task_start_ttl",
                            phase="task_start",
                            visible_ttl=self._task_start_visibility_ttl(),
                            cooldown=self._task_start_reentry_cooldown(),
                        )
                        self._refresh_intervention_diagnostics()
                else:
                    visible_prompt = (
                        ""
                        if self._delayed_task_level_intervention_enabled()
                        or self._memory_off_intervention_enabled()
                        else final_prompt
                    )
                    self._set_gmemory_prompt_state(cached_prompt=final_prompt, visible_prompt=visible_prompt, retrieve_step=0)
                self.gmemory_gate_diagnostics["final_memory_chars"] = len(final_prompt)
                self.gmemory_gate_diagnostics["final_memory_prompt"] = final_prompt
                self.gmemory_gate_diagnostics["memory_injected"] = bool(self.gmemory_prompt)
            else:
                final_prompt = self._filter_gmemory_prompt(raw_prompt)
                if self._need_aware_task_start_ttl_enabled():
                    self._set_gmemory_prompt_state(cached_prompt=final_prompt, visible_prompt="", retrieve_step=0)
                    if final_prompt:
                        self._inject_cached_gmemory(
                            step_id=-1,
                            progress_rate=0.0,
                            trigger_condition_satisfied=False,
                            reason="task_start_ttl",
                            phase="task_start",
                            visible_ttl=self._task_start_visibility_ttl(),
                            cooldown=self._task_start_reentry_cooldown(),
                        )
                        self._refresh_intervention_diagnostics()
                else:
                    visible_prompt = (
                        ""
                        if self._delayed_task_level_intervention_enabled()
                        or self._memory_off_intervention_enabled()
                        else final_prompt
                    )
                    self._set_gmemory_prompt_state(cached_prompt=final_prompt, visible_prompt=visible_prompt, retrieve_step=0)
            logger.info(
                "GMemory retrieve completed: memory_prompt_chars=%s",
                len(self.gmemory_prompt),
            )
        except Exception as exc:
            self._set_gmemory_prompt_state(cached_prompt="", visible_prompt="", retrieve_step=None)
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

    def _goal_contract_gate_state_finalization_action(self) -> str:
        return str(self.gmemory_goal_contract_gate_config.get("state_finalization_missing_action", "diagnostic")).strip()

    @staticmethod
    def _normalize_task_type(task_type: Optional[str]) -> str:
        return str(task_type or "").strip().lower()

    def _consume_pending_task_type(self) -> str:
        raw_task_type = self.pending_task_type if self.pending_task_type is not None else ""
        self.pending_task_type = None
        self.raw_current_task_type = str(raw_task_type or "")
        self.current_task_type = self._normalize_task_type(self.raw_current_task_type)
        return self.raw_current_task_type

    def _default_task_type_policy_metadata(self) -> Dict[str, Any]:
        return {
            "task_type_policy_enabled": False,
            "raw_task_type": getattr(self, "raw_current_task_type", ""),
            "detected_task_type": getattr(self, "current_task_type", ""),
            "selected_intervention_policy": "GLOBAL",
            "default_intervention_policy": None,
            "route_matched": False,
            "selected_profile_differs_from_default": False,
            "policy_resolution_source": "router_disabled",
        }

    def _resolve_task_type_intervention_policy(self, task_type: Optional[str]) -> Dict[str, Any]:
        base_config = self.gmemory_base_need_aware_intervention_config
        policy = base_config.get("task_type_policy", {}) or {}
        master_enabled = bool(base_config.get("enabled", False))
        if master_enabled and not isinstance(policy, dict):
            raise ValueError("need_aware_intervention.task_type_policy must be a mapping")
        policy_enabled = master_enabled and bool(policy.get("enabled", False))
        normalized_task_type = self._normalize_task_type(task_type)
        if not policy_enabled:
            return {
                "effective_config": copy.deepcopy(base_config),
                "metadata": {
                    "task_type_policy_enabled": False,
                    "raw_task_type": str(task_type or ""),
                    "detected_task_type": normalized_task_type,
                    "selected_intervention_policy": "GLOBAL",
                    "default_intervention_policy": None,
                    "route_matched": False,
                    "selected_profile_differs_from_default": False,
                    "policy_resolution_source": "router_disabled",
                },
            }

        routes = policy.get("routes", {}) or {}
        profiles = policy.get("profiles", {}) or {}
        if not isinstance(routes, dict):
            raise ValueError("task_type_policy.routes must be a mapping")
        if not isinstance(profiles, dict):
            raise ValueError("task_type_policy.profiles must be a mapping")

        default_profile = str(policy.get("default_profile") or "").strip()
        if not default_profile:
            raise ValueError("task_type_policy.default_profile is required when the router is enabled")
        if default_profile not in profiles:
            raise ValueError(f"Unknown task_type_policy default profile: {default_profile}")

        route_matched = bool(normalized_task_type) and normalized_task_type in routes
        if route_matched:
            selected_profile = str(routes.get(normalized_task_type) or "").strip()
            resolution_source = "explicit_route"
        else:
            selected_profile = default_profile
            resolution_source = "default_unknown" if normalized_task_type else "default_missing"
        if not selected_profile or selected_profile not in profiles:
            raise ValueError(
                f"Unknown task_type_policy profile {selected_profile!r} for task type {normalized_task_type!r}"
            )
        profile = profiles[selected_profile]
        if not isinstance(profile, dict):
            raise ValueError(f"task_type_policy profile {selected_profile!r} must be a mapping")

        effective_config = copy.deepcopy(base_config)
        effective_config.update(copy.deepcopy(profile))
        mode = str(effective_config.get("mode") or "").strip()
        supported_modes = {
            "gatev3_persistent",
            "memory_off",
            "delayed_task_level_memory_injection",
            "task_start_ttl_then_stuck_reactivation",
        }
        if mode not in supported_modes:
            raise ValueError(
                f"Unsupported task_type_policy mode {mode!r} in profile {selected_profile!r}"
            )
        return {
            "effective_config": effective_config,
            "metadata": {
                "task_type_policy_enabled": True,
                "raw_task_type": str(task_type or ""),
                "detected_task_type": normalized_task_type,
                "selected_intervention_policy": selected_profile,
                "default_intervention_policy": default_profile,
                "route_matched": route_matched,
                "selected_profile_differs_from_default": selected_profile != default_profile,
                "policy_resolution_source": resolution_source,
            },
        }

    def _activate_effective_intervention_policy(self, task_type: Optional[str]) -> None:
        resolved = self._resolve_task_type_intervention_policy(task_type)
        self.gmemory_need_aware_intervention_config = resolved["effective_config"]
        self.gmemory_task_type_policy_metadata = resolved["metadata"]

    def _need_aware_intervention_enabled(self) -> bool:
        return bool(self.gmemory_need_aware_intervention_config.get("enabled", False))

    def _need_aware_intervention_mode(self) -> str:
        return str(self.gmemory_need_aware_intervention_config.get("mode", "disabled")).strip() or "disabled"

    def _delayed_task_level_intervention_enabled(self) -> bool:
        return (
            self._need_aware_intervention_enabled()
            and self._need_aware_intervention_mode() == "delayed_task_level_memory_injection"
        )

    def _memory_off_intervention_enabled(self) -> bool:
        return (
            self._need_aware_intervention_enabled()
            and self._need_aware_intervention_mode() == "memory_off"
        )

    def _need_aware_task_start_ttl_enabled(self) -> bool:
        return (
            self._need_aware_intervention_enabled()
            and self._need_aware_intervention_mode() == "task_start_ttl_then_stuck_reactivation"
        )

    def _need_aware_prompt_scheduling_enabled(self) -> bool:
        return self._delayed_task_level_intervention_enabled() or self._need_aware_task_start_ttl_enabled()

    def _intervention_visibility_ttl(self) -> int:
        return max(1, int(self.gmemory_need_aware_intervention_config.get("visibility_ttl", 3)))

    def _stuck_visibility_ttl(self) -> int:
        return max(
            1,
            int(
                self.gmemory_need_aware_intervention_config.get(
                    "stuck_visibility_ttl",
                    self._intervention_visibility_ttl(),
                )
            ),
        )

    def _task_start_visibility_ttl(self) -> int:
        return max(1, int(self.gmemory_need_aware_intervention_config.get("task_start_visibility_ttl", 2)))

    def _intervention_reentry_cooldown(self) -> int:
        return max(
            0,
            int(
                self.gmemory_need_aware_intervention_config.get(
                    "reentry_cooldown_after_clear",
                    self._intervention_visibility_ttl(),
                )
            ),
        )

    def _task_start_reentry_cooldown(self) -> int:
        return max(
            0,
            int(
                self.gmemory_need_aware_intervention_config.get(
                    "reentry_cooldown_after_task_start_clear",
                    self._task_start_visibility_ttl(),
                )
            ),
        )

    def _intervention_stale_threshold(self) -> int:
        return max(0, int(self.gmemory_need_aware_intervention_config.get("stale_steps_since_last_progress", 8)))

    def _intervention_failure_observation_threshold(self) -> int:
        return max(0, int(self.gmemory_need_aware_intervention_config.get("failure_observation_threshold", 2)))

    def _intervention_requires_check_valid_actions(self) -> bool:
        return bool(self.gmemory_need_aware_intervention_config.get("require_check_valid_actions_since_progress", True))

    def _intervention_failure_signal_policy(self) -> str:
        return str(
            self.gmemory_need_aware_intervention_config.get(
                "failure_signal_policy",
                "nothing_happens_or_query_action_loop",
            )
        ).strip()

    def _query_action_loop_count_threshold(self) -> int:
        return max(0, int(self.gmemory_need_aware_intervention_config.get("query_action_loop_count_threshold", 3)))

    def _query_action_loop_ratio_threshold(self) -> float:
        return max(0.0, min(1.0, float(self.gmemory_need_aware_intervention_config.get("query_action_loop_ratio_threshold", 0.6))))

    def _suppress_ineffective_reactivation_enabled(self) -> bool:
        return bool(self.gmemory_need_aware_intervention_config.get("suppress_ineffective_reactivation", False))

    def _reactivation_effect_window(self) -> str:
        window = str(self.gmemory_need_aware_intervention_config.get("reactivation_effect_window", "ttl_plus_3")).strip()
        return window if window in {"ttl", "ttl_plus_3", "post"} else "ttl_plus_3"

    def _max_consecutive_ineffective_reactivations(self) -> int:
        return max(1, int(self.gmemory_need_aware_intervention_config.get("max_consecutive_ineffective_reactivations", 2)))

    def _allow_late_reactivation_after_new_signal(self) -> bool:
        return bool(self.gmemory_need_aware_intervention_config.get("allow_late_reactivation_after_new_signal", True))

    def _reset_intervention_runtime_state(self) -> None:
        self.gmemory_retrieve_step = None
        self.gmemory_visible_ttl_remaining = None
        self.gmemory_cooldown_remaining = 0
        self.gmemory_stale_steps_since_last_progress = 0
        self.gmemory_nothing_happens_since_last_progress = 0
        self.gmemory_check_valid_actions_since_last_progress = 0
        self.gmemory_inventory_since_last_progress = 0
        self.gmemory_last_intervention_decision = {}
        self.gmemory_injection_events = []
        self.gmemory_clear_events = []
        self.gmemory_active_injection_index = None
        self.gmemory_interaction_steps = 0
        self.gmemory_exposure_steps_total = 0
        self.gmemory_stuck_trigger_count = 0
        self.gmemory_cooldown_block_count = 0
        self.gmemory_trigger_no_cached_memory_count = 0
        self.gmemory_trigger_no_usable_memory_count = 0
        self.gmemory_ineffective_reactivation_streak = 0
        self.gmemory_effective_reactivation_count = 0
        self.gmemory_suppressed_reactivation_count = 0
        self.gmemory_last_ineffective_reactivation_signal = None
        self.gmemory_last_suppression_reason = ""

    def _effective_intervention_config_snapshot(self) -> Dict[str, Any]:
        keys = (
            "enabled",
            "mode",
            "visibility_ttl",
            "stuck_visibility_ttl",
            "task_start_visibility_ttl",
            "reentry_cooldown_after_clear",
            "reentry_cooldown_after_task_start_clear",
            "stale_steps_since_last_progress",
            "failure_observation_threshold",
            "require_check_valid_actions_since_progress",
            "failure_signal_policy",
            "query_action_loop_count_threshold",
            "query_action_loop_ratio_threshold",
            "refresh_ttl_on_progress",
            "clear_on_progress",
            "suppress_ineffective_reactivation",
            "reactivation_effect_window",
            "max_consecutive_ineffective_reactivations",
            "allow_late_reactivation_after_new_signal",
        )
        return {
            key: copy.deepcopy(self.gmemory_need_aware_intervention_config[key])
            for key in keys
            if key in self.gmemory_need_aware_intervention_config
        }

    def _intervention_policy_diagnostics(self) -> Dict[str, Any]:
        metadata = copy.deepcopy(
            getattr(self, "gmemory_task_type_policy_metadata", self._default_task_type_policy_metadata())
        )
        metadata.update(
            {
                "effective_mode": self._need_aware_intervention_mode(),
                "effective_stale_steps_since_last_progress": self._intervention_stale_threshold(),
                "effective_visibility_ttl": self._intervention_visibility_ttl(),
                "effective_stuck_visibility_ttl": self._stuck_visibility_ttl(),
                "effective_task_start_visibility_ttl": self._task_start_visibility_ttl(),
                "effective_reentry_cooldown_after_clear": self._intervention_reentry_cooldown(),
                "effective_reentry_cooldown_after_task_start_clear": self._task_start_reentry_cooldown(),
                "effective_failure_observation_threshold": self._intervention_failure_observation_threshold(),
                "effective_require_check_valid_actions_since_progress": self._intervention_requires_check_valid_actions(),
                "effective_failure_signal_policy": self._intervention_failure_signal_policy(),
                "effective_suppress_ineffective_reactivation": self._suppress_ineffective_reactivation_enabled(),
                "effective_reactivation_effect_window": self._reactivation_effect_window(),
                "effective_max_consecutive_ineffective_reactivations": self._max_consecutive_ineffective_reactivations(),
                "effective_allow_late_reactivation_after_new_signal": self._allow_late_reactivation_after_new_signal(),
                "effective_config": self._effective_intervention_config_snapshot(),
            }
        )
        return metadata

    def _empty_intervention_diagnostics(self) -> Dict[str, Any]:
        diagnostics = {
            "enabled": self._need_aware_intervention_enabled()
            if hasattr(self, "gmemory_need_aware_intervention_config")
            else False,
            "mode": self._need_aware_intervention_mode()
            if hasattr(self, "gmemory_need_aware_intervention_config")
            else "disabled",
            "retrieved": False,
            "cached": False,
            "injected": False,
            "visible": False,
            "retrieved_but_not_injected": False,
            "retrieve_step": None,
            "injection_events": [],
            "clear_events": [],
            "current_visible_ttl_remaining": None,
            "current_cooldown_remaining": None,
            "cached_memory_chars": 0,
            "visible_memory_chars": 0,
            "stale_steps_since_last_progress": 0,
            "nothing_happens_count_since_last_progress": 0,
            "check_valid_actions_count_since_last_progress": 0,
            "inventory_count_since_last_progress": 0,
            "query_action_count_since_last_progress": 0,
            "query_action_ratio_since_last_progress": 0.0,
            "failure_signal_policy": "nothing_happens_or_query_action_loop",
            "nothing_happens_trigger": False,
            "query_action_loop_trigger": False,
            "trigger_reason": "",
            "ineffective_reactivation_streak": 0,
            "effective_reactivation_count": 0,
            "suppressed_reactivation_count": 0,
            "suppression_reason": "",
            "reactivation_effect_window": "ttl_plus_3",
            "last_decision": {},
            "metrics": {},
        }
        if hasattr(self, "gmemory_need_aware_intervention_config"):
            diagnostics.update(self._intervention_policy_diagnostics())
        return diagnostics

    def _set_gmemory_prompt_state(
        self,
        cached_prompt: str,
        visible_prompt: str,
        retrieve_step: Optional[int],
        reset_events: bool = False,
    ) -> None:
        self.cached_gmemory_prompt = cached_prompt or ""
        self.visible_gmemory_prompt = visible_prompt or ""
        self.gmemory_prompt = self.visible_gmemory_prompt
        self.gmemory_retrieve_step = retrieve_step
        if reset_events:
            self.gmemory_injection_events = []
            self.gmemory_clear_events = []
            self.gmemory_active_injection_index = None
        if self.visible_gmemory_prompt and not self._need_aware_prompt_scheduling_enabled() and not self.gmemory_injection_events:
            self.gmemory_injection_events.append(
                {
                    "step": retrieve_step,
                    "reason": "reset_time_immediate",
                    "phase": "task_start_persistent",
                    "memory_chars": len(self.visible_gmemory_prompt),
                    "progress_at_injection": None,
                    "post_injection_progress_delta": 0.0,
                    "visible_ttl": None,
                    "trigger_condition_satisfied": False,
                }
            )
        self._refresh_intervention_diagnostics()

    def _refresh_intervention_diagnostics(self) -> None:
        diagnostics = self._empty_intervention_diagnostics()
        diagnostics.update(
            {
                "retrieved": self.gmemory_retrieve_step is not None,
                "cached": bool(self.cached_gmemory_prompt),
                "injected": bool(self.visible_gmemory_prompt),
                "visible": bool(self.visible_gmemory_prompt),
                "retrieved_but_not_injected": bool(self.cached_gmemory_prompt)
                and not bool(self.visible_gmemory_prompt),
                "retrieve_step": self.gmemory_retrieve_step,
                "injection_events": list(self.gmemory_injection_events),
                "clear_events": list(self.gmemory_clear_events),
                "current_visible_ttl_remaining": self.gmemory_visible_ttl_remaining,
                "current_cooldown_remaining": self.gmemory_cooldown_remaining,
                "cached_memory_chars": len(self.cached_gmemory_prompt),
                "visible_memory_chars": len(self.visible_gmemory_prompt),
                "stale_steps_since_last_progress": self.gmemory_stale_steps_since_last_progress,
                "nothing_happens_count_since_last_progress": self.gmemory_nothing_happens_since_last_progress,
                "check_valid_actions_count_since_last_progress": self.gmemory_check_valid_actions_since_last_progress,
                "inventory_count_since_last_progress": self.gmemory_inventory_since_last_progress,
                "query_action_count_since_last_progress": self._query_action_count_since_last_progress(),
                "query_action_ratio_since_last_progress": self._query_action_ratio_since_last_progress(),
                "failure_signal_policy": self._intervention_failure_signal_policy(),
                "nothing_happens_trigger": self._nothing_happens_trigger_satisfied(),
                "query_action_loop_trigger": self._query_action_loop_trigger_satisfied(),
                "trigger_reason": self._current_trigger_reason(),
                "ineffective_reactivation_streak": self.gmemory_ineffective_reactivation_streak,
                "effective_reactivation_count": self.gmemory_effective_reactivation_count,
                "suppressed_reactivation_count": self.gmemory_suppressed_reactivation_count,
                "suppression_reason": self.gmemory_last_suppression_reason,
                "reactivation_effect_window": self._reactivation_effect_window(),
                "last_decision": dict(self.gmemory_last_intervention_decision),
                "metrics": self._intervention_episode_metrics(),
            }
        )
        self.gmemory_intervention_diagnostics = diagnostics

    def _intervention_episode_metrics(self) -> Dict[str, Any]:
        injection_count = len(self.gmemory_injection_events)
        clear_count = len(self.gmemory_clear_events)
        ttl_expired_count = sum(1 for event in self.gmemory_clear_events if event.get("reason") == "ttl_expired")
        task_start_ttl_expired_count = sum(
            1 for event in self.gmemory_clear_events if event.get("reason") == "task_start_ttl_expired"
        )
        stuck_ttl_expired_count = sum(
            1
            for event in self.gmemory_clear_events
            if event.get("reason") == "ttl_expired" and event.get("phase") == "stuck"
        )
        post_deltas = [
            float(event.get("post_injection_progress_delta") or 0.0)
            for event in self.gmemory_injection_events
        ]
        task_start_deltas = [
            float(event.get("post_injection_progress_delta") or 0.0)
            for event in self.gmemory_injection_events
            if event.get("phase") == "task_start"
        ]
        stuck_deltas = [
            float(event.get("post_injection_progress_delta") or 0.0)
            for event in self.gmemory_injection_events
            if event.get("phase") == "stuck"
        ]
        task_start_injection_count = sum(1 for event in self.gmemory_injection_events if event.get("phase") == "task_start")
        stuck_reactivation_count = sum(1 for event in self.gmemory_injection_events if event.get("phase") == "stuck")
        first_stuck_reactivation_step = next(
            (event.get("step") for event in self.gmemory_injection_events if event.get("phase") == "stuck"),
            None,
        )
        recovered_events = sum(1 for delta in post_deltas if delta > 0)
        retrieved_but_not_injected = bool(self.cached_gmemory_prompt) and injection_count == 0
        return {
            "memory_injection_count": injection_count,
            "memory_injection_rate": 1.0 if injection_count > 0 else 0.0,
            "task_start_injection_count": task_start_injection_count,
            "stuck_reactivation_count": stuck_reactivation_count,
            "stuck_trigger_count": self.gmemory_stuck_trigger_count,
            "retrieved_but_not_injected_count": 1 if retrieved_but_not_injected else 0,
            "post_injection_progress_delta": max(post_deltas) if post_deltas else None,
            "post_task_start_progress_delta": max(task_start_deltas) if task_start_deltas else None,
            "post_stuck_reactivation_progress_delta": max(stuck_deltas) if stuck_deltas else None,
            "recovery_success_rate": recovered_events / injection_count if injection_count else None,
            "memory_harm_proxy": any(delta < 0 for delta in post_deltas),
            "ttl_expired_count": ttl_expired_count,
            "task_start_ttl_expired_count": task_start_ttl_expired_count,
            "stuck_ttl_expired_count": stuck_ttl_expired_count,
            "cooldown_block_count": self.gmemory_cooldown_block_count,
            "trigger_no_cached_memory_count": self.gmemory_trigger_no_cached_memory_count,
            "trigger_no_usable_memory_count": self.gmemory_trigger_no_usable_memory_count,
            "ineffective_reactivation_streak": self.gmemory_ineffective_reactivation_streak,
            "effective_reactivation_count": self.gmemory_effective_reactivation_count,
            "suppressed_reactivation_count": self.gmemory_suppressed_reactivation_count,
            "visible_clear_count": clear_count,
            "first_stuck_reactivation_step": first_stuck_reactivation_step,
            "memory_exposure_steps_total": self.gmemory_exposure_steps_total,
            "memory_exposure_rate": (
                self.gmemory_exposure_steps_total / self.gmemory_interaction_steps
                if self.gmemory_interaction_steps
                else 0.0
            ),
        }

    def _query_action_count_since_last_progress(self) -> int:
        return self.gmemory_check_valid_actions_since_last_progress + self.gmemory_inventory_since_last_progress

    def _query_action_ratio_since_last_progress(self) -> float:
        if self.gmemory_stale_steps_since_last_progress <= 0:
            return 0.0
        return self._query_action_count_since_last_progress() / float(self.gmemory_stale_steps_since_last_progress)

    def _nothing_happens_trigger_satisfied(self) -> bool:
        check_valid_actions_ok = (
            self.gmemory_check_valid_actions_since_last_progress > 0
            if self._intervention_requires_check_valid_actions()
            else True
        )
        return (
            self.gmemory_nothing_happens_since_last_progress
            >= self._intervention_failure_observation_threshold()
            and check_valid_actions_ok
        )

    def _query_action_loop_trigger_satisfied(self) -> bool:
        return (
            self._query_action_count_since_last_progress() >= self._query_action_loop_count_threshold()
            and self._query_action_ratio_since_last_progress() >= self._query_action_loop_ratio_threshold()
        )

    def _current_trigger_reason(self) -> str:
        nothing_happens_trigger = self._nothing_happens_trigger_satisfied()
        query_action_loop_trigger = self._query_action_loop_trigger_satisfied()
        if nothing_happens_trigger and query_action_loop_trigger:
            return "nothing_happens+query_action_loop"
        if nothing_happens_trigger:
            return "nothing_happens"
        if query_action_loop_trigger:
            return "query_action_loop"
        return ""

    def _intervention_trigger_condition_satisfied(self) -> bool:
        if self.gmemory_stale_steps_since_last_progress < self._intervention_stale_threshold():
            return False
        policy = self._intervention_failure_signal_policy()
        nothing_happens_trigger = self._nothing_happens_trigger_satisfied()
        query_action_loop_trigger = self._query_action_loop_trigger_satisfied()
        if policy == "nothing_happens_only":
            return self.gmemory_nothing_happens_since_last_progress >= self._intervention_failure_observation_threshold()
        if policy == "nothing_happens_and_check":
            return nothing_happens_trigger
        if policy == "query_action_loop":
            return query_action_loop_trigger
        if policy == "nothing_happens_or_query_action_loop":
            return nothing_happens_trigger or query_action_loop_trigger
        return nothing_happens_trigger

    def _record_intervention_decision(
        self,
        trigger_condition_satisfied: bool,
        intervention_allowed: bool,
        would_trigger: bool,
        skip_reason: str,
        memory_visible: bool,
        in_reentry_cooldown: bool,
    ) -> None:
        self.gmemory_last_intervention_decision = {
            "trigger_condition_satisfied": trigger_condition_satisfied,
            "intervention_allowed": intervention_allowed,
            "would_trigger": would_trigger,
            "skip_reason": skip_reason,
            "memory_visible": memory_visible,
            "in_reentry_cooldown": in_reentry_cooldown,
            "ttl_remaining": self.gmemory_visible_ttl_remaining,
            "cooldown_remaining": self.gmemory_cooldown_remaining,
        }

    def _reactivation_signal(self, executed_action: str, observation: str) -> str:
        trigger_reason = self._current_trigger_reason()
        action = str(executed_action or "").strip().lower()
        obs = re.sub(r"\s+", " ", str(observation or "").strip().lower())
        if action == "check valid actions":
            marker = "choose an action from these valid actions:"
            if marker in obs:
                obs = obs.split(marker, 1)[1]
            actions = [part.strip() for part in obs.split(",") if part.strip()]
            obs = ",".join(sorted(actions))[:500]
        else:
            obs = obs[:240]
        return f"{trigger_reason}|{action}|{obs}"

    def _reactivation_delta_for_window(self, event: Dict[str, Any]) -> float:
        window = self._reactivation_effect_window()
        if window == "ttl":
            return float(event.get("delta_within_ttl") or 0.0)
        if window == "post":
            return float(event.get("post_injection_progress_delta") or 0.0)
        return float(event.get("delta_within_ttl_plus_3") or 0.0)

    def _reactivation_effect_window_closed(self, event: Dict[str, Any], step_id: int) -> bool:
        injection_step = event.get("step")
        if injection_step is None:
            return False
        visible_ttl = int(event.get("visible_ttl") or 0)
        if self._reactivation_effect_window() == "ttl":
            return step_id > int(injection_step) + visible_ttl
        if self._reactivation_effect_window() == "post":
            return True
        return step_id > int(injection_step) + visible_ttl + 3

    def _update_reactivation_effect_state(self, step_id: int) -> None:
        for event in self.gmemory_injection_events:
            if event.get("phase") != "stuck" or event.get("reactivation_effect_evaluated"):
                continue
            if not self._reactivation_effect_window_closed(event, step_id):
                continue
            event["effective_within_ttl"] = float(event.get("delta_within_ttl") or 0.0) > 0.0
            event["effective_within_ttl_plus_3"] = float(event.get("delta_within_ttl_plus_3") or 0.0) > 0.0
            event["reactivation_effect_window"] = self._reactivation_effect_window()
            event["reactivation_effective"] = self._reactivation_delta_for_window(event) > 0.0
            event["reactivation_effect_evaluated"] = True
            if event["reactivation_effective"]:
                self.gmemory_ineffective_reactivation_streak = 0
                self.gmemory_effective_reactivation_count += 1
                self.gmemory_last_ineffective_reactivation_signal = None
            else:
                self.gmemory_ineffective_reactivation_streak += 1
                self.gmemory_last_ineffective_reactivation_signal = event.get("reactivation_signal")

    def _should_suppress_reactivation(self, current_signal: str) -> Optional[str]:
        if not self._suppress_ineffective_reactivation_enabled():
            return None
        if self.gmemory_ineffective_reactivation_streak < self._max_consecutive_ineffective_reactivations():
            return None
        if (
            self._allow_late_reactivation_after_new_signal()
            and current_signal
            and self.gmemory_last_ineffective_reactivation_signal
            and current_signal != self.gmemory_last_ineffective_reactivation_signal
        ):
            return None
        return "consecutive_ineffective_reactivation"

    def _inject_cached_gmemory(
        self,
        step_id: int,
        progress_rate: float,
        trigger_condition_satisfied: bool,
        reason: str = "stuck_trigger",
        phase: str = "stuck",
        visible_ttl: Optional[int] = None,
        cooldown: Optional[int] = None,
        reactivation_signal: Optional[str] = None,
    ) -> None:
        self.visible_gmemory_prompt = self.cached_gmemory_prompt
        self.gmemory_prompt = self.visible_gmemory_prompt
        effective_ttl = visible_ttl if visible_ttl is not None else self._stuck_visibility_ttl()
        effective_cooldown = cooldown if cooldown is not None else self._intervention_reentry_cooldown()
        self.gmemory_visible_ttl_remaining = effective_ttl
        event = {
            "step": step_id,
            "reason": reason,
            "phase": phase,
            "memory_chars": len(self.visible_gmemory_prompt),
            "progress_at_injection": progress_rate,
            "post_injection_progress_delta": 0.0,
            "delta_within_ttl": 0.0,
            "delta_within_ttl_plus_3": 0.0,
            "eventual_delta_after_injection": 0.0,
            "progress_during_visibility_count": 0,
            "visible_ttl": self.gmemory_visible_ttl_remaining,
            "cooldown": effective_cooldown,
            "trigger_condition_satisfied": trigger_condition_satisfied,
            "stale_steps_since_last_progress": self.gmemory_stale_steps_since_last_progress,
            "nothing_happens_count_since_last_progress": self.gmemory_nothing_happens_since_last_progress,
            "check_valid_actions_count_since_last_progress": self.gmemory_check_valid_actions_since_last_progress,
            "inventory_count_since_last_progress": self.gmemory_inventory_since_last_progress,
            "query_action_count_since_last_progress": self._query_action_count_since_last_progress(),
            "query_action_ratio_since_last_progress": self._query_action_ratio_since_last_progress(),
            "trigger_reason": self._current_trigger_reason(),
            "reactivation_signal": reactivation_signal,
            "reactivation_effect_window": self._reactivation_effect_window(),
            "reactivation_effective": False,
            "reactivation_effect_evaluated": False,
            "effective_within_ttl": False,
            "effective_within_ttl_plus_3": False,
            "gate_kept_count": self.gmemory_gate_diagnostics.get("kept_count"),
            "gate_dropped_count": self.gmemory_gate_diagnostics.get("dropped_count"),
        }
        self.gmemory_injection_events.append(event)
        self.gmemory_active_injection_index = len(self.gmemory_injection_events) - 1

    def _update_injection_progress_metrics(self, step_id: int, progress_rate: float, progress_improved: bool) -> None:
        for idx, event in enumerate(self.gmemory_injection_events):
            injection_step = event.get("step")
            if injection_step is None:
                continue
            progress_at_injection = float(event.get("progress_at_injection") or 0.0)
            delta = progress_rate - progress_at_injection
            visible_ttl = int(event.get("visible_ttl") or 0)
            event["eventual_delta_after_injection"] = delta
            event["post_injection_progress_delta"] = max(float(event.get("post_injection_progress_delta") or 0.0), delta)
            if step_id <= injection_step + visible_ttl:
                event["delta_within_ttl"] = max(float(event.get("delta_within_ttl") or 0.0), delta)
                if progress_improved:
                    event["progress_during_visibility_count"] = int(event.get("progress_during_visibility_count") or 0) + 1
            if step_id <= injection_step + visible_ttl + 3:
                event["delta_within_ttl_plus_3"] = max(float(event.get("delta_within_ttl_plus_3") or 0.0), delta)
            if idx == self.gmemory_active_injection_index:
                event["post_injection_progress_delta"] = delta

    def _clear_visible_gmemory(self, step_id: int, reason: str, progress_rate: float) -> None:
        if not self.visible_gmemory_prompt:
            return
        progress_delta = 0.0
        event = {}
        if self.gmemory_active_injection_index is not None:
            event = self.gmemory_injection_events[self.gmemory_active_injection_index]
            progress_delta = progress_rate - float(event.get("progress_at_injection") or 0.0)
            event["post_injection_progress_delta"] = progress_delta
        self.gmemory_clear_events.append(
            {
                "step": step_id,
                "reason": reason,
                "phase": event.get("phase") if event else None,
                "progress_delta_since_injection": progress_delta,
                "visible_steps": step_id - int(event.get("step") or step_id) if self.gmemory_active_injection_index is not None else 0,
                "progress_during_visibility_count": int(event.get("progress_during_visibility_count") or 0)
                if self.gmemory_active_injection_index is not None
                else 0,
            }
        )
        self.visible_gmemory_prompt = ""
        self.gmemory_prompt = ""
        self.gmemory_visible_ttl_remaining = None
        self.gmemory_cooldown_remaining = int(event.get("cooldown") or self._intervention_reentry_cooldown()) if event else self._intervention_reentry_cooldown()
        self.gmemory_active_injection_index = None

    def update_intervention_state(
        self,
        step_id: int,
        executed_action: str,
        observation: str,
        progress_rate: float,
        previous_progress_rate: float,
        is_valid_action: bool,
        nothing_happens: bool,
        is_check_valid_actions: bool,
    ) -> None:
        self.gmemory_interaction_steps += 1
        memory_visible_at_step = bool(self.visible_gmemory_prompt)
        if memory_visible_at_step:
            self.gmemory_exposure_steps_total += 1
        if not self._need_aware_prompt_scheduling_enabled():
            self._refresh_intervention_diagnostics()
            return

        progress_improved = progress_rate > previous_progress_rate
        if progress_improved:
            self.gmemory_stale_steps_since_last_progress = 0
            self.gmemory_nothing_happens_since_last_progress = 0
            self.gmemory_check_valid_actions_since_last_progress = 0
            self.gmemory_inventory_since_last_progress = 0
            self.gmemory_ineffective_reactivation_streak = 0
            self.gmemory_last_ineffective_reactivation_signal = None
            if self.gmemory_active_injection_index is not None:
                event = self.gmemory_injection_events[self.gmemory_active_injection_index]
                event["post_injection_progress_delta"] = progress_rate - float(event.get("progress_at_injection") or 0.0)
        else:
            self.gmemory_stale_steps_since_last_progress += 1
            if nothing_happens:
                self.gmemory_nothing_happens_since_last_progress += 1
            if is_check_valid_actions:
                self.gmemory_check_valid_actions_since_last_progress += 1
            if str(executed_action or "").strip() == "inventory":
                self.gmemory_inventory_since_last_progress += 1

        self._update_injection_progress_metrics(step_id, progress_rate, progress_improved)
        self._update_reactivation_effect_state(step_id)
        trigger_condition_satisfied = self._intervention_trigger_condition_satisfied()
        trigger_reason = self._current_trigger_reason() if trigger_condition_satisfied else ""
        in_reentry_cooldown = self.gmemory_cooldown_remaining > 0
        if trigger_condition_satisfied:
            self.gmemory_stuck_trigger_count += 1
        if memory_visible_at_step:
            self._record_intervention_decision(
                trigger_condition_satisfied=trigger_condition_satisfied,
                intervention_allowed=False,
                would_trigger=False,
                skip_reason="memory_visible",
                memory_visible=True,
                in_reentry_cooldown=in_reentry_cooldown,
            )
        elif in_reentry_cooldown:
            self._record_intervention_decision(
                trigger_condition_satisfied=trigger_condition_satisfied,
                intervention_allowed=False,
                would_trigger=trigger_condition_satisfied,
                skip_reason="reentry_cooldown",
                memory_visible=False,
                in_reentry_cooldown=True,
            )
            if trigger_condition_satisfied:
                self.gmemory_cooldown_block_count += 1
        elif trigger_condition_satisfied:
            if self.gmemory_retrieve_step is None:
                skip_reason = "no_cached_memory"
                allowed = False
                self.gmemory_trigger_no_cached_memory_count += 1
            elif not self.cached_gmemory_prompt:
                skip_reason = "no_usable_memory"
                allowed = False
                self.gmemory_trigger_no_usable_memory_count += 1
            else:
                reactivation_signal = self._reactivation_signal(executed_action, observation)
                suppression_reason = self._should_suppress_reactivation(reactivation_signal)
                if suppression_reason:
                    skip_reason = suppression_reason
                    allowed = False
                    self.gmemory_suppressed_reactivation_count += 1
                    self.gmemory_last_suppression_reason = suppression_reason
                else:
                    skip_reason = "none"
                    allowed = True
                    self.gmemory_last_suppression_reason = ""
                    self._inject_cached_gmemory(
                        step_id,
                        progress_rate,
                        trigger_condition_satisfied,
                        reason="stuck_reactivation" if self._need_aware_task_start_ttl_enabled() else "stuck_trigger",
                        phase="stuck",
                        visible_ttl=self._stuck_visibility_ttl(),
                        cooldown=self._intervention_reentry_cooldown(),
                        reactivation_signal=reactivation_signal,
                    )
            self._record_intervention_decision(
                trigger_condition_satisfied=trigger_condition_satisfied,
                intervention_allowed=allowed,
                would_trigger=trigger_condition_satisfied,
                skip_reason=skip_reason,
                memory_visible=False,
                in_reentry_cooldown=False,
            )
        else:
            self._record_intervention_decision(
                trigger_condition_satisfied=False,
                intervention_allowed=False,
                would_trigger=False,
                skip_reason="none",
                memory_visible=False,
                in_reentry_cooldown=False,
            )
        if self.gmemory_last_intervention_decision is not None:
            self.gmemory_last_intervention_decision["trigger_reason"] = trigger_reason

        if memory_visible_at_step and self.gmemory_visible_ttl_remaining is not None:
            self.gmemory_visible_ttl_remaining = max(0, self.gmemory_visible_ttl_remaining - 1)
            if self.gmemory_visible_ttl_remaining == 0:
                active_event = (
                    self.gmemory_injection_events[self.gmemory_active_injection_index]
                    if self.gmemory_active_injection_index is not None
                    else {}
                )
                clear_reason = "task_start_ttl_expired" if active_event.get("phase") == "task_start" else "ttl_expired"
                self._clear_visible_gmemory(step_id, clear_reason, progress_rate)
        elif not memory_visible_at_step and self.gmemory_cooldown_remaining > 0:
            self.gmemory_cooldown_remaining = max(0, self.gmemory_cooldown_remaining - 1)

        if self.gmemory_last_intervention_decision:
            self.gmemory_last_intervention_decision["ttl_remaining"] = self.gmemory_visible_ttl_remaining
            self.gmemory_last_intervention_decision["cooldown_remaining"] = self.gmemory_cooldown_remaining
        self._refresh_intervention_diagnostics()

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

        if self._has_puttwo_state_workflow_pollution(contract, text):
            reasons.append("puttwo_state_workflow_pollution")

        if self._has_puttwo_weak_cardinality_signal(contract, text):
            reasons.append("puttwo_weak_cardinality_signal")

        if self._has_look_workflow_pollution(contract, text):
            reasons.append("look_workflow_pollution")

        if self._has_obvious_verification_loop_risk(text):
            reasons.append("obvious_verification_loop_risk")

        broad_over_verification = self._assess_broad_over_verification_risk(text)
        if broad_over_verification == "drop":
            reasons.append("broad_over_verification_workflow_pollution")
        elif broad_over_verification == "diagnostic":
            diagnostic_reasons.append("broad_over_verification_workflow_pollution")

        if self._has_clean_heat_cool_missing_finalization_signal(contract, text):
            if self._goal_contract_gate_state_finalization_action() == "drop":
                reasons.append("missing_finalization_signal")
            else:
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

    def _is_puttwo_contract(self, contract: Dict[str, Any]) -> bool:
        return contract.get("task_type") == "puttwo" or contract.get("count") in {"two", "multiple"}

    def _has_puttwo_cardinality_signal(self, text: str) -> bool:
        cardinality_terms = [
            "two",
            "both",
            "second",
            "another",
            "remaining",
            "repeat",
            "each",
            "all",
            "until all",
            "count",
        ]
        return self._contains_any(text, cardinality_terms)

    def _has_puttwo_finalization_or_target_signal(self, text: str) -> bool:
        puttwo_useful_terms = [
            "target receptacle",
            "target container",
            "target location",
            "destination",
            "final container",
            "final receptacle",
            "return to the target",
            "return to target",
            "go back to the target",
            "put",
            "place",
            "placed",
            "placing",
        ]
        return self._contains_any(text, puttwo_useful_terms)

    def _has_puttwo_state_workflow_pollution(self, contract: Dict[str, Any], text: str) -> bool:
        if not self._is_puttwo_contract(contract):
            return False
        if self._has_puttwo_cardinality_signal(text):
            return False
        state_workflow_terms = [
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
            "property",
            "state",
            "state change",
            "transformation",
            "processed",
            "processing",
            "appliance",
            "device",
        ]
        return self._contains_any(text, state_workflow_terms)

    def _has_puttwo_weak_cardinality_signal(self, contract: Dict[str, Any], text: str) -> bool:
        if not self._is_puttwo_contract(contract):
            return False
        if self._has_puttwo_cardinality_signal(text) or self._has_puttwo_finalization_or_target_signal(text):
            return False
        weak_single_object_patterns = [
            r"\bexact target object\b",
            r"\btarget object\b",
            r"\bacquire\b",
            r"\bpick up\b",
            r"\bretrieve\b",
            r"\bcurrent location\b",
            r"\bits current location\b",
            r"\bmanipulation\b",
            r"\bmanipulate\b",
            r"\bopen (?:a |the )?(?:container|cabinet|drawer)\b",
            r"\bcontainers? before\b",
            r"\bcheck\b",
            r"\bverify\b",
            r"\bconfirm\b",
            r"\bgroup consecutive actions\b",
            r"\bsame location\b",
        ]
        return any(re.search(pattern, text) for pattern in weak_single_object_patterns)

    def _has_look_workflow_pollution(self, contract: Dict[str, Any], text: str) -> bool:
        if contract.get("task_type") != "look":
            return False
        if self._has_look_state_workflow_signal(text):
            return True
        return self._has_look_final_target_routine(text)

    def _has_look_state_workflow_signal(self, text: str) -> bool:
        state_terms = [
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
            "stoveburner",
            "state change",
            "processed",
            "processing",
            "transformation",
            "appliance",
            "device readiness",
        ]
        return self._contains_any(text, state_terms)

    def _has_look_final_target_routine(self, text: str) -> bool:
        if not self._contains_any(text, ["put", "place", "placing", "placed"]):
            return False
        final_target_terms = [
            "final location",
            "final target",
            "target receptacle",
            "target container",
            "target location",
            "destination",
            "receptacle",
            "container",
            "cabinet",
            "countertop",
            "safe",
            "sofa",
            "garbagecan",
            "toilet",
            "shelf",
        ]
        return self._contains_any(text, final_target_terms)

    def _has_obvious_verification_loop_risk(self, text: str) -> bool:
        if self._contains_any(text, ["repeated", "repeatedly", "again", "loop", "nothing happens"]):
            return True
        probe_terms = ["check", "examine", "inventory"]
        return any(len(re.findall(r"\b" + re.escape(term) + r"\b", text)) >= 2 for term in probe_terms)

    def _assess_broad_over_verification_risk(self, text: str) -> str:
        if not self._has_broad_over_verification_scope(text) or not self._has_verification_verb(text):
            return "keep"
        if self._has_concrete_task_advancing_action(text):
            return "diagnostic"
        return "drop"

    def _has_broad_over_verification_scope(self, text: str) -> bool:
        patterns = [
            r"\bbefore (?:each|every) action\b",
            r"\bafter (?:each|every) action\b",
            r"\bbefore (?:each|every) step\b",
            r"\bafter (?:each|every) step\b",
            r"\bat (?:each|every) step\b",
            r"\bevery step\b",
            r"\beach step\b",
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    def _has_verification_verb(self, text: str) -> bool:
        return self._contains_any(text, ["verify", "confirm", "check", "ensure"])

    def _has_concrete_task_advancing_action(self, text: str) -> bool:
        concrete_terms = [
            "take",
            "pick up",
            "pickup",
            "put",
            "place",
            "placing",
            "placed",
            "open",
            "close",
            "clean",
            "heat",
            "cool",
            "return",
            "go back",
            "search",
            "find",
            "locate",
            "locates",
            "acquire",
            "acquires",
            "select",
            "selects",
            "apply",
            "applies",
            "retrieve",
            "retrieves",
            "store",
            "stores",
            "move",
            "moves",
        ]
        return self._contains_any(text, concrete_terms) or bool(re.search(r"\bexamine\b.+\bwith\b", text))

    def _has_clean_heat_cool_missing_finalization_signal(self, contract: Dict[str, Any], text: str) -> bool:
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
            and not self._has_correct_state_device_action(contract, text)
            and not self._has_state_action_precondition(text, contract)
        )

    def _has_correct_state_device_action(self, contract: Dict[str, Any], text: str) -> bool:
        task_type = contract.get("task_type")
        if task_type == "clean":
            action_terms = ["clean", "cleaning"]
            device_terms = ["sinkbasin"]
        elif task_type == "heat":
            action_terms = ["heat", "heating", "hot", "cook", "cooking", "warm"]
            device_terms = ["microwave", "stoveburner", "toaster", "coffeemachine"]
        elif task_type == "cool":
            action_terms = ["cool", "cooling", "cold", "chill", "chilling"]
            device_terms = ["fridge", "refrigerator"]
        else:
            return False
        return self._contains_any(text, action_terms) and self._contains_any(text, device_terms)

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
            "gmemory_intervention": self.gmemory_intervention_diagnostics,
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
