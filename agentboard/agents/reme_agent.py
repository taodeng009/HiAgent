"""Context-efficient HiAgent variant with ReMe retrieval/finish hooks."""
from __future__ import annotations

import logging
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Dict, List, Optional

from common.registry import registry

from .cme_final import ContextEfficientAgentV2
from .reme_client import ReMeClient


logger = logging.getLogger(__name__)


@registry.register_agent("ReMeContextEfficientAgent")
class ReMeContextEfficientAgent(ContextEfficientAgentV2):
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
        reme: Optional[Dict[str, Any]] = None,
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
        self.reme_config = reme or {}
        self.reme_enabled = bool(self.reme_config.get("enabled", False))
        self.reme_fail_open = bool(self.reme_config.get("fail_open", True))
        self.reme_workspace_id = self.reme_config.get("workspace_id", "")
        self.reme_retrieve_config = self.reme_config.get("retrieve", {}) or {}
        self.reme_finish_trial_config = self.reme_config.get("finish_trial", {}) or {}
        self.reme_prompt_wrapper_config = self.reme_config.get("prompt_wrapper", {}) or {}
        self.reme_retrieve_enabled = bool(self.reme_retrieve_config.get("enabled", True))
        self.reme_finish_trial_enabled = bool(self.reme_finish_trial_config.get("enabled", False))
        self.reme_client = self._build_reme_client()
        self.reme_memory_prompt = ""
        self.reme_retrieval_id = None
        self.reme_retrieved_memory_ids: List[str] = []
        self.reme_retrieved_memories: List[Dict[str, Any]] = []
        self.reme_diagnostics = self._empty_diagnostics()
        self.reme_finish_submit_index = 0

    def _build_reme_client(self) -> Optional[ReMeClient]:
        if not self.reme_enabled:
            return None
        base_url = self.reme_config.get("base_url")
        if not base_url:
            logger.warning("ReMe is enabled but base_url is incomplete")
            return None
        timeout_config = self.reme_config.get("timeout", {}) or {}
        return ReMeClient(
            base_url=base_url,
            retrieve_timeout=float(timeout_config.get("retrieve", 15.0)),
            finish_trial_timeout=float(timeout_config.get("finish_trial", 120.0)),
        )

    def reset(self, goal, init_obs, init_act=None):
        super().reset(goal, init_obs, init_act)
        self.reme_memory_prompt = ""
        self.reme_retrieval_id = None
        self.reme_retrieved_memory_ids = []
        self.reme_retrieved_memories = []
        self.reme_finish_submit_index = 0
        self.reme_diagnostics = self._empty_diagnostics()
        self.reme_diagnostics["query"] = self._raw_goal_query(goal)
        current_context = self._build_current_context(init_obs)
        self.reme_diagnostics["current_context_chars"] = len(current_context)

        if not self.reme_enabled or not self.reme_retrieve_enabled or self.reme_client is None:
            return
        try:
            response = self.reme_client.retrieve(
                workspace_id=self.reme_workspace_id,
                query=self.reme_diagnostics["query"],
                top_k=int(self.reme_retrieve_config.get("top_k", 5)),
                min_score=self.reme_retrieve_config.get("min_score"),
                rerank=bool(self.reme_retrieve_config.get("rerank", False)),
                rewrite=bool(self.reme_retrieve_config.get("rewrite", False)),
                max_context_chars=int(self.reme_retrieve_config.get("max_context_chars", 3000)),
                current_context=current_context,
            )
            self.reme_memory_prompt = self._normalise_memory_prompt(response.get("memory_prompt", ""))
            self.reme_retrieval_id = response.get("retrieval_id")
            memories = response.get("memories") or []
            self.reme_retrieved_memory_ids = [
                str(memory.get("memory_id"))
                for memory in memories
                if isinstance(memory, dict) and memory.get("memory_id") is not None
            ]
            self.reme_retrieved_memories = self._summarise_retrieved_memories(memories)
            self.reme_diagnostics.update(
                {
                    "retrieve_attempted": True,
                    "retrieve_success": True,
                    "retrieval_id": self.reme_retrieval_id,
                    "retrieved_memory_ids": list(self.reme_retrieved_memory_ids),
                    "retrieved_memories": list(self.reme_retrieved_memories),
                    "memory_prompt_chars": len(self.reme_memory_prompt),
                    "returned_count": len(memories),
                }
            )
            logger.info(
                "ReMe retrieve completed: memory_prompt_chars=%s, scores=%s",
                len(self.reme_memory_prompt),
                [
                    {
                        "memory_id": item.get("memory_id"),
                        "retrieval_score": item.get("retrieval_score"),
                        "validation_score": item.get("validation_score"),
                        "score": item.get("score"),
                    }
                    for item in self.reme_retrieved_memories
                ],
            )
        except Exception as exc:
            self.reme_memory_prompt = ""
            self.reme_diagnostics.update(
                {
                    "retrieve_attempted": True,
                    "retrieve_success": False,
                    "retrieve_error": str(exc),
                }
            )
            logger.warning("ReMe retrieve failed: %s", exc)
            if not self.reme_fail_open:
                raise

    def make_prompt(self, need_goal=False, check_actions="check valid actions", check_inventory="inventory", system_message=''):
        with redirect_stdout(StringIO()):
            prompt = super().make_prompt(
                need_goal=need_goal,
                check_actions=check_actions,
                check_inventory=check_inventory,
                system_message=system_message,
            )
        if self.reme_memory_prompt:
            prompt = self._inject_reme_prompt(prompt)
            self.reme_diagnostics["memory_injected"] = True
            self.reme_diagnostics["injected_prompt_chars"] = len(self.reme_memory_prompt)
            self.reme_diagnostics["wrapped_prompt_chars"] = len(self._build_reme_prompt_block())
        print(f'------------[Prompt Start]-----------\n{prompt}\n----------[Prompt END]------------')
        return prompt

    def _inject_reme_prompt(self, prompt: str) -> str:
        block = f"{self._build_reme_prompt_block()}\n\n"
        marker = self._current_history_marker()
        if marker:
            idx = prompt.find(marker)
            if idx >= 0:
                return prompt[:idx] + block + prompt[idx:]
        return block + prompt

    def _build_reme_prompt_block(self) -> str:
        memory_prompt = self.reme_memory_prompt.strip()
        if not memory_prompt:
            return ""
        if not bool(self.reme_prompt_wrapper_config.get("enabled", True)):
            self.reme_diagnostics["prompt_wrapper_enabled"] = False
            return memory_prompt
        heading = str(
            self.reme_prompt_wrapper_config.get(
                "heading",
                "## Relevant Memories from Related Tasks",
            )
        ).strip()
        instruction = str(
            self.reme_prompt_wrapper_config.get(
                "instruction",
                (
                    "The following memories describe when they are useful and what strategy they suggest. "
                    "You may refer to them during your task execution to improve problem-solving accuracy."
                ),
            )
        ).strip()
        delimiter = str(self.reme_prompt_wrapper_config.get("delimiter", "---")).strip()
        parts = [part for part in [heading, instruction, memory_prompt, delimiter] if part]
        self.reme_diagnostics["prompt_wrapper_enabled"] = True
        return "\n".join(parts)

    def remember_current_task(
        self,
        task_type: str = "",
        success: Optional[bool] = None,
        progress_rate: Optional[float] = None,
        score_change_record=None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not self.reme_enabled or not self.reme_finish_trial_enabled or self.reme_client is None:
            self.reme_diagnostics["finish_trial_skipped"] = True
            self.reme_diagnostics["finish_trial_skip_reason"] = "disabled"
            return None
        if success is None:
            self.reme_diagnostics["finish_trial_skipped"] = True
            self.reme_diagnostics["finish_trial_skip_reason"] = "missing_success"
            logger.warning("ReMe finish-trial skipped: success is missing")
            return None

        trajectory = self._memory_to_reme_trajectory(metadata=metadata)
        outcome = {
            "success": bool(success),
            "score": 1.0 if success else 0.0,
            "progress_rate": self._json_scalar(progress_rate),
        }
        self.reme_finish_submit_index += 1
        trajectory_id = trajectory["trajectory_id"]
        request_id = f"{trajectory_id}:submit:{self.reme_finish_submit_index}"

        try:
            response = self.reme_client.finish_trial(
                workspace_id=self.reme_workspace_id,
                request_id=request_id,
                retrieval_id=self.reme_retrieval_id,
                trajectory=trajectory,
                outcome=outcome,
            )
            self.reme_diagnostics.update(
                {
                    "finish_trial_attempted": True,
                    "finish_trial_success": True,
                    "finish_trial_request_id": request_id,
                    "finish_trial_response": response,
                }
            )
            logger.info("ReMe finish-trial completed: request_id=%s", request_id)
            return response
        except Exception as exc:
            self.reme_diagnostics.update(
                {
                    "finish_trial_attempted": True,
                    "finish_trial_success": False,
                    "finish_trial_request_id": request_id,
                    "finish_trial_error": str(exc),
                }
            )
            logger.warning("ReMe finish-trial failed: %s", exc)
            if not self.reme_fail_open:
                raise
            return None

    def _memory_to_reme_trajectory(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = []
        goal = getattr(self, "goal", None)
        initial_observation = getattr(self, "init_obs", None)
        first_content = f"Goal: {goal or ''}\nInitial observation: {initial_observation or ''}"
        messages.append({"role": "user", "content": first_content})

        for turn in getattr(self, "memory", [])[1:]:
            fields = dict(turn)
            action = fields.get("Action")
            observation = fields.get("Observation")
            if action is not None:
                messages.append({"role": "assistant", "content": f"Action: {action}"})
            if observation is not None:
                messages.append({"role": "user", "content": f"Observation: {observation}"})

        trajectory_id = str((metadata or {}).get("index", getattr(self, "steps", 0)))
        return {
            "trajectory_id": trajectory_id,
            "messages": messages,
            "metadata": {"query": self._raw_goal_query(goal)},
        }

    def _current_history_marker(self) -> Optional[str]:
        history = getattr(self, "memory", [])[-self.memory_size:]
        if not history or not history[0]:
            return None
        key, value = history[0][0]
        marker = f"{key}: {value}"
        return marker if value is not None else f"{key}: "

    def _normalise_memory_prompt(self, text: Any) -> str:
        if not text:
            return ""
        return str(text).strip()

    def _raw_goal_query(self, goal: Any) -> str:
        return str(goal or "").strip()

    def _build_current_context(self, init_obs: Any) -> str:
        observation = str(init_obs or "").strip()
        return f"Initial observation: {observation}"

    def _json_scalar(self, value: Any):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return value

    def _summarise_retrieved_memories(self, memories: Any) -> List[Dict[str, Any]]:
        if not isinstance(memories, list):
            return []
        summaries: List[Dict[str, Any]] = []
        for rank, memory in enumerate(memories, start=1):
            if not isinstance(memory, dict):
                continue
            summaries.append(
                {
                    "rank": rank,
                    "memory_id": self._json_scalar(memory.get("memory_id")),
                    "retrieval_score": self._json_scalar(memory.get("retrieval_score")),
                    "validation_score": self._json_scalar(memory.get("validation_score")),
                    "score": self._json_scalar(memory.get("score")),
                    "source_trajectory_id": self._json_scalar(memory.get("source_trajectory_id")),
                    "memory_type": self._json_scalar(memory.get("memory_type")),
                }
            )
        return summaries

    def _empty_diagnostics(self) -> Dict[str, Any]:
        return {
            "reme_enabled": self.reme_enabled,
            "retrieve_enabled": self.reme_retrieve_enabled,
            "finish_trial_enabled": self.reme_finish_trial_enabled,
            "retrieve_attempted": False,
            "retrieve_success": False,
            "memory_injected": False,
            "finish_trial_attempted": False,
            "finish_trial_success": False,
        }

    def get_diagnostics(self):
        return {"reme": dict(self.reme_diagnostics)}

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
        reme = config.get("reme", {})
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
            reme,
        )
