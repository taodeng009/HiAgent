"""VanillaAgent variant with GMemory retrieval and episode upload hooks."""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

from common.registry import registry
from utils.logging.agent_logger import AgentLogger

from .gmemory_client import GMemoryClient
from .vanilla_agent import VanillaAgent


logger = AgentLogger(__name__)


@registry.register_agent("GMemoryVanillaAgent")
class GMemoryVanillaAgent(VanillaAgent):
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
        )
        self.gmemory_config = gmemory or {}
        self.gmemory_enabled = bool(self.gmemory_config.get("enabled", False))
        self.gmemory_recall_on_reset = bool(self.gmemory_config.get("recall_on_reset", True))
        self.gmemory_upload_on_finish = bool(self.gmemory_config.get("upload_on_finish", True))
        self.gmemory_max_context_chars = int(self.gmemory_config.get("max_context_chars", 1000))
        self.gmemory_memory_only = bool(self.gmemory_config.get("memory_only", False))
        self.gmemory_prompt = ""
        self.gmemory_client = self._build_gmemory_client()

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
        if not self.gmemory_enabled or not self.gmemory_recall_on_reset or self.gmemory_client is None:
            return
        try:
            response = self.gmemory_client.retrieve(
                task_type=self.gmemory_config.get("task_type") or os.environ.get("EVALTASK", ""),
                goal=goal,
                initial_observation=init_obs,
                query=self.gmemory_config.get("query"),
                max_chars=self.gmemory_max_context_chars,
            )
            self.gmemory_prompt = self._filter_gmemory_prompt(response.get("memory_prompt", ""))
            logger.info(
                "GMemory retrieve completed: memory_prompt_chars=%s",
                len(self.gmemory_prompt),
            )
        except Exception as exc:
            self.gmemory_prompt = ""
            logger.warning("GMemory retrieve failed: %s", exc)

    def make_prompt(self, need_goal=False, check_actions="check valid actions", check_inventory="inventory", system_message=''):
        prompt = super().make_prompt(
            need_goal=need_goal,
            check_actions=check_actions,
            check_inventory=check_inventory,
            system_message=system_message,
        )
        if self.gmemory_prompt:
            prompt = self._inject_gmemory_prompt(prompt)
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

    def _current_history_marker(self) -> Optional[str]:
        history = getattr(self, "memory", [])[-self.memory_size:]
        if not history:
            return None
        key, value = history[0]
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
        observations = []
        actions = []

        for key, value in memory:
            if key == "Observation":
                observations.append(value)
                if initial_observation is None and value is not None:
                    initial_observation = value
            elif key == "Action" and value:
                actions.append(value)

        steps = []
        for action_step_id, action in enumerate(actions):
            observation = observations[action_step_id + 1] if action_step_id + 1 < len(observations) else ""
            step = {
                "action": str(action),
                "observation": str(observation or ""),
            }
            if action_step_id in reward_by_step:
                step["reward"] = reward_by_step[action_step_id]
            steps.append(step)

        if not steps:
            logger.warning("GMemory episode upload skipped: no action steps recorded")
            return None

        episode_metadata = {
            "agent": "GMemoryVanillaAgent",
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
            gmemory,
        )
