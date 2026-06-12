"""HiAgent variants that split subgoal planning from action grounding."""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

from common.registry import registry
from utils.logging.agent_logger import AgentLogger

from .cme_final import ContextEfficientAgentV2, extract_numbers
from .gmemory_client import GMemoryClient
from .summarize import TrajectorySummarizer


logger = AgentLogger(__name__)


@registry.register_agent("SplitSubgoalActionContextEfficientAgent")
class SplitSubgoalActionContextEfficientAgent(ContextEfficientAgentV2):
    """Run HiAgent planning first, then ground the active subgoal in a second call."""

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
        fallback_to_draft_action=False,
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
        self.fallback_to_draft_action = bool(fallback_to_draft_action)
        self.active_subgoal = None
        self.diagnostics = []
        self.subgoal_texts = []
        self.action_only_call_count = 0
        self.draft_action_discard_count = 0
        self.action_parse_failure_count = 0
        self.missing_active_subgoal_count = 0
        self._last_hiagent_prompt = ""
        self._last_action_only_prompt = ""
        self._pristine_alfworld_instruction = instruction

    def reset(self, goal, init_obs, init_act=None):
        super().reset(goal, init_obs, init_act)
        self.active_subgoal = None
        self.diagnostics = []
        self.subgoal_texts = []
        self.action_only_call_count = 0
        self.draft_action_discard_count = 0
        self.action_parse_failure_count = 0
        self.missing_active_subgoal_count = 0
        self._last_hiagent_prompt = ""
        self._last_action_only_prompt = ""

    def run(self, init_prompt_dict=None):
        if init_prompt_dict is not None:
            self.init_prompt_dict = init_prompt_dict
            self._pristine_alfworld_instruction = init_prompt_dict.get("instruction", "")
            self.instruction = init_prompt_dict["instruction"]
            self.examples = init_prompt_dict["examples"]
        else:
            self._pristine_alfworld_instruction = self.init_prompt_dict.get("instruction", self.instruction)

        system_message = self.init_prompt_dict["system_msg"]
        hiagent_prompt = self.make_prompt(
            need_goal=self.need_goal,
            check_actions=self.check_actions,
            check_inventory=self.check_inventory,
            system_message=system_message,
        )
        self._last_hiagent_prompt = hiagent_prompt
        self.log_example_prompt_subgoal(hiagent_prompt)

        success, hiagent_raw_output = self.llm_model.generate(system_message, hiagent_prompt)
        print(f'-------------GPT Response---------\n{hiagent_raw_output}\n---------------[END]------------')
        if not success:
            return success, hiagent_raw_output

        parsed_hiagent = self._parse_hiagent_output(hiagent_raw_output or "")
        if parsed_hiagent["subgoal"]:
            self.active_subgoal = parsed_hiagent["subgoal"]
            self.subgoal_texts.append(self.active_subgoal)
            self.subgoal_idx = []
            self.memory.append([("Subgoal", self.active_subgoal)])
        elif self.active_subgoal is None:
            self.active_subgoal = self._latest_subgoal_from_memory()
            if self.active_subgoal is None:
                self.missing_active_subgoal_count += 1

        if parsed_hiagent["draft_action"]:
            self.draft_action_discard_count += 1

        action_prompt = self.make_action_only_prompt(
            system_message=system_message,
            pristine_instruction=self._pristine_alfworld_instruction,
            gmemory_prompt_or_none=self._action_memory_prompt(),
        )
        self._last_action_only_prompt = action_prompt
        self.log_example_prompt_action(action_prompt)
        self.action_only_call_count += 1

        action_success, action_only_raw_output = self.llm_model.generate(system_message, action_prompt)
        print(f'-------------Action-Only Response---------\n{action_only_raw_output}\n---------------[END]------------')
        if not action_success:
            self._record_step_diagnostics(
                hiagent_raw_output=hiagent_raw_output,
                parsed_hiagent=parsed_hiagent,
                action_only_prompt=action_prompt,
                action_only_raw_output=action_only_raw_output,
                action_only_parsed_action=None,
                agent_returned_action=None,
            )
            return action_success, action_only_raw_output

        action_only_parsed_action = self._parse_action_only_output(action_only_raw_output or "")
        if not action_only_parsed_action:
            self.action_parse_failure_count += 1

        agent_returned_action = action_only_parsed_action
        if not agent_returned_action and self.fallback_to_draft_action:
            agent_returned_action = parsed_hiagent["draft_action"]

        self._record_step_diagnostics(
            hiagent_raw_output=hiagent_raw_output,
            parsed_hiagent=parsed_hiagent,
            action_only_prompt=action_prompt,
            action_only_raw_output=action_only_raw_output,
            action_only_parsed_action=action_only_parsed_action,
            agent_returned_action=agent_returned_action,
        )
        return True, agent_returned_action

    def make_action_only_prompt(
        self,
        system_message="",
        pristine_instruction="",
        gmemory_prompt_or_none=None,
    ):
        base_instruction = pristine_instruction if pristine_instruction is not None else ""
        memory_prompt = gmemory_prompt_or_none if gmemory_prompt_or_none else "None"
        history = self.memory[-self.memory_size:]
        serialized_history = self._serialize_history_for_action_only(history)
        active_subgoal = self.active_subgoal or "None"
        input_prompt = (
            "<instruction>\n"
            f"{base_instruction}\n\n"
            "Note: A subgoal is a milestone goal that you need to complete in order to achieve the final goal.\n"
            "There is an unfinished subgoal. You need to ground the given subgoal to corresponding executable actions for solving the given task in the following format: \"Action: {action}\".\n"
            "Instructions:\n"
            "1. Do not output a new subgoal.\n"
            "2. Output only one valid action.\n"
            "3. If the current action fails, you need to execute \"check valid actions\" to get a list of valid actions and select one from the list.\n"
            "</instruction>\n"
            "<goal>\n"
            f"You should perform actions to accomplish the goal: {self.goal}\n"
            "</goal>\n"
            "You should use the following commands for help when your action cannot be understood: check valid actions\n"
            "You should use the following commands for help when your action cannot be understood: inventory\n"
            f"Current subgoal: {active_subgoal}\n\n"
            f"{serialized_history}\n\n"
            "Retrieved memory:\n"
            f"{memory_prompt}\n\n"
            "Use retrieved memory only as procedural guidance for the next action under the current subgoal.\n\n"
            "Action:"
        )
        return self._trim_prompt_to_context(system_message, input_prompt)

    def _parse_hiagent_output(self, output: str) -> Dict[str, Optional[str]]:
        subgoal = None
        draft_text = output
        if "Subgoal" in output:
            lines = output.splitlines()
            subgoal_line = lines[0] if lines else output
            if "Subgoal" in subgoal_line and ":" in subgoal_line:
                subgoal = ":".join(subgoal_line.split(":")[1:]).strip()
            else:
                subgoal = subgoal_line.replace("Subgoal", "").strip(" :")
            draft_text = "\n".join(lines[1:])
        draft_action = self.action_parser_for_special_llms(draft_text) if draft_text.strip() else ""
        return {
            "subgoal": subgoal,
            "draft_action": draft_action,
            "draft_action_raw": draft_text,
        }

    def _parse_action_only_output(self, output: str) -> str:
        if not output:
            return ""
        action = self.action_parser_for_special_llms(output) if self.use_parser else output.strip().split("\n")[0].strip()
        if "retrieve(" in action.lower():
            numbers = extract_numbers(action.lower())
            self.subgoal_idx += numbers
            return ""
        return action

    def _action_memory_prompt(self):
        return "None"

    def _memory_injected_to_action_prompt(self):
        return False

    def _latest_subgoal_from_memory(self):
        for item in reversed(getattr(self, "memory", [])):
            fields = dict(item)
            if "Subgoal" in fields:
                return fields.get("Subgoal")
        return None

    def _serialize_history_for_action_only(self, history):
        return self._serialize_history(history)

    def _serialize_history(self, history):
        def vanilla_serialize_history(items):
            res = []
            for item in items:
                for field in item:
                    res.append(field[0] + ": " + field[1])
            return "\n".join(res)

        task = os.environ.get("EVALTASK", "")
        summarization = not any(name in task for name in ["gripper", "blocksworld"])
        subgoal_index_list = []
        keep_subgoal_index_list = [idx - 1 for idx in self.subgoal_idx]
        for i, item in enumerate(history):
            if item and item[0][0] == "Subgoal":
                subgoal_index_list.append(i)
        if len(subgoal_index_list) <= 1:
            return vanilla_serialize_history(history)

        final_subgoal = subgoal_index_list[-1]
        new_history = history[:subgoal_index_list[0]]
        for i in range(0, len(subgoal_index_list) - 1):
            if i in keep_subgoal_index_list:
                new_history += history[subgoal_index_list[i]:subgoal_index_list[i + 1]]
                continue
            index = subgoal_index_list[i]
            obs_index = subgoal_index_list[i + 1] - 1
            subgoal = history[index][0]
            subgoal = (f"{i + 1} {subgoal[0]}", subgoal[1])
            if not summarization:
                new_history.append([subgoal, ("Observation", history[obs_index][1][1])])
            else:
                summarizer = TrajectorySummarizer(self.llm_model)
                trajectory = history[index + 1:obs_index + 1]
                trajectory = [pair for pair in trajectory if pair[0][0] != "Action" or "check valid" not in pair[0][1]]
                summary = summarizer.generate_summary([trajectory], [history[index][0]])[0]
                new_history.append([subgoal, ("Observation", summary)])

        subgoal = history[final_subgoal][0]
        subgoal = (f"{len(subgoal_index_list)} {subgoal[0]}", subgoal[1])
        new_history += [[subgoal]] + history[final_subgoal + 1:]
        return vanilla_serialize_history(new_history)

    def _trim_prompt_to_context(self, system_message, input_prompt):
        if not hasattr(self.llm_model, "num_tokens_from_messages"):
            return input_prompt
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_prompt},
        ]
        try:
            num_of_tokens = self.llm_model.num_tokens_from_messages(messages)
        except Exception:
            return input_prompt
        history = self.memory[-self.memory_size:]
        while num_of_tokens > self.max_context_length - self.llm_model.max_tokens and len(history) > 1:
            history = history[1:]
            serialized_history = self._serialize_history_for_action_only(history)
            input_prompt = re.sub(
                r"Current subgoal: .*?\n\n.*?\n\nRetrieved memory:",
                f"Current subgoal: {self.active_subgoal or 'None'}\n\n{serialized_history}\n\nRetrieved memory:",
                input_prompt,
                count=1,
                flags=re.DOTALL,
            )
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_prompt},
            ]
            try:
                num_of_tokens = self.llm_model.num_tokens_from_messages(messages)
            except Exception:
                break
        return input_prompt

    def _record_step_diagnostics(
        self,
        hiagent_raw_output,
        parsed_hiagent,
        action_only_prompt,
        action_only_raw_output,
        action_only_parsed_action,
        agent_returned_action,
    ):
        step = {
            "step": self.action_only_call_count - 1,
            "hiagent_prompt": self._last_hiagent_prompt,
            "hiagent_raw_output": hiagent_raw_output,
            "hiagent_draft_action_raw": parsed_hiagent.get("draft_action_raw"),
            "hiagent_draft_action": parsed_hiagent.get("draft_action"),
            "current_subgoal": self.active_subgoal,
            "action_only_prompt": action_only_prompt,
            "action_only_raw_output": action_only_raw_output,
            "action_only_parsed_action": action_only_parsed_action,
            "agent_returned_action": agent_returned_action,
            "executed_action": None,
            "memory_injected_to_subgoal_prompt": False,
            "memory_injected_to_action_prompt": self._memory_injected_to_action_prompt(),
            "gmemory_prompt_chars": len(getattr(self, "gmemory_prompt", "") or ""),
        }
        self.diagnostics.append(step)

    def get_diagnostics(self):
        return {
            "agent_name": self.__class__.__name__,
            "memory_injected_to_subgoal_prompt": False,
            "memory_injected_to_action_prompt": self._memory_injected_to_action_prompt(),
            "gmemory_prompt_chars": len(getattr(self, "gmemory_prompt", "") or ""),
            "subgoal_count": len(self.subgoal_texts),
            "subgoal_texts": list(self.subgoal_texts),
            "subgoal_change_frequency": len(self.subgoal_texts),
            "action_only_call_count": self.action_only_call_count,
            "draft_action_discard_count": self.draft_action_discard_count,
            "action_parse_failure_count": self.action_parse_failure_count,
            "missing_active_subgoal_count": self.missing_active_subgoal_count,
            "steps": list(self.diagnostics),
        }

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
        fallback_to_draft_action = config.get("fallback_to_draft_action", False)
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
            fallback_to_draft_action,
        )


@registry.register_agent("GMemoryActionOnlyContextEfficientAgent")
class GMemoryActionOnlyContextEfficientAgent(SplitSubgoalActionContextEfficientAgent):
    """Split HiAgent variant that exposes GMemory only to action grounding."""

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
        fallback_to_draft_action=False,
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
            fallback_to_draft_action,
        )
        self.gmemory_config = gmemory or {}
        self.gmemory_enabled = bool(self.gmemory_config.get("enabled", False))
        self.gmemory_recall_on_reset = bool(self.gmemory_config.get("recall_on_reset", True))
        self.gmemory_upload_on_finish = bool(self.gmemory_config.get("upload_on_finish", True))
        self.gmemory_max_context_chars = int(self.gmemory_config.get("max_context_chars", 1000))
        self.gmemory_memory_only = bool(self.gmemory_config.get("memory_only", False))
        self.gmemory_prompt = ""
        self.gmemory_client = self._build_gmemory_client()

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
            logger.info("GMemory retrieve completed: memory_prompt_chars=%s", len(self.gmemory_prompt))
        except Exception as exc:
            self.gmemory_prompt = ""
            logger.warning("GMemory retrieve failed: %s", exc)

    def _build_gmemory_client(self) -> Optional[GMemoryClient]:
        if not self.gmemory_enabled:
            return None
        base_url = self.gmemory_config.get("base_url")
        if not base_url:
            logger.warning("GMemory is enabled but base_url is incomplete")
            return None
        timeout = self.gmemory_config.get("timeout", 10.0)
        return GMemoryClient(base_url=base_url, timeout=timeout)

    def _action_memory_prompt(self):
        return self.gmemory_prompt if self.gmemory_prompt else "None"

    def _memory_injected_to_action_prompt(self):
        return bool(self.gmemory_prompt)

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
        initial_observation = getattr(self, "init_obs", None)
        reward_by_step = self._score_change_record_to_dict(score_change_record)
        steps = []
        current_subgoal = None
        action_step_id = 0

        for item in getattr(self, "memory", []):
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
            "agent": "GMemoryActionOnlyContextEfficientAgent",
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
        fallback_to_draft_action = config.get("fallback_to_draft_action", False)
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
            fallback_to_draft_action,
            gmemory,
        )
