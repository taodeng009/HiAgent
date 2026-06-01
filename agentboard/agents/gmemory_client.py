"""Lightweight HTTP client for the GMemory API."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional
from urllib import error, request


class GMemoryClient:
    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def retrieve(
        self,
        task_type: str,
        goal: Optional[str],
        initial_observation: str,
        query: Optional[str] = None,
        max_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "task_type": task_type,
            "goal": goal,
            "initial_observation": initial_observation,
            "query": query,
        }
        if max_chars is not None:
            payload["max_chars"] = max_chars
        data = self._post("/api/v1/memory/retrieve", payload)
        if "memory_prompt" not in data:
            raise RuntimeError("GMemory retrieve response is missing `memory_prompt`")
        return data

    def save_episode(
        self,
        task_type: str,
        goal: Optional[str],
        initial_observation: str,
        success: bool,
        progress_rate: Optional[float],
        steps: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "task_type": task_type,
            "goal": goal,
            "initial_observation": initial_observation,
            "success": success,
            "progress_rate": progress_rate,
            "steps": steps,
            "metadata": metadata or {},
        }
        return self._post("/api/v1/memory/episodes", payload)

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response_body = self._post_raw(path, payload)
        if not response_body:
            return {}
        try:
            return json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"GMemory response is not valid JSON: {response_body}") from exc

    def _post_raw(self, path: str, payload: Dict[str, Any]) -> str:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8")
        except error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GMemory request failed with HTTP {exc.code}: {response_body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"GMemory request failed: {exc}") from exc
