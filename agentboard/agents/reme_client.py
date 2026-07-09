"""Lightweight HTTP client for the ReMe HiAgent API."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional
from urllib import error, request


class ReMeClient:
    def __init__(
        self,
        base_url: str,
        retrieve_timeout: float = 15.0,
        finish_trial_timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.retrieve_timeout = retrieve_timeout
        self.finish_trial_timeout = finish_trial_timeout

    def retrieve(
        self,
        workspace_id: str,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        rerank: bool = False,
        rewrite: bool = False,
        max_context_chars: int = 3000,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "workspace_id": workspace_id,
            "query": query,
            "top_k": top_k,
            "min_score": min_score,
            "rerank": rerank,
            "rewrite": rewrite,
            "max_context_chars": max_context_chars,
        }
        data = self._post("/api/v1/memory/retrieve", payload, self.retrieve_timeout)
        self._validate_retrieve_response(data)
        return data

    def finish_trial(
        self,
        workspace_id: str,
        request_id: str,
        retrieval_id: Optional[str],
        trajectory: Dict[str, Any],
        outcome: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "workspace_id": workspace_id,
            "request_id": request_id,
            "retrieval_id": retrieval_id,
            "trajectory": trajectory,
            "outcome": outcome,
        }
        data = self._post("/api/v1/memory/finish-trial", payload, self.finish_trial_timeout)
        if not isinstance(data, dict):
            raise RuntimeError("ReMe finish-trial response must be a JSON object")
        return data

    def _post(self, path: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        response_body = self._post_raw(path, payload, timeout)
        if not response_body:
            return {}
        try:
            data = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"ReMe response is not valid JSON: {response_body}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"ReMe response must be a JSON object: {response_body}")
        return data

    def _post_raw(self, path: str, payload: Dict[str, Any], timeout: float) -> str:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ReMe request failed with HTTP {exc.code}: {response_body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"ReMe request failed: {exc}") from exc

    def _validate_retrieve_response(self, data: Dict[str, Any]) -> None:
        if "memory_prompt" not in data:
            raise RuntimeError("ReMe retrieve response is missing `memory_prompt`")
        if "memories" not in data:
            raise RuntimeError("ReMe retrieve response is missing `memories`")
        if not isinstance(data.get("memories"), list):
            raise RuntimeError("ReMe retrieve response `memories` must be a list")
