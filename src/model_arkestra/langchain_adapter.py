"""LangChain LCEL wrapper for ModelArkestra."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Union

from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables.schema import StandardStreamEvent, CustomStreamEvent
from model_arkestra.arkestra import ModelArkestra


# ── Input normalization ───────────────────────────────────────────────

def _normalize_messages(input: LanguageModelInput) -> List[Dict[str, Any]]:
    """Convert LangChain's LanguageModelInput into a list of message dicts."""
    if isinstance(input, str):
        return [{"role": "user", "content": input}]

    if isinstance(input, PromptValue):
        # PromptValue has .to_messages() → list[BaseMessage]
        messages = input.to_messages()
        return [_message_to_dict(m) for m in messages]

    # Single message dict  
    if isinstance(input, dict):
        return [input]

    # Sequence of BaseMessage or dict-like items
    if isinstance(input, (list, tuple)):
        return [_message_to_dict(item) for item in input]

    raise TypeError(f"Unsupported LanguageModelInput type: {type(input).__name__}")


def _message_to_dict(msg: Any) -> Dict[str, Any]:
    """Convert a LangChain BaseMessage into an OpenAI-style message dict."""
    if isinstance(msg, dict):
        return msg  # already a dict

    role = getattr(msg, "type", None) or getattr(msg, "_type", "unknown")
    # Map LangChain type names to OpenAI roles
    role_map: Dict[str, str] = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "function": "function",
        "tool": "tool",
        "generic": "user",
    }
    openai_role = role_map.get(role, "user")

    # Get content — may be str or list[ContentBlock] depending on LangChain version
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return {"role": openai_role, "content": content}
    elif isinstance(content, list):
        return {"role": openai_role, "content": content}
    else:
        return {"role": openai_role, "content": ""}


# ── Wrapper ───────────────────────────────────────────────────────────

class LangChainModelAdapter:
    """Wraps a ModelArkestra instance to implement LangChain's LCEL chat model interface.

    Usage:
        arkestra = ModelArkestra("config.yaml")
        await arkestra.start("qwen3-4b")

        adapter = LangChainModelAdapter(arkestra, "qwen3-4b")

        # Blocking invocation
        result = await adapter.ainvoke("What is quantum entanglement?")

        # Token-by-token streaming
        async for chunk in adapter.astream("Write a haiku"):
            print(chunk.content, end="", flush=True)

        # Typed event streaming (LangGraph-compatible)
        async for event in adapter.astream_events("Explain photosynthesis"):
            print(f"{event['event']}: {event.get('data', {})}")
    """

    def __init__(self, arkestra: ModelArkestra, model_name: str, **kwargs: Any):
        self._arkestra = arkestra
        self._model_name = model_name
        self._config: Dict[str, Any] = kwargs

    # ── Core LCEL methods ───────────────────────────────────────────────

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AIMessageChunk:
        """Blocking invocation — returns full response as an AIMessageChunk."""
        messages = _normalize_messages(input)

        # Build kwargs for the underlying request (stop, temperature, etc.)
        req_kwargs = self._build_request_kwargs(stop, kwargs)

        # Pass full message list to backend (with stop/other params as extra kwargs)
        result_text = await self._arkestra.ainvoke(
            self._model_name,
            prompt=messages[-1]["content"] if messages else "",
            messages=messages,
        )

        return AIMessageChunk(content=result_text)

    async def astream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        """Token-by-token streaming — yields partial AIMessageChunks that accumulate."""
        messages = _normalize_messages(input)

        # Build extra params (stop, temperature, etc.) and pass full message list
        req_kwargs = self._build_request_kwargs(stop, kwargs)

        payload: Dict[str, Any] = {"messages": messages}
        payload.update(req_kwargs)

        buffer: List[str] = []
        total_tokens = 0

        async for event in self._arkestra.astream(
            self._model_name, payload=payload
        ):
            if "token" in event:
                buffer.append(event["token"])
                yield AIMessageChunk(content="".join(buffer))
            elif "usage" in event:
                usage = event["usage"]
                yield AIMessageChunk(
                    content="".join(buffer),
                    response_metadata={
                        "model": usage.get("model"),
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    },
                )

    async def astream_events(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        version: str = "v2",
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StandardStreamEvent | CustomStreamEvent]:
        """Typed event stream — LangGraph-compatible streaming protocol."""
        messages = _normalize_messages(input)
        req_kwargs = self._build_request_kwargs(stop, kwargs)

        # Pass full message list (with extra params like stop, temperature)
        payload: Dict[str, Any] = {"messages": messages}
        payload.update(req_kwargs)

        # Emit start event
        yield StandardStreamEvent(
            event="on_chat_model_start",
            name=self._model_name,
            data={"input": [m for m in messages] if not isinstance(messages[0], str) else messages},
        )

        buffer: List[str] = []
        total_tokens = 0

        async for event in self._arkestra.astream(
            self._model_name, payload=payload
        ):
            if "token" in event:
                buffer.append(event["token"])
                yield StandardStreamEvent(
                    event="on_chat_model_stream",
                    name=self._model_name,
                    data={"chunk": AIMessageChunk(content=event["token"])},
                )

        full_text = "".join(buffer)
        response_metadata: Dict[str, Any] = {}

        # Collect usage from the last event if it had one
        # Re-emit with final content
        yield StandardStreamEvent(
            event="on_chat_model_stream",
            name=self._model_name,
            data={"chunk": AIMessageChunk(content=full_text)},
        )
        yield StandardStreamEvent(
            event="on_chat_model_end",
            name=self._model_name,
            data={"output": AIMessageChunk(content=full_text, response_metadata=response_metadata)},
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    def _build_request_kwargs(self, stop: Optional[List[str]], extra: Dict[str, Any]) -> Dict[str, Any]:
        """Map LangChain kwargs to backend request parameters."""
        req_kwargs: Dict[str, Any] = {}
        if stop:
            req_kwargs["stop"] = stop
        req_kwargs.update(extra)
        return req_kwargs
