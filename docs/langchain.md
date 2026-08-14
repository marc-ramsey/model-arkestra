# LangChain LCEL Integration

ModelArkestra ships with a LangChain adapter that wraps any started model to implement the standard LangChain chat model interface. This enables drop-in compatibility with LangGraph, LangServe, and other LangChain ecosystem tools.

```python
from model_arkestra.arkestra import ModelArkestra
from model_arkestra.langchain_adapter import LangChainModelAdapter

async with ModelArkestra("config.yaml") as runner:
    await runner.start("qwen3-4b")

    adapter = LangChainModelAdapter(runner, "qwen3-4b")

    # ── Blocking invocation ──────────────────────────────
    result = await adapter.ainvoke("What is quantum entanglement?")
    print(result.content)  # → "Quantum entanglement is a phenomenon..."

    # ── Token-by-token streaming ─────────────────────────
    async for chunk in adapter.astream("Write a haiku about code"):
        print(chunk.content, end="", flush=True)
    # → partial tokens accumulating (Hello World!)

    # ── Typed event stream (LangGraph-compatible) ────────
    async for event in adapter.astream_events("Explain photosynthesis"):
        if event["event"] == "on_chat_model_stream":
            print(event["data"]["chunk"].content, end="", flush=True)
        elif event["event"] == "on_chat_model_end":
            print("\n[done]")
```

## Input Types

The adapter accepts all LangChain `LanguageModelInput` variants:

| Input type | Example |
|---|---|
| `str` | `"Hello world"` |
| OpenAI-style dicts | `{"role": "user", "content": "Hi"}` |
| List of dicts | `[{"role": "system", "content": "Be nice"}, {"role": "user", "content": "Say hello"}]` |
| LangChain `BaseMessage` list | `[HumanMessage(content="Hi"), AIMessage(content="Hello!")]` |
| `PromptValue` | A LangChain prompt template's `.invoke()` output |

The adapter normalizes all inputs to the OpenAI-compatible message format (`[{"role": "...", "content": "..."}, ...]`) and passes the full conversation history to the underlying runner.

## Supported Parameters

Both `ainvoke` and `astream` accept:

| Parameter | Type | Description |
|---|---|---|
| `input` | `LanguageModelInput` | User input (see above) |
| `config` | `RunnableConfig` | LangChain runnable config (passed through, reserved for future use) |
| `stop` | `list[str]` | Stop sequences sent to the server |
| `**kwargs` | — | Additional parameters forwarded to the model (e.g., `temperature`, `max_tokens`, `top_p`) |

## With LangChain Expression Language / LangGraph

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("placeholder", "{messages}"),
])

# The adapter works as an LCEL runnable:
chain = prompt | adapter  # or use .bind(stop=["\n"])

result = await chain.ainvoke({"messages": [("user", "What's the weather?")]})
```

## Related Documentation

- [Usage Guide](./usage.md) — how to start and run models before wrapping with LangChain
- [API Reference — ModelArkestra](./api/model-arkestra.md) — the underlying `runner` that the adapter wraps
- [Server Documentation](./server.md) — alternative: use the OpenAI-compatible API server instead of LangChain integration
