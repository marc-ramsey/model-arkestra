# HTTP Client (`http_client.py`)

The package ships a lightweight `ModelHttpClient` class in `model_arkestra.http_client` that encapsulates aiohttp usage patterns:

| Method | Description |
|---|---|
| `get_json(url)` | GET and return parsed JSON body. |
| `post_json(url, json_body)` | POST with JSON body and return parsed response. |
| `post_raw(url, json_body)` | POST and return an async context manager for raw response streaming (SSE, large binaries). |
| `stream_sse(url, json_body)` | Iterate SSE `data:` lines from a POST endpoint — yields raw strings without the `data:` prefix. |

Sessions are scoped to each call; no manual session management needed.

```python
from model_arkestra.http_client import ModelHttpClient

async with ModelHttpClient(timeout=60) as client:
    data = await client.get_json("http://127.0.0.1:8080/health")
    async for line in client.stream_sse(url, {"prompt": "hi"}):
        print(line)
```

> **Note:** `ModelHttpClient` is a standalone utility — the runner classes use aiohttp directly internally and do not depend on this wrapper. It exists primarily for testing and external integrations.

## Related Documentation

- [Server Documentation](./server.md) — HTTP endpoints you can call with this client
- [Usage Guide](./usage.md) — Python API (recommended over direct HTTP calls)
