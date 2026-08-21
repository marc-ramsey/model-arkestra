# Log Ring Buffer

Fixed-capacity ring buffer for model log capture, backed by `bytearray`. Supports wrap-around writes, UTF-8 encoding/decoding, and length-prefixed entry storage.

## Format

Each stored entry is a flat binary blob:

```
[2-byte BE length][4-byte BE seq][UTF-8 text]
```

| Field | Size | Endian | Purpose |
|---|---|---|---|
| Length | 2 bytes | Big-endian | Byte length of `[seq + text]` (excludes the 2-byte length field itself) |
| Sequence | 4 bytes | Big-endian | Monotonically increasing counter assigned at write time |
| Text | Variable | UTF-8 | The log line content (trailing `\n` included in storage, stripped on read) |

Total entry size = `length + 2` bytes.

## Capacity and Overflow

The buffer reserves one byte (`_usable = capacity - 1`) to keep `head == tail` as the unambiguous empty condition. When no free space remains:

1. `read_entries()` consumes what it can read from the head of the ring (oldest entries).
2. `_append_log_line()` catches `BufferFullError` and discards the current write — effectively evicting the oldest entry to make room for the new one.

### Minimum Validation

At `_ModelContext.__init__`, computed buffer size is checked:

```python
buf_bytes = max_log_lines * AVG_LINE_BYTES  # default 200
if buf_bytes < 10:
    raise ValueError("log buffer too small")
```

Smaller values are rejected to prevent a ring that immediately overwrites itself.

## API

### `write(seq, text) -> int`

Packs `[2-byte len][4-byte seq][text]` and writes to the ring. Raises `BufferFullError` if no space; caller must handle (typically by discarding oldest).

Returns total bytes written including header.

### `read_entries(max_lines=1, next_line=None) -> list[tuple[int, str]]`

Parses all complete entries from the buffer's current head position:

- Skips entries with `seq <= next_line` (incremental read support)
- Strips trailing `\n` from text before returning
- Returns the last `max_lines` matching entries

The `next_line` parameter enables streaming consumption — callers pass the highest sequence they've already seen. Incomplete entries (no full length prefix or truncated data) remain unconsumed until more data arrives.

### `peek() -> str | None`

Returns all buffered content decoded as UTF-8 string, useful for debugging. Does not consume data.

## Lifecycle in `_ModelContext`

```
append_log_line(line: str) -> int
  ├── self._log_seq += 1
  ├── ensure line ends with \n
  └── ring.write(self._log_seq, line)    # BufferFullError → discard oldest

get_lines_since(since: int, max_lines: int) -> [(seq, text), ...], int
  └── ring.read_entries(max_lines=max_lines, next_line=since)
```

Callers (`ProcessModelRunner`, `ContainerModelRunner`) invoke `_append_log_line()` from async log capture tasks. The admin SSE endpoint and `get_logs()` call `_get_lines_since()` to stream or snapshot recent output.

### Restart Behavior

On restart, `_before_restart()` allocates a fresh `UnicodeRingBuffer` — no state copying from the old buffer. Sequence counter resets to 0 for clean monotonic ordering.
