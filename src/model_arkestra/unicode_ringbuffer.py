"""A fixed-size ring buffer backed by a bytearray that handles Unicode I/O."""
import struct


class UnicodeRingBuffer:
    """Fixed-capacity ring buffer.  Accepts str on write, yields str lines on read.

    One slot is reserved internally so head==tail always means empty -- the
    standard trick to distinguish full from empty with just head+tail+count.
    Capacity N holds up to N-1 bytes."""

    class BufferFullError(Exception):
        pass

    def __init__(self, capacity: int) -> None:
        if capacity <= 1:
            raise ValueError(
                "capacity must be > 1 (one slot is reserved to distinguish full from empty)"
            )
        self._buf = bytearray(capacity)
        self._capacity = capacity
        self._usable = capacity - 1
        self._head = 0          # next write position (wraps mod _capacity)
        self._tail = 0          # next read position   (wraps mod _capacity)
        self._count = 0         # valid bytes currently in the ring

    @property
    def free_space(self) -> int:
        return self._usable - self._count

    @property
    def used_space(self) -> int:
        return self._count

    def _read_bytes(self, n: int | None = None) -> bytes:
        """Copy valid bytes out of the ring as a flat bytes object."""
        amount = self._count if n is None else min(n, self._count)
        if amount == 0:
            return b''
        # Because _usable < _capacity, head can never equal tail when count > 0.
        if self._head >= self._tail:
            data = bytes(self._buf[self._tail : self._tail + amount])
        else:
            first_part = self._capacity - self._tail
            read_first = min(amount, first_part)
            data = (
                bytes(self._buf[self._tail : self._tail + read_first])
                + bytes(self._buf[: amount - read_first])
            )
        return data

    def _consume(self, n: int) -> None:
        """Advance tail by n valid bytes."""
        self._tail = (self._tail + n) % self._capacity
        self._count -= n

    def write(self, seq: int, text: str | bytes) -> int:
        """Encode *text* as UTF-8 and store it in the ring buffer.

        Each entry is prefixed with a 2-byte big-endian length field,
        followed by a 4-byte big-endian sequence number, then the data.

        Returns len(encoded) on success (including the 6-byte header).
        Raises ValueError if encoded payload > usable capacity.
        Raises BufferFullError if not enough free space."""
        if isinstance(text, bytes):
            payload = text
        else:
            payload = text.encode("utf-8")
        entry = struct.pack(">H", len(payload) + 4) + struct.pack(">I", seq) + payload
        n = len(entry)

        if n > self._usable:
            raise ValueError(
                f"encoded size ({n}) exceeds usable capacity ({self._usable}); "
                f"split the input or grow the buffer"
            )
        if n > self.free_space:
            raise self.BufferFullError(
                f"not enough free space: need {n}, have {self.free_space}"
            )

        end = self._head + n
        if end <= self._capacity:
            # Single contiguous write -- no wrap.
            self._buf[self._head : end] = entry
        else:
            # Wraps around the physical array boundary -- two slice writes.
            first = self._capacity - self._head
            self._buf[self._head :] = entry[:first]
            self._buf[: n - first] = entry[first:]

        self._head = (self._head + n) % self._capacity
        self._count += n
        return n

    def read_lines(self) -> list[str]:
        """Decode all *complete* lines currently in the buffer.

        Incomplete trailing fragments are kept until a newline arrives on a
        subsequent write + read call.  Returns [] when nothing complete
        is available yet."""
        if self._count == 0:
            return []

        data = self._read_bytes()
        lines: list[str] = []
        pos = 0
        while True:
            nl = data.find(b'\n', pos)
            if nl == -1:
                break
            raw_line = data[pos : nl]
            try:
                decoded = raw_line.decode('utf-8')
            except UnicodeDecodeError as exc:
                raise RuntimeError(
                    f"corrupted buffer at positions {pos}..{nl}: {exc}"
                ) from exc
            lines.append(decoded)
            pos = nl + 1

        self._consume(pos)
        return lines

    def read_entries(self, max_lines: int = 1, next_line: int | None = None) -> list[tuple[int, str]]:
        """Read up to *max_lines* most-recent entries from the ring.

        Each entry has a 2-byte big-endian length prefix followed by a
        4-byte big-endian sequence number and the data.  If *next_line*
        is given, entries with ``seq <= next_line`` are skipped.

        Returns [] when nothing is available."""
        if self._count == 0:
            return []

        data = self._read_bytes()
        all_entries: list[tuple[int, str]] = []
        pos = 0
        while pos + 2 <= len(data):  # need at least 2 bytes for length prefix
            entry_len = struct.unpack(">H", data[pos:pos+2])[0]
            if pos + 2 + entry_len > len(data):
                break  # incomplete entry — keep in ring
            entry_data = data[pos+2 : pos+2+entry_len]
            seq = struct.unpack(">I", entry_data[:4])[0]
            text = entry_data[4:].decode("utf-8", errors="replace").removesuffix("\n")
            if next_line is None or seq > next_line:
                all_entries.append((seq, text))
            pos += 2 + entry_len

        self._consume(pos)
        return all_entries[-max_lines:]

    def peek(self) -> str | None:
        """Return all buffered content as a string, or None if empty."""
        raw = self._read_bytes()
        return raw.decode('utf-8', errors='replace') if raw else None

    def flush_lines(self) -> list[str]:
        """Read everything including the incomplete trailing fragment.
        Use at end-of-stream when no more data is expected."""
        lines = self.read_lines()
        remainder = self._read_bytes()
        if remainder:
            try:
                lines.append(remainder.decode('utf-8'))
            except UnicodeDecodeError:
                lines.append(remainder.decode('utf-8', errors='replace'))
            self._count = 0
        return lines

    def __len__(self) -> int:
        return self.used_space

    def __bool__(self) -> bool:
        return self._count > 0

    def __repr__(self) -> str:
        return (
            f'UnicodeRingBuffer(cap={self._capacity}, '
            f'used={self.used_space}, free={self.free_space})'
        )
