# Examples

Example programs demonstrating ModelArkestra usage patterns.

| Program | Description |
|---|---|
| [`cli.py`](./cli.py) | Interactive CLI client — sends prompts to a local LLM via streaming, with one-shot `--prompt` and REPL modes. Supports backend/container overrides and pass-through model parameters. |

## Adding examples

Place example programs here. Each should:

- Be self-contained (import from `model_arkestra`, not from other examples)
- Have an entry in this README table
- Accept a `--help` flag that documents its options
- Clean up resources on exit (e.g. `stop_all()` in a `finally` block)
