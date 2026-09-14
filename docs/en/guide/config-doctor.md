# Config Doctor

Config Doctor runs Relax's existing training argument parser and backend validators before any Ray, SGLang, or GPU worker is started. It is intended for local checks and CI, not as a replacement training launcher.

## Usage

Pass the complete arguments that would follow `python -m relax.entrypoints.train` after `--`:

```bash
python -m relax.entrypoints.doctor -- \
  --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
  --colocate \
  --hf-checkpoint /models/Qwen3-4B \
  --prompt-data /data/train.jsonl \
  # ...the remaining training arguments
```

Use `--format json` for machine-readable output:

```bash
python -m relax.entrypoints.doctor --format json -- <training arguments>
```

A valid configuration exits with code `0` and prints the merged configuration, active roles, GPU requirement, and expected training command. Invalid or unknown options return a non-zero exit code with the original validation message and a suggested correction.

The command does not execute shell scripts. Expand a recipe first and pass the resulting training arguments directly. Sensitive values such as API keys, tokens, passwords, credentials, and `--wandb-key` are redacted from reports.
