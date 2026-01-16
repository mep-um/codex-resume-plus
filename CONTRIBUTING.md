# Contributing

Thanks for contributing!

## Development setup

Prereqs:

- macOS
- Python 3.9+
- `poetry` available on `PATH`

Install dependencies:

```bash
poetry install
```

Run the tool locally:

```bash
poetry run codex-resume-plus
```

## Checks (required)

Before opening a PR, run:

```bash
make check
```

This runs:

- `black --check` (formatting)
- `ruff check` (lint)
- `pyright` (typecheck)
- `pytest` (tests)

## Testing philosophy

- Prefer hermetic tests that don’t require a live Codex session.
- Avoid touching real `~/.codex` in tests; use `CODEX_HOME` overrides (the test
  suite does this).
- For risky operations (archive/unarchive/trust), add tests around safety checks
  and error messaging.

## PR guidelines

- Keep changes small and focused.
- Add/adjust tests for behavior changes (especially around file moves or config edits).
- Preserve backward compatibility with existing sidecar files when possible.

## Reporting bugs

Please include:

- macOS version
- Terminal app + version (e.g., iTerm2 3.x, Terminal.app)
- Shell (e.g., zsh) and `$TERM` (e.g., `xterm-256color`)
- `fzf --version`
- How Codex is installed (and `codex --version` if available)
- A minimal reproduction and what you expected vs what happened

If the issue involves a specific rollout file, avoid sharing sensitive content:
it’s usually enough to share the rollout filename and the relevant metadata block.
