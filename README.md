# codex-resume-plus

`codex resume`, but usable at scale: a fast, safe `fzf` picker for Codex CLI
sessions with titles, rapid archiving, and per-session workdir overrides.

## Why this exists

Codex CLI doesn’t currently give you a fast, safe way to do a few high-value
maintenance tasks:

- **Rename sessions:** there’s no built-in way to rename sessions in `codex resume`,
  and rewriting Codex’s JSONL logs is risky (they’re append-only and may be actively
  written).
- **Archive/unarchive sessions:** there’s no built-in way to archive sessions in
  `codex resume`, and doing it manually is both slow and easy to get wrong.
- **“Take a session with you” to a new project root:** if you move/rename a repo
  (or want to continue a session in a different directory), resuming from the
  wrong working directory breaks relative paths and can trigger lots of approval
  prompts because Codex treats the new repo as outside the current writable
  scope.
- **Trust the current directory quickly:** there’s no one-key “trust this
  directory” flow inside Codex CLI.

This tool reads Codex session rollouts from `~/.codex/sessions/**/rollout-*.jsonl`
and gives you a single-screen workflow to identify the right session, label it,
resume it in the right place, and archive it safely.

## What it does

- **List + search:** shows sessions (default: only ones in your current directory)
  with a stable identifier and your optional title.
- **Summary preview (default):** shows metadata + the first meaningful user
  message plus the most recent user/assistant messages, with boilerplate and
  noisy IDE context (documents/images lists) stripped.
- **Titles without touching logs:** store a human title in a sidecar keyed by
  `session_id`.
- **Rapid archive/unarchive:** archive/unarchive from the same screen with
  double-confirm guardrails.
- **Per-session workdir override (Ctrl-O):** store a “resume root” for a session,
  so opening it from this tool runs:
  `codex resume --add-dir <workdir> -C <workdir> <session_id>`.
  This is what lets you “move” a session to a new repo without editing the
  rollout log:
  - relative paths resolve against the new repo
  - Codex can treat the repo as a writable root (reducing approval prompts)
- **Trust current directory (Ctrl-Y):** add the current directory to Codex’s
  config as trusted (`trust_level = "trusted"`), reducing approval friction.

## Requirements

- macOS
- `codex` on your `PATH`
- `fzf` on your `PATH` (required)
- `lsof` on your `PATH` (recommended; used to block archiving open rollouts)
- Python 3.9+

## Install

For development / local use:

```bash
brew install fzf
poetry install
poetry run codex-resume-plus
```

If you don’t use Poetry, you can also run the script directly:

```bash
./scripts/codex_resume_plus.py
```

## Usage

Run from any repo directory (default behavior is to show sessions whose effective
working directory matches your current directory):

```bash
codex-resume-plus
```

Options:

- `--all`: show sessions from other directories (disables current-directory filtering)
- `--include-archived`: include archived sessions in the list
- `--preview {summary,full}`: initial preview mode (default: `summary`)

## Key bindings

| Key | Action |
| --- | ------ |
| `Enter` | Resume in Codex |
| `Ctrl-E` | Rename (sidecar title) |
| `Ctrl-O` | Set “resume workdir” override (sidecar `-C` root) |
| `Ctrl-X` | Archive (press twice to confirm) |
| `Ctrl-U` | Unarchive (only when `--include-archived`; press twice to confirm) |
| `Ctrl-Y` | Trust current directory (press twice to confirm) |
| `Ctrl-R` | Reload list |
| `Ctrl-T` | Summary preview |
| `Ctrl-V` | Full transcript preview |

## How the UI works

The picker has two panes:

- **List (left):** one row per session, showing last-updated time, a short id,
  and either a title you set or a derived label (plus optional CWD when using
  `--all`).
- **Preview (right):** a summary view by default:
  - metadata header (title, archive status, cwd, resume workdir override, etc.)
  - first meaningful user message (skipping injected boilerplate)
  - latest user + assistant messages

Toggle the preview with:

- `Ctrl-T`: summary (default)
- `Ctrl-V`: full transcript (still skips obvious boilerplate user messages)

## How it stays safe

- **Never rewrites rollout logs:** titles and workdir overrides live in sidecar
  files keyed by `session_id`.
- **Archive/unarchive is guarded:** operations require a double keypress and are
  blocked if the rollout file looks open (via `lsof`). If `lsof` is missing,
  archive/unarchive is treated as a hard block.
- **Trust updates are serialized:** updates to `~/.codex/config.toml` are written
  atomically and guarded by a lock file.

## Common workflows

- **Find + label:** run the picker, use the preview to confirm, then `Ctrl-E` to
  set a short title so it’s easy to spot next time.
- **Rapid cleanup:** `Ctrl-X` twice to archive a session you’re done with; use
  `--include-archived` + `Ctrl-U` twice to bring it back.
- **Resume into a different repo:** `Ctrl-O` to set a per-session workdir, then
  `Enter` to resume; optionally `Ctrl-Y` to trust the repo to reduce approval
  friction for future edits.

## Where data is stored

This tool does **not** rewrite Codex’s rollout logs. It writes sidecar files under
`~/.codex/`:

- `~/.codex/session_titles.json`: session id → title
- `~/.codex/session_cwd_overrides.json`: session id → resume root (used for `codex resume -C …`)

And it may update Codex’s config:

- `~/.codex/config.toml`: adds/updates `[projects."<path>"] trust_level = "trusted"`

## Safety notes

- Archiving moves JSONL files from `~/.codex/sessions/...` into `~/.codex/archived_sessions/`.
- To reduce corruption risk, archiving/unarchiving is blocked if the rollout file
  appears open (via `lsof`). If `lsof` is missing, the tool treats this as a hard
  block for archive/unarchive operations.

## Contributing

See `CONTRIBUTING.md`.

## License

MIT. See `LICENSE`.
