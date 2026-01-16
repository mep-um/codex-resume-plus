#!/usr/bin/env python3
"""
An interactive codex session picker with archiving and titles (codex_resume_plus).

This is intentionally a standalone helper (not a Codex slash command):
- Rollout logs are append-only and may be actively written by Codex.
- Archiving uses a safety guard (`lsof`) to reduce the risk of moving an active
  rollout file.

Key bindings (fzf):
- Enter: resume in Codex (`codex resume <session_id>`)
- Ctrl-E: rename (writes ~/.codex/session_titles.json; blank = keep; '-' = clear with confirm)
- Ctrl-O: set resume workdir (writes ~/.codex/session_cwd_overrides.json; '.' = current dir; '-' = clear with confirm)
- Ctrl-Y: trust current directory (press twice to confirm; updates ~/.codex/config.toml)
- Ctrl-X: archive (press twice to confirm; moves rollout JSONL to ~/.codex/archived_sessions)
- Ctrl-U: unarchive (only when --include-archived; press twice to confirm; moves rollout JSONL back under ~/.codex/sessions)
- Ctrl-R: reload list
- Ctrl-T: summary preview
- Ctrl-V: full transcript preview

Notes:
- Workdir override (Ctrl-O) is per-session. It changes how this tool resumes the
  selected session by passing a working directory and writable scope flags, so
  relative paths and edits land in the intended repo:
    codex resume --add-dir <workdir> -C <workdir> <id>
- Trust (Ctrl-Y) is global/persistent. It adds the current directory to the
  Codex config as trusted to reduce approval friction, but it does not change
  where a session resumes.

Design note:
- This tool intentionally does not pass `--sandbox`, `--ask-for-approval`, or
  `--search` to `codex resume`. Those settings come from your Codex config
  (`~/.codex/config.toml`) or whatever flags you pass when invoking `codex`.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import termios
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Sequence, TextIO, Tuple

FZF_DELIM = "\t"

IDE_CONTEXT_HEADER = "# Context from my IDE setup:"

FZF_FIELD_ROLLOUT_PATH = 1
FZF_FIELD_SESSION_ID = 6

# Matches fzf placeholders like `{}`, `{1}`, `{q}`, `{+}`.
_FZF_PLACEHOLDER_RE = re.compile(r"^\{[^}]*\}$")

ANSI_BOLD = "\x1b[1m"
ANSI_BOLD_OFF = "\x1b[22m"


@dataclasses.dataclass(frozen=True)
class SessionSummary:
    rollout_path: Path
    session_id: str
    cwd: Optional[str]
    effective_cwd: Optional[str]
    updated_ts: float
    title: Optional[str]
    first_user_message: Optional[str]
    last_message: Optional[str]
    last_message_type: Optional[str]  # "user_message" | "agent_message"

    def to_fzf_line(self, *, cwd_display: str) -> str:
        updated = _format_ts(self.updated_ts)
        name = _build_session_display_name(self.title, self.first_user_message)
        name_display = name
        if self.title:
            # Use SGR 22 to disable bold without resetting colors that fzf sets.
            name_display = f"{ANSI_BOLD}★ {name}{ANSI_BOLD_OFF}"
        id_short = self.session_id[-8:]
        # Field order:
        # 1) rollout_path (hidden, used for selection + preview)
        # 2) updated
        # 3) id_short
        # 4) name (title if present, else derived label)
        # 5) cwd display
        # 6) full session_id (hidden but searchable)
        fields = [
            str(self.rollout_path),
            updated,
            id_short,
            _one_line(name_display),
            cwd_display,
            self.session_id,
            # Keep the first user message searchable even when a custom title is set.
            # (Not displayed in the list UI; fzf still matches against the full line.)
            _one_line(self.first_user_message or "-"),
        ]
        return FZF_DELIM.join(fields)


def get_codex_home() -> Path:
    # RATIONALE: Allow overrides for testing / multi-profile setups.
    env = os.environ.get("CODEX_HOME")
    return Path(env).expanduser() if env else Path.home() / ".codex"


def get_codex_config_path() -> Path:
    return get_codex_home() / "config.toml"


@dataclasses.dataclass(frozen=True)
class _ProjectTrustTable:
    header_start: int
    header_end: int
    body_start: int
    body_end: int
    key_str: str  # unescaped TOML project key
    path: Path  # normalized
    trust_level: Optional[str]


_TOML_TABLE_HEADER_RE = re.compile(r"^\s*\[[^\]]+\]\s*(?:#.*)?$", re.MULTILINE)
_TOML_PROJECT_HEADER_RE = re.compile(
    r"^\s*\[projects\.(?P<quote>[\"'])(?P<path>.*?)(?P=quote)\]\s*(?:#.*)?$",
    re.MULTILINE,
)
_TOML_TRUST_LEVEL_RE = re.compile(
    r"^(?P<indent>\s*)trust_level\s*=\s*(?P<quote>[\"'])(?P<value>.*?)(?P=quote)(?P<rest>\s*(?:#.*)?)$",
    re.MULTILINE,
)


def _toml_unescape_basic_string(value: str) -> str:
    # RATIONALE: Codex project keys are filesystem paths; for macOS we only need a
    # minimal unescape for \" and \\ to support spaces/quotes safely.
    out: list[str] = []
    i = 0
    while i < len(value):
        ch = value[i]
        if ch == "\\" and i + 1 < len(value):
            nxt = value[i + 1]
            if nxt in {'"', "\\"}:
                out.append(nxt)
                i += 2
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _toml_escape_basic_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _path_is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _iter_project_trust_tables(config_text: str) -> List[_ProjectTrustTable]:
    tables: List[_ProjectTrustTable] = []
    for match in _TOML_PROJECT_HEADER_RE.finditer(config_text):
        header_start = match.start()
        header_end = match.end()

        body_start = header_end
        if body_start < len(config_text) and config_text[body_start] == "\n":
            body_start += 1

        next_header = _TOML_TABLE_HEADER_RE.search(config_text, pos=header_end)
        body_end = next_header.start() if next_header else len(config_text)

        quote = match.group("quote")
        raw_key = match.group("path")
        key_str = raw_key
        if quote == '"':
            key_str = _toml_unescape_basic_string(raw_key)
        else:
            # TOML literal strings don't use backslash escapes.
            key_str = raw_key.replace("''", "'")

        try:
            key_path = Path(key_str).expanduser().resolve()
        except Exception:
            continue

        segment = config_text[body_start:body_end]
        tl_match = _TOML_TRUST_LEVEL_RE.search(segment)
        trust_level = tl_match.group("value").strip() if tl_match else None
        tables.append(
            _ProjectTrustTable(
                header_start=header_start,
                header_end=header_end,
                body_start=body_start,
                body_end=body_end,
                key_str=key_str,
                path=key_path,
                trust_level=trust_level,
            )
        )
    return tables


def is_path_trusted(path: Path, *, config_path: Optional[Path] = None) -> bool:
    config_path = config_path or get_codex_config_path()
    try:
        config_text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    except OSError:
        return False

    try:
        resolved = path.expanduser().resolve()
    except Exception:
        return False

    trusted_roots: List[Path] = []
    for table in _iter_project_trust_tables(config_text):
        if (table.trust_level or "").lower() == "trusted":
            trusted_roots.append(table.path)

    # RATIONALE: Codex treats a trusted project root as trusted for its
    # subdirectories; reflect that in the UI.
    return any(_path_is_within(resolved, root) for root in trusted_roots)


@contextmanager
def _locked_codex_config_file(timeout_s: float = 5.0) -> Iterator[None]:
    # RATIONALE: We update `config.toml` via atomic replace (os.replace). Locking the
    # file itself doesn't serialize concurrent writers across a replace because the
    # lock is held on the old inode. Locking the Codex home directory keeps the lock
    # stable without creating a `config.toml.lock` file.
    lock_dir = get_codex_home()
    lock_dir.mkdir(parents=True, exist_ok=True)

    lock_fd: Optional[int] = None
    try:
        lock_fd = os.open(str(lock_dir), os.O_RDONLY)
        start = time.monotonic()
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - start > timeout_s:
                    raise RuntimeError(
                        f"Timed out waiting for lock on Codex home: {lock_dir} "
                        "(is another process updating the Codex config?)"
                    )
                time.sleep(0.1)
        yield
    finally:
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except OSError:
                pass


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 0o600
    try:
        mode = path.stat().st_mode & 0o777
    except FileNotFoundError:
        pass
    except OSError:
        pass
    if not text.endswith("\n"):
        text += "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
    ) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    os.chmod(tmp_path, mode)
    os.replace(tmp_path, path)


def ensure_path_trusted(path: Path, *, config_path: Optional[Path] = None) -> None:
    config_path = config_path or get_codex_config_path()
    try:
        resolved = path.expanduser().resolve()
    except Exception as exc:
        raise RuntimeError(f"Unable to resolve path: {path} ({exc})") from exc

    with _locked_codex_config_file():
        try:
            config_text = config_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            config_text = ""
        except OSError as exc:
            raise RuntimeError(f"Unable to read Codex config: {config_path} ({exc})")

        if is_path_trusted(resolved, config_path=config_path):
            return

        matching: Optional[_ProjectTrustTable] = None
        for table in _iter_project_trust_tables(config_text):
            if table.path == resolved:
                matching = table
                break

        if matching:
            segment = config_text[matching.body_start : matching.body_end]
            if _TOML_TRUST_LEVEL_RE.search(segment):
                segment = _TOML_TRUST_LEVEL_RE.sub(
                    r'\g<indent>trust_level = "trusted"\g<rest>', segment, count=1
                )
                config_text = (
                    config_text[: matching.body_start]
                    + segment
                    + config_text[matching.body_end :]
                )
            else:
                insert_at = matching.header_end
                if insert_at < len(config_text) and config_text[insert_at] == "\n":
                    insert_at += 1
                config_text = (
                    config_text[:insert_at]
                    + 'trust_level = "trusted"\n'
                    + config_text[insert_at:]
                )
        else:
            if config_text and not config_text.endswith("\n"):
                config_text += "\n"
            if config_text.strip():
                config_text += "\n"
            escaped = _toml_escape_basic_string(str(resolved))
            config_text += f'[projects."{escaped}"]\ntrust_level = "trusted"\n'

        _write_text_atomic(config_path, config_text)


def get_codex_sessions_dir() -> Path:
    return get_codex_home() / "sessions"


def get_codex_archived_sessions_dir() -> Path:
    return get_codex_home() / "archived_sessions"


def get_titles_path() -> Path:
    return get_codex_home() / "session_titles.json"


def get_titles_lock_path() -> Path:
    return get_codex_home() / "session_titles.json.lock"


def get_cwd_overrides_path() -> Path:
    return get_codex_home() / "session_cwd_overrides.json"


def get_cwd_overrides_lock_path() -> Path:
    return get_codex_home() / "session_cwd_overrides.json.lock"


def load_cwd_overrides() -> Dict[str, str]:
    overrides_path = get_cwd_overrides_path()
    if not overrides_path.exists():
        return {}
    try:
        data = json.loads(overrides_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # RATIONALE: Overrides are convenience-only. If the sidecar is corrupted,
        # return {} so core functionality keeps working, and a subsequent update
        # will rewrite the file atomically.
        return {}
    except OSError:
        return {}
    if not isinstance(data, dict):
        return {}
    overrides: Dict[str, str] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, str):
            overrides[key] = value
    return overrides


def list_rollouts(*, include_archived: bool) -> List[Path]:
    paths: List[Path] = []

    sessions_dir = get_codex_sessions_dir()
    if sessions_dir.exists():
        paths.extend(sorted(sessions_dir.glob("**/rollout-*.jsonl")))

    if include_archived:
        archived_dir = get_codex_archived_sessions_dir()
        if archived_dir.exists():
            paths.extend(sorted(archived_dir.glob("rollout-*.jsonl")))

    entries: List[Tuple[float, Path]] = []
    for p in paths:
        try:
            entries.append((p.stat().st_mtime, p))
        except FileNotFoundError:
            # RATIONALE: Rollouts can be moved concurrently (e.g., by an archive
            # operation). Missing files should not crash listing.
            continue
        except OSError:
            continue

    entries.sort(key=lambda item: item[0], reverse=True)
    return [p for _, p in entries]


def load_titles() -> Dict[str, str]:
    titles_path = get_titles_path()
    if not titles_path.exists():
        return {}
    try:
        data = json.loads(titles_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # RATIONALE: Titles are purely cosmetic, and we don't want a corrupted
        # sidecar file to break the picker. Returning {} preserves core
        # functionality. A subsequent rename will rewrite the file atomically.
        return {}
    except OSError:
        return {}
    if not isinstance(data, dict):
        return {}
    titles: Dict[str, str] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, str):
            titles[key] = value
    return titles


@contextmanager
def _locked_titles_file(timeout_s: float = 5.0) -> Iterator[None]:
    lock_path = get_titles_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        start = time.monotonic()
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - start > timeout_s:
                    raise RuntimeError(
                        f"Timed out waiting for lock: {lock_path} (is another "
                        "process updating titles?)"
                    )
                time.sleep(0.1)
        yield


def set_title(session_id: str, title: str) -> None:
    if not session_id:
        raise ValueError("session_id is required")
    title = title.strip()
    if not title:
        raise ValueError("title must be non-empty; use unset_title to remove")

    with _locked_titles_file():
        titles = load_titles()
        titles[session_id] = title
        _write_titles_atomic(titles)


def unset_title(session_id: str) -> bool:
    if not session_id:
        raise ValueError("session_id is required")
    with _locked_titles_file():
        titles = load_titles()
        existed = session_id in titles
        if existed:
            del titles[session_id]
            _write_titles_atomic(titles)
        return existed


def _write_titles_atomic(titles: Dict[str, str]) -> None:
    titles_path = get_titles_path()
    titles_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(titles, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(titles_path.parent),
        delete=False,
    ) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    os.chmod(tmp_path, 0o600)
    os.replace(tmp_path, titles_path)


@contextmanager
def _locked_cwd_overrides_file(timeout_s: float = 5.0) -> Iterator[None]:
    lock_path = get_cwd_overrides_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        start = time.monotonic()
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - start > timeout_s:
                    raise RuntimeError(
                        f"Timed out waiting for lock: {lock_path} (is another "
                        "process updating cwd overrides?)"
                    )
                time.sleep(0.1)
        yield


def set_cwd_override(session_id: str, cwd: str) -> None:
    if not session_id:
        raise ValueError("session_id is required")
    cwd = cwd.strip()
    if not cwd:
        raise ValueError(
            "cwd override must be non-empty; use unset_cwd_override to remove"
        )

    path = Path(cwd).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"cwd override must be an existing directory: {path}")

    with _locked_cwd_overrides_file():
        overrides = load_cwd_overrides()
        overrides[session_id] = str(path)
        _write_cwd_overrides_atomic(overrides)


def unset_cwd_override(session_id: str) -> bool:
    if not session_id:
        raise ValueError("session_id is required")
    with _locked_cwd_overrides_file():
        overrides = load_cwd_overrides()
        existed = session_id in overrides
        if existed:
            del overrides[session_id]
            _write_cwd_overrides_atomic(overrides)
        return existed


def _write_cwd_overrides_atomic(overrides: Dict[str, str]) -> None:
    overrides_path = get_cwd_overrides_path()
    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(overrides, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(overrides_path.parent),
        delete=False,
    ) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    os.chmod(tmp_path, 0o600)
    os.replace(tmp_path, overrides_path)


def build_session_summaries(
    *,
    include_archived: bool,
    include_all_cwds: bool,
    current_scope: Path,
) -> List[SessionSummary]:
    titles = load_titles()
    cwd_overrides = load_cwd_overrides()
    summaries: List[SessionSummary] = []

    for rollout_path in list_rollouts(include_archived=include_archived):
        meta = read_session_meta(rollout_path)
        if not meta:
            continue
        session_id, cwd = meta
        effective_cwd = cwd_overrides.get(session_id) or cwd
        if not include_all_cwds:
            # Default: only include sessions whose recorded `cwd` is within the
            # directory where the picker is launched.
            if not effective_cwd:
                continue
            if not _path_within_scope(current_scope, Path(effective_cwd)):
                continue

        try:
            updated_ts = rollout_path.stat().st_mtime
        except FileNotFoundError:
            continue
        first_user = read_first_user_message(rollout_path)
        last_msg_type, last_msg = read_last_message(rollout_path)
        summaries.append(
            SessionSummary(
                rollout_path=rollout_path,
                session_id=session_id,
                cwd=cwd,
                effective_cwd=effective_cwd,
                updated_ts=updated_ts,
                title=titles.get(session_id),
                first_user_message=first_user,
                last_message=last_msg,
                last_message_type=last_msg_type,
            )
        )

    summaries.sort(key=lambda s: s.updated_ts, reverse=True)
    return summaries


def get_current_scope() -> Path:
    # RATIONALE: By default we filter to sessions from the current directory.
    # Use `--all` to disable filtering.
    return Path.cwd().resolve()


def read_session_meta(rollout_path: Path) -> Optional[Tuple[str, Optional[str]]]:
    for obj in _iter_jsonl(rollout_path):
        if obj.get("type") != "session_meta":
            continue
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            return None
        session_id = payload.get("id")
        cwd = payload.get("cwd")
        if not isinstance(session_id, str) or not session_id:
            return None
        if cwd is not None and not isinstance(cwd, str):
            cwd = None
        return session_id, cwd
    return None


def read_first_user_message(rollout_path: Path) -> Optional[str]:
    msg = read_first_user_message_text(rollout_path)
    if not msg:
        return None
    return _truncate(_one_line(msg), 240)


def read_first_user_message_text(rollout_path: Path) -> Optional[str]:
    """
    Return the first meaningful user message in the rollout (for previews/labels).

    Supports both legacy `event_msg` format and newer `response_item` format.
    """
    for obj in _iter_jsonl(rollout_path):
        extracted = _extract_message_record(obj)
        if not extracted:
            continue
        msg_type, msg = extracted
        if msg_type != "user_message":
            continue
        msg = sanitize_message_for_preview(msg)
        if not _is_probable_boilerplate_user_message(msg):
            return msg
    return None


def read_last_message(rollout_path: Path) -> Tuple[Optional[str], Optional[str]]:
    # RATIONALE: Some sessions can be large; scanning from the end avoids reading
    # the entire file just to get the last message snippet.
    for line in _iter_lines_reversed(rollout_path):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        extracted = _extract_message_record(obj)
        if not extracted:
            continue
        msg_type, msg = extracted
        msg = sanitize_message_for_preview(msg)
        return msg_type, _truncate(_one_line(msg), 240)
    return None, None


def read_last_user_and_agent_messages(
    rollout_path: Path,
) -> Tuple[Optional[str], Optional[str]]:
    last_user_non_boilerplate: Optional[str] = None
    last_agent: Optional[str] = None

    for line in _iter_lines_reversed(rollout_path):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        extracted = _extract_message_record(obj)
        if not extracted:
            continue
        msg_type, msg = extracted
        msg = sanitize_message_for_preview(msg)

        if msg_type == "user_message":
            if (
                last_user_non_boilerplate is None
                and not _is_probable_boilerplate_user_message(msg)
            ):
                last_user_non_boilerplate = msg
        elif msg_type == "agent_message" and last_agent is None:
            last_agent = msg

        if last_user_non_boilerplate is not None and last_agent is not None:
            break

    return last_user_non_boilerplate, last_agent


def require_fzf() -> None:
    if shutil_which("fzf") is None:
        raise RuntimeError("fzf is required but was not found on PATH")


def require_macos() -> None:
    if sys.platform != "darwin":
        raise RuntimeError(
            "This helper is macOS-only (expected sys.platform == 'darwin')"
        )


def shutil_which(name: str) -> Optional[str]:
    # Avoid importing shutil just for which() in hot preview paths.
    for p in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(p) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def shell_join_fzf_command(parts: Sequence[str]) -> str:
    """
    Join a command template for fzf actions (execute/reload/preview/become).

    RATIONALE:
    - Fixed argv parts must be shell-quoted to survive spaces, etc.
    - fzf placeholder expressions (e.g., {}, {1}, {q}, {+}) must be left
      unquoted because fzf expands them to already-quoted strings.
    """
    rendered: List[str] = []
    for part in parts:
        if _FZF_PLACEHOLDER_RE.match(part):
            rendered.append(part)
        else:
            rendered.append(shlex.quote(part))
    return " ".join(rendered)


def _iter_jsonl(path: Path) -> Iterator[dict]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except FileNotFoundError:
        return


def _iter_lines_reversed(path: Path, block_size: int = 64 * 1024) -> Iterator[str]:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            position = f.tell()
            buffer = b""
            while position > 0:
                read_size = block_size if position >= block_size else position
                position -= read_size
                f.seek(position)
                chunk = f.read(read_size)
                buffer = chunk + buffer
                lines = buffer.split(b"\n")
                buffer = lines[0]
                for raw in reversed(lines[1:]):
                    yield raw.decode("utf-8", errors="replace")
            if buffer:
                yield buffer.decode("utf-8", errors="replace")
    except FileNotFoundError:
        return


def _path_within_scope(scope: Path, candidate: Path) -> bool:
    try:
        scope_resolved = scope.resolve()
        candidate_resolved = candidate.resolve()
        return (
            candidate_resolved == scope_resolved
            or candidate_resolved.is_relative_to(scope_resolved)
        )
    except Exception:
        return False


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _build_session_display_name(title: Optional[str], label: Optional[str]) -> str:
    title = title.strip() if title else ""
    label = label.strip() if label else ""

    if title:
        return _truncate(title, 120)
    if label:
        return _truncate(label, 120)
    return "(boilerplate-only)"


def _extract_message_text(payload: dict) -> Optional[str]:
    msg = payload.get("message") or payload.get("text")
    if isinstance(msg, str) and msg.strip():
        return msg
    return None


def _extract_response_item_message_text(payload: dict) -> Optional[str]:
    content = payload.get("content")
    if not isinstance(content, list):
        return None

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text)
    if not parts:
        return None
    return "\n".join(parts)


def _extract_message_record(obj: dict) -> Optional[Tuple[str, str]]:
    """
    Return `(msg_type, text)` for message records.

    msg_type is normalized to:
    - "user_message" for user content
    - "agent_message" for assistant/agent content
    """
    obj_type = obj.get("type")
    if obj_type == "event_msg":
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            return None
        msg_type = payload.get("type")
        if msg_type not in ("user_message", "agent_message"):
            return None
        msg = _extract_message_text(payload)
        if not msg:
            return None
        return msg_type, msg

    if obj_type == "response_item":
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            return None
        if payload.get("type") != "message":
            return None
        role = payload.get("role")
        if role not in ("user", "assistant"):
            return None
        msg = _extract_response_item_message_text(payload)
        if not msg:
            return None
        msg_type = "user_message" if role == "user" else "agent_message"
        return msg_type, msg

    return None


def _is_probable_boilerplate_user_message(message: str) -> bool:
    # RATIONALE: Many Codex logs include preamble "messages" that are actually
    # injected instructions/environment context. These are rarely useful as
    # human session labels, so we skip them when possible.
    stripped = message.lstrip()
    return stripped.startswith("# AGENTS.md instructions for ") or stripped.startswith(
        "<environment_context>"
    )


def sanitize_message_for_preview(message: str) -> str:
    """
    Remove noisy editor-context sections from the message for list/preview display.

    RATIONALE: Codex sessions often include an auto-injected "Context from my IDE"
    block with long lists of open documents/tabs and images. This is useful for the
    model but makes it harder for humans to quickly identify a session in a picker.
    """
    if IDE_CONTEXT_HEADER not in message:
        return message

    lines = message.splitlines()
    kept: list[str] = []
    in_context = False
    skipping_section = False

    def is_heading(line: str) -> bool:
        return line.strip().startswith("## ")

    def should_skip_heading(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("## Active file:") or stripped in {
            "## Open tabs:",
            "## Documents:",
            "## Images:",
        }

    for line in lines:
        if not in_context and line.strip() == IDE_CONTEXT_HEADER:
            in_context = True
            continue

        if in_context and is_heading(line):
            skipping_section = should_skip_heading(line)
            if skipping_section:
                continue

        if skipping_section:
            # Skip bullets/continuation lines until the next heading.
            if is_heading(line):
                skipping_section = should_skip_heading(line)
                if skipping_section:
                    continue
                kept.append(line)
            continue

        kept.append(line)

    # RATIONALE: IDE context templates sometimes include headings like
    # "## My request for Codex:" which are repeated across sessions and don't
    # help identify the content.
    kept = [
        ln
        for ln in kept
        if ln.strip() not in {"## My request for Codex:", "## My request:"}
    ]

    sanitized = "\n".join(kept).strip()
    return sanitized or message


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


ROLLOUT_FILENAME_RE = re.compile(
    r"^rollout-(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})-(?P<id>[0-9a-fA-F-]{36})\.jsonl$"
)


def _parse_started_ts_from_rollout_filename(filename: str) -> Optional[dt.datetime]:
    match = ROLLOUT_FILENAME_RE.match(filename)
    if not match:
        return None
    ts_str = match.group("ts")
    try:
        return dt.datetime.strptime(ts_str, "%Y-%m-%dT%H-%M-%S")
    except ValueError:
        return None


def _format_dt(value: dt.datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M")


def _format_age_seconds(seconds: float) -> str:
    if seconds < 0:
        return "0s ago"
    total = int(seconds)
    if total < 60:
        return f"{total}s ago"

    minutes, sec = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m ago"

    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        if minutes:
            return f"{hours}h {minutes}m ago"
        return f"{hours}h ago"

    days, hours = divmod(hours, 24)
    if days < 7:
        if hours:
            return f"{days}d {hours}h ago"
        return f"{days}d ago"

    weeks, days = divmod(days, 7)
    if days:
        return f"{weeks}w {days}d ago"
    return f"{weeks}w ago"


def _archive_status_for_preview(rollout_path: Path) -> Tuple[str, str]:
    archived_dir = get_codex_archived_sessions_dir()
    try:
        is_archived = rollout_path.resolve().is_relative_to(archived_dir.resolve())
    except Exception:
        is_archived = False

    if is_archived:
        return "N/A", "(already archived)"

    lsof_available, pids = lsof_pids(rollout_path)
    if pids:
        return "BLOCK", f"(open by pid(s): {', '.join(pids)})"
    if not lsof_available:
        # RATIONALE: Treat missing `lsof` as a hard block so we don't move a
        # potentially active rollout file.
        return "BLOCK", "(lsof not found)"

    return "OK", ""


def lsof_pids(path: Path) -> Tuple[bool, List[str]]:
    try:
        proc = subprocess.run(
            ["lsof", "-t", "--", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        return False, []
    pids = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return True, pids


def assert_rollout_not_open(rollout_path: Path, *, action: str) -> None:
    """
    Raise RuntimeError if `rollout_path` is currently open or cannot be checked.

    RATIONALE: Moving a rollout file that is currently open by another process
    can be risky (and actively written rollouts can be corrupted if moved).
    """
    lsof_available, pids = lsof_pids(rollout_path)
    if not lsof_available:
        raise RuntimeError(
            f"Refusing to {action} because `lsof` was not found on PATH, so we can't "
            "verify the rollout file isn't currently open."
        )
    if pids:
        raise RuntimeError(
            f"Refusing to {action} an open rollout file "
            f"(pids: {', '.join(pids)}): {rollout_path}"
        )

    try:
        rollout_path.stat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Rollout file no longer exists: {rollout_path}") from exc


def assert_rollout_safe_to_archive(
    rollout_path: Path,
) -> None:
    """
    Raise RuntimeError if it's unsafe to archive the rollout.

    RATIONALE: Moving a rollout file that Codex is actively writing can corrupt
    the session. We use `lsof` as a guard to block files that are currently open.
    """
    assert_rollout_not_open(rollout_path, action="archive")


def _print_message_block(
    label: str,
    message: Optional[str],
    *,
    max_lines: int,
    head_tail: bool,
) -> None:
    print(f"{label}:")
    if not message:
        print("  -")
        return

    lines = [ln.rstrip() for ln in message.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        print("  -")
        return

    truncated = False
    shown: List[str]

    if head_tail and len(lines) > max_lines and max_lines >= 4:
        head_count = max_lines // 2
        tail_count = max_lines - head_count - 1
        shown = lines[:head_count] + ["…"] + lines[-tail_count:]
        truncated = True
    else:
        shown = lines[:max_lines]
        truncated = len(lines) > max_lines

    for line in shown:
        print(f"  {line}")

    if truncated:
        suffix = f"(showing {len(shown)} of {len(lines)} lines)"
        if head_tail and max_lines >= 4:
            print(f"  {suffix}")
        else:
            print(f"  … {suffix}")


def _format_ts(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _format_cwd_for_list(cwd: Optional[str], scope: Path) -> str:
    if not cwd:
        return "-"
    p = Path(cwd).expanduser()
    try:
        # If the session cwd is within the current scope, keep it short.
        rel = p.resolve().relative_to(scope.resolve())
        return f"./{rel}" if str(rel) != "." else "."
    except Exception:
        return _abbrev_home(p)


def _abbrev_home(path: Path) -> str:
    home = Path.home().resolve()
    try:
        rel = path.resolve().relative_to(home)
    except Exception:
        return str(path)
    return str(Path("~") / rel)


def _format_meta_block(path: Path) -> List[str]:
    meta = read_session_meta(path)
    if not meta:
        return [f"PATH: {_abbrev_home(path)}", "TITLE: (unknown)"]

    session_id, cwd = meta

    try:
        updated_ts = path.stat().st_mtime
    except FileNotFoundError:
        return [f"PATH: {_abbrev_home(path)}", "TITLE: (missing file)"]

    started_ts = _parse_started_ts_from_rollout_filename(path.name)
    title = load_titles().get(session_id)
    cwd_override = load_cwd_overrides().get(session_id)

    status, status_note = _archive_status_for_preview(path)

    started_str = _format_dt(started_ts) if started_ts else "-"
    updated_str = _format_ts(updated_ts)
    age_str = _format_age_seconds(time.time() - updated_ts)

    scope = get_current_scope()
    recorded_display = _format_cwd_for_list(cwd, scope)
    override_display = _format_cwd_for_list(cwd_override, scope)

    archive_line = f"ARCHIVE: {status}"
    if status_note:
        archive_line += f" {status_note}"

    lines = [
        f"TITLE: {title or '(not set)'}",
        archive_line,
        f"WHEN: started {started_str}  |  updated {updated_str} ({age_str})",
    ]

    if cwd:
        cwd_abbrev = _abbrev_home(Path(cwd).expanduser())
        cwd_line = f"CWD: {cwd_abbrev}"
        if recorded_display not in {cwd_abbrev, "-", ""}:
            cwd_line += f" ({recorded_display})"
        lines.append(cwd_line)
    else:
        lines.append("CWD: -")

    if cwd_override:
        override_abbrev = _abbrev_home(Path(cwd_override).expanduser())
        override_line = f"RESUME -C: {override_abbrev}"
        if override_display not in {override_abbrev, "-", ""}:
            override_line += f" ({override_display})"
        lines.append(override_line)
    else:
        lines.append("RESUME -C: -")

    lines.append(f"ID: {session_id}")
    lines.append(f"PATH: {_abbrev_home(path)}")
    return lines


def _preview(path: Path, *, mode: str) -> int:
    meta = read_session_meta(path)
    if not meta:
        print("Unable to read session meta.")
        return 1

    for line in _format_meta_block(path):
        print(line)
    print()

    if mode == "full":
        printed = 0
        for obj in _iter_jsonl(path):
            extracted = _extract_message_record(obj)
            if not extracted:
                continue
            msg_type, msg = extracted
            msg = sanitize_message_for_preview(msg)
            if msg_type == "user_message" and _is_probable_boilerplate_user_message(
                msg
            ):
                continue
            label = "User" if msg_type == "user_message" else "Assistant"
            print(f"{label}:")
            for line in msg.splitlines():
                print(f"  {line.rstrip()}")
            print()
            printed += 1
        if printed == 0:
            print("No non-boilerplate messages found.")
        return 0

    # Summary preview (default).
    first = read_first_user_message_text(path)
    last_user, last_agent = read_last_user_and_agent_messages(path)
    if first and last_user and _one_line(first) == _one_line(last_user):
        last_user = None

    _print_message_block(
        "First user",
        first or "(boilerplate-only)",
        max_lines=10,
        head_tail=False,
    )
    if last_user:
        print()
        _print_message_block("Last user", last_user, max_lines=12, head_tail=True)
    print()
    _print_message_block("Last assistant", last_agent, max_lines=12, head_tail=True)
    return 0


def _list_sessions(
    *, include_archived: bool, include_all_cwds: bool, show_cwd: bool
) -> int:
    scope = get_current_scope()
    summaries = build_session_summaries(
        include_archived=include_archived,
        include_all_cwds=include_all_cwds,
        current_scope=scope,
    )
    try:
        for s in summaries:
            cwd_display = (
                _format_cwd_for_list(s.effective_cwd, scope) if show_cwd else "-"
            )
            print(s.to_fzf_line(cwd_display=cwd_display))
    except BrokenPipeError:
        # Common when piping to `head`; treat as normal termination.
        try:
            sys.stdout.close()
        except Exception:
            pass
        return 0
    return 0


_INTERNAL_COMMANDS = {
    "list",
    "preview",
    "rename",
    "cwd-override",
    "fzf-trust-pwd",
    "fzf-archive",
    "fzf-unarchive",
    "fzf-refresh-ui",
}


def _parse_public_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show sessions from other directories (disables current-directory filtering).",
    )
    parser.add_argument(
        "--include-archived",
        action="store_true",
        help="Include archived sessions in the picker list.",
    )
    parser.add_argument(
        "--preview",
        choices=["summary", "full"],
        default="summary",
        help="Initial preview mode (default: summary).",
    )
    parser.add_argument(
        "--no-mouse",
        action="store_true",
        help=(
            "Disable mouse support in fzf (enables normal click-and-drag selection/"
            "copy, but disables mouse-wheel scrolling in the UI)."
        ),
    )
    return parser.parse_args(argv)


def _parse_internal_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    sub = parser.add_subparsers(dest="cmd", required=True)

    preview = sub.add_parser("preview")  # internal fzf hook
    preview.add_argument("--path", required=True)
    preview.add_argument(
        "--mode",
        choices=["summary", "full"],
        default="summary",
        help="Preview mode: summary (default) or full transcript.",
    )

    list_cmd = sub.add_parser("list")  # internal fzf hook
    list_cmd.add_argument(
        "--all",
        action="store_true",
        help="Show sessions from other directories (disables current-directory filtering).",
    )
    list_cmd.add_argument(
        "--include-archived",
        action="store_true",
        help="Include archived sessions.",
    )
    list_cmd.add_argument(
        "--show-cwd",
        action="store_true",
        help="Show a CWD column (useful with --all).",
    )

    rename = sub.add_parser("rename")  # internal fzf hook
    rename.add_argument("--session-id", required=True)
    rename.add_argument(
        "--title",
        help="Set the title without prompting (use --unset to clear).",
    )
    rename.add_argument(
        "--unset",
        action="store_true",
        help="Remove any stored title for the session.",
    )

    cwd_override = sub.add_parser("cwd-override")  # internal fzf hook
    cwd_override.add_argument("--session-id", required=True)
    cwd_override.add_argument(
        "--cwd",
        help="Set the resume root (-C) without prompting (use --unset to clear).",
    )
    cwd_override.add_argument(
        "--unset",
        action="store_true",
        help="Remove any stored resume-root override for the session.",
    )

    fzf_archive = sub.add_parser("fzf-archive")
    fzf_archive.add_argument("--path", required=True)
    fzf_archive.add_argument("--session-id", required=True)
    fzf_archive.add_argument("--all", action="store_true")
    fzf_archive.add_argument("--include-archived", action="store_true")

    fzf_unarchive = sub.add_parser("fzf-unarchive")
    fzf_unarchive.add_argument("--path", required=True)
    fzf_unarchive.add_argument("--session-id", required=True)
    fzf_unarchive.add_argument("--all", action="store_true")
    fzf_unarchive.add_argument("--include-archived", action="store_true")

    fzf_trust_pwd = sub.add_parser("fzf-trust-pwd")
    fzf_trust_pwd.add_argument("--include-archived", action="store_true")

    fzf_refresh_ui = sub.add_parser("fzf-refresh-ui")
    fzf_refresh_ui.add_argument("--all", action="store_true")
    fzf_refresh_ui.add_argument("--include-archived", action="store_true")

    return parser.parse_args(argv)


_ARCHIVE_ARM_TTL_S = 10.0
_ARCHIVE_ARM_STATE_FILENAME = ".codex_resume_plus_archive_arm.json"

_UNARCHIVE_ARM_TTL_S = 10.0
_UNARCHIVE_ARM_STATE_FILENAME = ".codex_resume_plus_unarchive_arm.json"

_TRUST_ARM_STATE_FILENAME = ".codex_resume_plus_trust_arm.json"


def _trust_arm_state_path() -> Path:
    return get_codex_home() / _TRUST_ARM_STATE_FILENAME


def _clear_trust_arm() -> None:
    try:
        _trust_arm_state_path().unlink()
    except FileNotFoundError:
        pass


def _read_trust_arm() -> Optional[str]:
    path = _trust_arm_state_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    cwd = data.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        return None
    return cwd


def _write_trust_arm(cwd: str) -> None:
    state_path = _trust_arm_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"cwd": cwd, "armed_at": time.time()})
    state_path.write_text(payload, encoding="utf-8")


def _archive_arm_state_path() -> Path:
    return get_codex_home() / _ARCHIVE_ARM_STATE_FILENAME


def _clear_archive_arm() -> None:
    try:
        _archive_arm_state_path().unlink()
    except FileNotFoundError:
        pass


def _read_archive_arm() -> tuple[Optional[str], Optional[float]]:
    path = _archive_arm_state_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, None
    except Exception:
        return None, None
    if not isinstance(data, dict):
        return None, None
    session_id = data.get("session_id")
    armed_at = data.get("armed_at")
    if not isinstance(session_id, str) or not session_id:
        return None, None
    if not isinstance(armed_at, (int, float)):
        return None, None
    return session_id, float(armed_at)


def _write_archive_arm(session_id: str) -> None:
    state_path = _archive_arm_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"session_id": session_id, "armed_at": time.time()})
    state_path.write_text(payload, encoding="utf-8")


def _unarchive_arm_state_path() -> Path:
    return get_codex_home() / _UNARCHIVE_ARM_STATE_FILENAME


def _clear_unarchive_arm() -> None:
    try:
        _unarchive_arm_state_path().unlink()
    except FileNotFoundError:
        pass


def _read_unarchive_arm() -> tuple[Optional[str], Optional[float]]:
    path = _unarchive_arm_state_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, None
    except Exception:
        return None, None
    if not isinstance(data, dict):
        return None, None
    session_id = data.get("session_id")
    armed_at = data.get("armed_at")
    if not isinstance(session_id, str) or not session_id:
        return None, None
    if not isinstance(armed_at, (int, float)):
        return None, None
    return session_id, float(armed_at)


def _write_unarchive_arm(session_id: str) -> None:
    state_path = _unarchive_arm_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"session_id": session_id, "armed_at": time.time()})
    state_path.write_text(payload, encoding="utf-8")


def _build_list_cmd(
    *, script_path: Path, show_all: bool, include_archived: bool
) -> str:
    parts: list[str] = [
        sys.executable,
        str(script_path),
        "list",
        *(["--all"] if show_all else []),
        *(["--include-archived"] if include_archived else []),
        *(["--show-cwd"] if show_all else []),
    ]
    return shell_join_fzf_command(parts)


def _archive_rollout_no_prompt(src: Path) -> Optional[str]:
    if not src.exists():
        return f"Rollout file no longer exists: {src}"

    dest_dir = get_codex_archived_sessions_dir()
    try:
        src.resolve().relative_to(dest_dir.resolve())
    except ValueError:
        pass
    else:
        return "Already archived."

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        return f"Archive destination already exists: {dest}"

    try:
        assert_rollout_safe_to_archive(src)
    except RuntimeError as exc:
        return str(exc)

    try:
        shutil.move(str(src), str(dest))
    except OSError as exc:
        return f"Failed to archive session: {exc}"

    return None


def _is_archived_rollout(path: Path) -> bool:
    archived_dir = get_codex_archived_sessions_dir()
    try:
        return path.resolve().is_relative_to(archived_dir.resolve())
    except Exception:
        return False


def _default_unarchive_dest(src: Path) -> Path:
    """
    Compute the default unarchive destination under ~/.codex/sessions/YYYY/MM/DD.

    Falls back to ~/.codex/sessions/unknown if the filename doesn't match.
    """
    sessions_dir = get_codex_sessions_dir()
    name = src.name

    # Expected: rollout-YYYY-MM-DDTHH-MM-SS-<uuid>.jsonl
    if name.startswith("rollout-") and "T" in name:
        date_part = name[len("rollout-") :].split("T", 1)[0]
        parts = date_part.split("-")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            year, month, day = parts
            return sessions_dir / year / month / day / name

    return sessions_dir / "unknown" / name


def _unarchive_rollout_no_prompt(src: Path) -> Optional[str]:
    if not src.exists():
        return f"Rollout file no longer exists: {src}"

    if not _is_archived_rollout(src):
        return "Not archived."

    dest = _default_unarchive_dest(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return f"Unarchive destination already exists: {dest}"

    try:
        assert_rollout_not_open(src, action="unarchive")
    except RuntimeError as exc:
        return str(exc)

    try:
        shutil.move(str(src), str(dest))
    except OSError as exc:
        return f"Failed to unarchive session: {exc}"

    return None


def _fzf_archive(
    *,
    session_id: str,
    rollout_path: str,
    show_all: bool,
    include_archived: bool,
) -> int:
    now = time.time()
    armed_session_id, armed_at = _read_archive_arm()
    is_armed = (
        armed_session_id == session_id
        and armed_at is not None
        and (now - armed_at) <= _ARCHIVE_ARM_TTL_S
    )

    if not is_armed:
        _write_archive_arm(session_id)
        short = session_id[-8:]
        print(f"change-header(Confirm archive {short}: press ^X again)")
        return 0

    _clear_archive_arm()
    error = _archive_rollout_no_prompt(Path(rollout_path).expanduser())
    if error:
        message = " ".join(str(error).split())
        if len(message) > 140:
            message = message[:139] + "…"
        print(f"change-header(Archive blocked: {message})")
        return 0

    list_cmd = _build_list_cmd(
        script_path=Path(__file__).resolve(),
        show_all=show_all,
        include_archived=include_archived,
    )
    header = _build_fzf_header(
        include_archived=include_archived,
        cwd_trusted=is_path_trusted(Path.cwd().resolve()),
    )
    print(f"reload-sync({list_cmd})+change-header({header})")
    return 0


def _fzf_unarchive(
    *,
    session_id: str,
    rollout_path: str,
    show_all: bool,
    include_archived: bool,
) -> int:
    now = time.time()
    armed_session_id, armed_at = _read_unarchive_arm()
    is_armed = (
        armed_session_id == session_id
        and armed_at is not None
        and (now - armed_at) <= _UNARCHIVE_ARM_TTL_S
    )

    if not is_armed:
        _write_unarchive_arm(session_id)
        short = session_id[-8:]
        print(f"change-header(Confirm unarchive {short}: press ^U again)")
        return 0

    _clear_unarchive_arm()
    error = _unarchive_rollout_no_prompt(Path(rollout_path).expanduser())
    if error:
        message = " ".join(str(error).split())
        if len(message) > 140:
            message = message[:139] + "…"
        print(f"change-header(Unarchive blocked: {message})")
        return 0

    list_cmd = _build_list_cmd(
        script_path=Path(__file__).resolve(),
        show_all=show_all,
        include_archived=include_archived,
    )
    header = _build_fzf_header(
        include_archived=include_archived,
        cwd_trusted=is_path_trusted(Path.cwd().resolve()),
    )
    print(f"reload-sync({list_cmd})+change-header({header})")
    return 0


def _open_tty(mode: Literal["r", "w"]) -> TextIO:
    return open("/dev/tty", mode, encoding="utf-8", errors="replace")


def _ensure_fd_blocking(fd: int) -> None:
    # RATIONALE: Some TUIs (including fzf) can flip O_NONBLOCK on stdio fds.
    # Those flags live on the shared file description, so they can leak into
    # the next process even after the child exits. Codex expects blocking IO.
    try:
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        if flags & os.O_NONBLOCK:
            fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
    except Exception:
        pass


def _ensure_fd_inheritable(fd: int) -> None:
    # RATIONALE: CLOEXEC on 0/1/2 can make the next exec'd process behave like
    # it has no TTY (or has stdout/stderr closed), which breaks TUIs.
    try:
        flags = fcntl.fcntl(fd, fcntl.F_GETFD)
        if flags & fcntl.FD_CLOEXEC:
            fcntl.fcntl(fd, fcntl.F_SETFD, flags & ~fcntl.FD_CLOEXEC)
    except Exception:
        pass


def _prepare_terminal_for_exec() -> None:
    """
    Best-effort reset before `execvp` into Codex.

    RATIONALE: `fzf` (and its preview window) can leave the terminal in a state
    that confuses the next TUI (e.g. stale scroll region or pending input). This
    can make `codex resume` appear to stop replaying history partway through
    even though the session data is present.
    """
    for fd in (0, 1, 2):
        _ensure_fd_blocking(fd)
        _ensure_fd_inheritable(fd)

    # Flush any queued keypresses (e.g. from fzf) so Codex doesn't treat them as
    # interactive input during resume/replay. Prefer `/dev/tty` because stdin
    # might not be the controlling TTY in all launch modes.
    try:
        with _open_tty("r") as tty_in:
            termios.tcflush(tty_in.fileno(), termios.TCIOFLUSH)
    except Exception:
        try:
            termios.tcflush(sys.stdin.fileno(), termios.TCIOFLUSH)
        except Exception:
            pass

    # Reset terminal modes (echo/canonical/etc) without relying on shell state.
    try:
        with _open_tty("r") as tty_in:
            subprocess.run(
                ["stty", "sane"],
                stdin=tty_in,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
    except Exception:
        pass

    # Reset formatting + scrolling region, ensure the cursor is visible, and
    # disable a few common terminal modes that can leak across TUIs.
    try:
        with _open_tty("w") as tty_out:
            tty_out.write(
                # Exit alternate screen, if enabled (common for TUIs).
                "\x1b[?1049l\x1b[?1047l\x1b[?47l"
                # Disable bracketed paste and common mouse reporting modes.
                "\x1b[?2004l"
                "\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?1015l"
                # Disable kitty keyboard protocol if a previous app enabled it
                # and crashed before restoring. Otherwise Ctrl-C, etc. may be
                # delivered as CSI u sequences rather than SIGINT.
                "\x1b[<u"
                # Reset attributes and scroll region; show cursor.
                "\x1b[0m\x1b[r\x1b[?25h"
                # Overwrite any previously-saved cursor position. Some copied
                # text can contain CSI ... u (cursor restore) sequences (e.g.
                # kitty keyboard protocol escapes). If fzf left a saved cursor
                # location behind, replaying that text can jump the cursor and
                # make the transcript look truncated.
                "\x1b[s"
            )
            tty_out.flush()
    except Exception:
        pass


def _read_from_tty(prompt: str) -> str:
    with _open_tty("r") as tty_in, _open_tty("w") as tty_out:
        tty_out.write(prompt)
        tty_out.flush()
        return tty_in.readline().rstrip("\n")


def _build_fzf_header(*, include_archived: bool, cwd_trusted: bool) -> str:
    base = "Enter resume | ^E rename | ^O workdir | ^X archive×2"
    if include_archived:
        base += " | ^U unarchive×2"
    if not cwd_trusted:
        base += " | ^Y trust pwd×2"
    base += " | ^R reload | ^T sum | ^V full"
    return base


def _build_fzf_prompt(*, cwd_trusted: bool) -> str:
    status = "trusted" if cwd_trusted else "untrusted"
    return f"codex_resume_plus[{status}]> "


def _rename_one(*, session_id: str, title: Optional[str], unset: bool) -> int:
    try:
        if unset:
            existed = unset_title(session_id)
            if existed:
                print(f"Cleared title for {session_id}.", file=sys.stderr)
            else:
                print(f"No title stored for {session_id}.", file=sys.stderr)
            return 0

        existing_title = load_titles().get(session_id)

        if title is None:
            prompt = "New title (blank = keep; '-' = clear)"
            if existing_title:
                existing_display = " ".join(existing_title.split())
                prompt += f" [currently: {existing_display}]"
            prompt += ": "
            title = _read_from_tty(prompt).strip()

        if title == "-":
            if not existing_title:
                print(f"No title stored for {session_id}.", file=sys.stderr)
                return 0

            existing_display = " ".join(existing_title.split())
            confirm = _read_from_tty(
                f'Clear title "{existing_display}"? [y/N]: '
            ).strip()
            if confirm.lower() not in {"y", "yes"}:
                print("Rename canceled.", file=sys.stderr)
                return 0

            existed = unset_title(session_id)
            if existed:
                print(f"Cleared title for {session_id}.", file=sys.stderr)
            else:
                print(f"No title stored for {session_id}.", file=sys.stderr)
            return 0

        if not title:
            print("No change.", file=sys.stderr)
            return 0

        set_title(session_id, title)
        print(f"Set title for {session_id}: {title}", file=sys.stderr)
        return 0
    except KeyboardInterrupt:
        print("Rename canceled.", file=sys.stderr)
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _cwd_override_one(*, session_id: str, cwd: Optional[str], unset: bool) -> int:
    try:
        if unset:
            existed = unset_cwd_override(session_id)
            if existed:
                print(f"Cleared resume root for {session_id}.", file=sys.stderr)
            else:
                print(f"No resume root stored for {session_id}.", file=sys.stderr)
            return 0

        existing = load_cwd_overrides().get(session_id)

        if cwd is None:
            prompt = "Resume root (-C) override (blank = cancel; '.' = current dir; '-' = clear)"
            if existing:
                existing_display = " ".join(existing.split())
                prompt += f" [currently: {existing_display}]"
            prompt += ": "
            cwd = _read_from_tty(prompt).strip()

        if not cwd:
            print("Canceled.", file=sys.stderr)
            return 0

        if cwd == ".":
            cwd = str(Path.cwd().resolve())

        if cwd == "-":
            if not existing:
                print(f"No resume root stored for {session_id}.", file=sys.stderr)
                return 0

            existing_display = " ".join(existing.split())
            confirm = _read_from_tty(
                f'Clear resume root "{existing_display}"? [y/N]: '
            ).strip()
            if confirm.lower() not in {"y", "yes"}:
                print("Canceled.", file=sys.stderr)
                return 0

            existed = unset_cwd_override(session_id)
            if existed:
                print(f"Cleared resume root for {session_id}.", file=sys.stderr)
            else:
                print(f"No resume root stored for {session_id}.", file=sys.stderr)
            return 0

        set_cwd_override(session_id, cwd)
        resolved = Path(cwd).expanduser().resolve()
        print(f"Set resume root for {session_id}: {resolved}", file=sys.stderr)
        return 0
    except KeyboardInterrupt:
        print("Canceled.", file=sys.stderr)
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _run_picker(args: argparse.Namespace) -> int:
    require_fzf()

    self_script = Path(__file__).resolve()

    cwd_trusted = is_path_trusted(Path.cwd().resolve())

    list_cmd = _build_list_cmd(
        script_path=self_script,
        show_all=bool(args.all),
        include_archived=bool(args.include_archived),
    )
    fzf_header = _build_fzf_header(
        include_archived=bool(args.include_archived),
        cwd_trusted=cwd_trusted,
    )
    fzf_prompt = _build_fzf_prompt(cwd_trusted=cwd_trusted)

    path_field = f"{{{FZF_FIELD_ROLLOUT_PATH}}}"

    preview_cmd_full = shell_join_fzf_command(
        [
            sys.executable,
            str(self_script),
            "preview",
            "--mode",
            "full",
            "--path",
            path_field,
        ]
    )
    preview_cmd_summary = shell_join_fzf_command(
        [
            sys.executable,
            str(self_script),
            "preview",
            "--mode",
            "summary",
            "--path",
            path_field,
        ]
    )

    session_id_field = f"{{{FZF_FIELD_SESSION_ID}}}"

    rename_cmd_parts: list[str] = [
        sys.executable,
        str(self_script),
        "rename",
        "--session-id",
        session_id_field,
    ]
    rename_cmd = shell_join_fzf_command(rename_cmd_parts)

    cwd_override_cmd_parts: list[str] = [
        sys.executable,
        str(self_script),
        "cwd-override",
        "--session-id",
        session_id_field,
    ]
    cwd_override_cmd = shell_join_fzf_command(cwd_override_cmd_parts)

    archive_fzf_cmd_parts: list[str] = [
        sys.executable,
        str(self_script),
        "fzf-archive",
        "--path",
        path_field,
        "--session-id",
        session_id_field,
        *(["--all"] if args.all else []),
        *(["--include-archived"] if args.include_archived else []),
    ]
    archive_fzf_cmd = shell_join_fzf_command(archive_fzf_cmd_parts)

    unarchive_fzf_cmd_parts: list[str] = [
        sys.executable,
        str(self_script),
        "fzf-unarchive",
        "--path",
        path_field,
        "--session-id",
        session_id_field,
        *(["--all"] if args.all else []),
        *(["--include-archived"] if args.include_archived else []),
    ]
    unarchive_fzf_cmd = shell_join_fzf_command(unarchive_fzf_cmd_parts)

    trust_fzf_cmd = shell_join_fzf_command(
        [
            sys.executable,
            str(self_script),
            "fzf-trust-pwd",
            *(["--include-archived"] if args.include_archived else []),
        ]
    )

    refresh_ui_cmd_parts: list[str] = [
        sys.executable,
        str(self_script),
        "fzf-refresh-ui",
        *(["--all"] if args.all else []),
        *(["--include-archived"] if args.include_archived else []),
    ]
    refresh_ui_cmd = shell_join_fzf_command(refresh_ui_cmd_parts)

    show_cwd = bool(args.all)
    # RATIONALE: fzf cannot search hidden fields when --with-nth is set, so we
    # append the full session id to the visible line (but hide it via ANSI
    # "conceal") so pasting a full UUID (or a rollout filename/path) filters
    # correctly.
    if show_cwd:
        with_nth = (
            f"{{2}}{FZF_DELIM}{{3}}{FZF_DELIM}{{5}}{FZF_DELIM}{{4}} \x1b[1m|\x1b[22m"
            "\x1b[8m {6} {7} {1}\x1b[28m"
        )
    else:
        with_nth = (
            f"{{2}}{FZF_DELIM}{{3}}{FZF_DELIM}{{4}} \x1b[1m|\x1b[22m"
            "\x1b[8m {6} {7} {1}\x1b[28m"
        )

    fzf_args = [
        "fzf",
        "--ansi",
        # Make the active row visually obvious even when a terminal theme's
        # selection background is subtle.
        "--pointer=>>",
        "--tabstop=4",
        "--info=inline-right",
        "--delimiter",
        FZF_DELIM,
        "--nth",
        "1..",
        "--with-nth",
        with_nth,
        "--header",
        fzf_header,
        "--prompt",
        fzf_prompt,
        "--preview",
        preview_cmd_summary if args.preview == "summary" else preview_cmd_full,
        "--preview-window",
        "right,50%,wrap,~7",
        "--height",
        "90%",
        "--bind",
        f"ctrl-v:change-preview({preview_cmd_full})",
        "--bind",
        f"ctrl-t:change-preview({preview_cmd_summary})",
        "--bind",
        f"ctrl-r:transform({refresh_ui_cmd})",
        "--bind",
        f"ctrl-e:execute({rename_cmd})+reload-sync({list_cmd})",
        "--bind",
        f"ctrl-o:execute({cwd_override_cmd})+reload-sync({list_cmd})",
        "--bind",
        f"ctrl-x:transform({archive_fzf_cmd})",
        "--bind",
        f"ctrl-y:transform({trust_fzf_cmd})",
        "--bind",
        f"focus:transform({refresh_ui_cmd})",
    ]
    if getattr(args, "no_mouse", False):
        # RATIONALE: With mouse enabled, terminals enter mouse reporting mode,
        # which blocks normal selection/copy. Disabling mouse restores normal
        # click-and-drag selection at the cost of mouse-wheel scrolling.
        fzf_args.append("--no-mouse")
    if args.include_archived:
        fzf_args.extend(["--bind", f"ctrl-u:transform({unarchive_fzf_cmd})"])

    env = os.environ.copy()
    env["FZF_DEFAULT_COMMAND"] = list_cmd

    proc = subprocess.run(
        fzf_args,
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        # Returncode 1/130 are common for "no match"/ESC; treat as cancellation.
        if proc.returncode not in {1, 130}:
            print(f"fzf failed (exit code {proc.returncode}).", file=sys.stderr)
        return 0

    selected = (proc.stdout or "").strip()
    if not selected:
        return 0

    fields = selected.split(FZF_DELIM)
    if len(fields) < FZF_FIELD_SESSION_ID:
        print("Unexpected fzf output; refusing to resume.", file=sys.stderr)
        return 2

    rollout_path_raw = fields[FZF_FIELD_ROLLOUT_PATH - 1].strip()
    rollout_path: Optional[Path] = None
    if rollout_path_raw:
        rollout_path = Path(rollout_path_raw).expanduser()
        if _is_archived_rollout(rollout_path):
            print(
                "Selected session is archived; unarchive it first (Ctrl-U).",
                file=sys.stderr,
            )
            return 0

    session_id = fields[FZF_FIELD_SESSION_ID - 1].strip()
    if not session_id:
        print("Missing session id in selection; refusing to resume.", file=sys.stderr)
        return 2

    if rollout_path:
        # RATIONALE: Resuming the same session concurrently in multiple Codex
        # instances can be confusing (and sometimes looks like missing history).
        # We block when the rollout file is currently open by any process.
        lsof_available, pids = lsof_pids(rollout_path)
        if lsof_available and pids:
            print(
                "Selected session appears to be open in another process "
                f"(pids: {', '.join(pids)}). Close the other session and try again.",
                file=sys.stderr,
            )
            return 0
        if not lsof_available:
            # Resume is read-only, so we can still proceed, but callers should
            # know the safety check is disabled.
            print(
                "Warning: `lsof` not found; cannot detect if a session is already "
                "open elsewhere.",
                file=sys.stderr,
            )

    cwd_override = load_cwd_overrides().get(session_id)
    argv_resume = [
        "codex",
        "resume",
        session_id,
    ]
    if cwd_override:
        cwd_path = Path(cwd_override).expanduser()
        if not cwd_path.is_dir():
            print(
                f"Resume root override no longer exists: {cwd_path} "
                "(clear it with Ctrl-O).",
                file=sys.stderr,
            )
            return 0
        cwd_path = cwd_path.resolve()
        argv_resume = [
            "codex",
            "resume",
            "--add-dir",
            str(cwd_path),
            "-C",
            str(cwd_path),
            session_id,
        ]

    try:
        _prepare_terminal_for_exec()
        os.execvp("codex", argv_resume)
    except FileNotFoundError:
        print("Unable to find `codex` on PATH.", file=sys.stderr)
        return 127
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)

    if argv_list and argv_list[0] in _INTERNAL_COMMANDS:
        args = _parse_internal_args(argv_list)
    else:
        args = _parse_public_args(argv_list)
    try:
        require_macos()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    cmd = getattr(args, "cmd", None)
    if cmd == "preview":
        try:
            return _preview(
                Path(args.path),
                mode=str(args.mode),
            )
        except Exception as exc:
            # RATIONALE: Preview runs inside fzf; avoid tracebacks in the pane.
            print(str(exc) or exc.__class__.__name__)
            return 2
    if cmd == "list":
        try:
            return _list_sessions(
                include_archived=bool(args.include_archived),
                include_all_cwds=bool(args.all),
                show_cwd=bool(args.show_cwd),
            )
        except Exception as exc:
            print(str(exc) or exc.__class__.__name__, file=sys.stderr)
            return 2
    if cmd == "rename":
        if args.unset and args.title:
            print("--title and --unset are mutually exclusive.", file=sys.stderr)
            return 2
        title = args.title
        if title is not None:
            title = title.strip()
        return _rename_one(
            session_id=str(args.session_id),
            title=title,
            unset=bool(args.unset),
        )
    if cmd == "cwd-override":
        if args.unset and args.cwd:
            print("--cwd and --unset are mutually exclusive.", file=sys.stderr)
            return 2
        cwd = args.cwd
        if cwd is not None:
            cwd = cwd.strip()
        return _cwd_override_one(
            session_id=str(args.session_id),
            cwd=cwd,
            unset=bool(args.unset),
        )
    if cmd == "fzf-archive":
        return _fzf_archive(
            session_id=str(args.session_id),
            rollout_path=str(args.path),
            show_all=bool(args.all),
            include_archived=bool(args.include_archived),
        )
    if cmd == "fzf-unarchive":
        return _fzf_unarchive(
            session_id=str(args.session_id),
            rollout_path=str(args.path),
            show_all=bool(args.all),
            include_archived=bool(args.include_archived),
        )
    if cmd == "fzf-trust-pwd":
        try:
            cwd = Path.cwd().resolve()
        except Exception:
            cwd = Path.cwd()

        include_archived = bool(args.include_archived)
        already_trusted = is_path_trusted(cwd)
        if already_trusted:
            header = _build_fzf_header(
                include_archived=include_archived,
                cwd_trusted=True,
            )
            prompt = _build_fzf_prompt(cwd_trusted=True)
            _clear_trust_arm()
            print(f"change-header({header})+change-prompt({prompt})")
            return 0

        armed_cwd = _read_trust_arm()
        if armed_cwd != str(cwd):
            _write_trust_arm(str(cwd))
            display = _abbrev_home(cwd)
            print(f"change-header(Confirm trust {display}: press ^Y again)")
            return 0

        _clear_trust_arm()
        try:
            ensure_path_trusted(cwd)
        except Exception as exc:
            message = " ".join(str(exc).split())
            if len(message) > 140:
                message = message[:139] + "…"
            print(f"change-header(Trust failed: {message})")
            return 0

        header = _build_fzf_header(
            include_archived=include_archived,
            cwd_trusted=True,
        )
        prompt = _build_fzf_prompt(cwd_trusted=True)
        print(f"change-header({header})+change-prompt({prompt})")
        return 0
    if cmd == "fzf-refresh-ui":
        _clear_archive_arm()
        _clear_unarchive_arm()
        _clear_trust_arm()

        include_archived = bool(args.include_archived)
        cwd_trusted = is_path_trusted(Path.cwd().resolve())
        header = _build_fzf_header(
            include_archived=include_archived,
            cwd_trusted=cwd_trusted,
        )
        prompt = _build_fzf_prompt(cwd_trusted=cwd_trusted)

        list_cmd = _build_list_cmd(
            script_path=Path(__file__).resolve(),
            show_all=bool(args.all),
            include_archived=include_archived,
        )
        print(
            f"reload-sync({list_cmd})+change-header({header})+change-prompt({prompt})"
        )
        return 0

    try:
        return _run_picker(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
