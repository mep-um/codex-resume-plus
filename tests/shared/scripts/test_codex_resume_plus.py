from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import codex_resume_plus as resume_plus


def _picker_args(*, all: bool, include_archived: bool, preview: str = "summary"):
    return argparse.Namespace(
        all=all, include_archived=include_archived, preview=preview
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj) + "\n")


def _session_meta(*, session_id: str, cwd: str | None) -> dict:
    payload: dict = {"id": session_id}
    if cwd is not None:
        payload["cwd"] = cwd
    return {"type": "session_meta", "payload": payload}


def _response_item(*, role: str, text: str) -> dict:
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": role,
            "content": [{"type": "input_text", "text": text}],
        },
    }


@pytest.fixture
def codex_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    return tmp_path


def test_default_unarchive_dest_uses_rollout_date(codex_home: Path):
    name = "rollout-2025-12-26T12-13-29-019b5ba6-b140-70a3-9cdd-e25ecaf1cdbb.jsonl"
    src = resume_plus.get_codex_archived_sessions_dir() / name
    dest = resume_plus._default_unarchive_dest(src)
    assert dest == resume_plus.get_codex_sessions_dir() / "2025" / "12" / "26" / name


def test_default_unarchive_dest_falls_back_to_unknown(codex_home: Path):
    src = resume_plus.get_codex_archived_sessions_dir() / "rollout-unknown.jsonl"
    dest = resume_plus._default_unarchive_dest(src)
    assert (
        dest
        == resume_plus.get_codex_sessions_dir() / "unknown" / "rollout-unknown.jsonl"
    )


def test_archive_rollout_moves_file(codex_home: Path, monkeypatch: pytest.MonkeyPatch):
    sessions_dir = resume_plus.get_codex_sessions_dir()
    src_dir = sessions_dir / "2026" / "01" / "01"
    src_dir.mkdir(parents=True, exist_ok=True)
    src = (
        src_dir
        / "rollout-2026-01-01T00-00-00-019b0000-0000-7000-8000-000000000201.jsonl"
    )
    src.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(resume_plus, "assert_rollout_safe_to_archive", lambda _p: None)

    err = resume_plus._archive_rollout_no_prompt(src)
    assert err is None

    dest = resume_plus.get_codex_archived_sessions_dir() / src.name
    assert dest.exists()
    assert not src.exists()


def test_archive_rollout_noops_when_already_archived(codex_home: Path):
    archived_dir = resume_plus.get_codex_archived_sessions_dir()
    archived_dir.mkdir(parents=True, exist_ok=True)
    src = (
        archived_dir
        / "rollout-2026-01-01T00-00-00-019b0000-0000-7000-8000-000000000202.jsonl"
    )
    src.write_text("{}", encoding="utf-8")
    assert resume_plus._archive_rollout_no_prompt(src) == "Already archived."


def test_unarchive_rollout_moves_file(
    codex_home: Path, monkeypatch: pytest.MonkeyPatch
):
    archived_dir = resume_plus.get_codex_archived_sessions_dir()
    archived_dir.mkdir(parents=True, exist_ok=True)
    src = (
        archived_dir
        / "rollout-2025-12-26T18-03-32-019b5ce7-2be4-76c3-82c8-15415c1d5f73.jsonl"
    )
    src.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(resume_plus, "assert_rollout_not_open", lambda _p, action: None)

    err = resume_plus._unarchive_rollout_no_prompt(src)
    assert err is None

    dest = resume_plus.get_codex_sessions_dir() / "2025" / "12" / "26" / src.name
    assert dest.exists()
    assert not src.exists()


def test_unarchive_rollout_blocks_when_not_archived(codex_home: Path):
    sessions_dir = resume_plus.get_codex_sessions_dir()
    sessions_dir.mkdir(parents=True, exist_ok=True)
    src = (
        sessions_dir
        / "rollout-2026-01-01T00-00-00-019b0000-0000-7000-8000-000000000203.jsonl"
    )
    src.write_text("{}", encoding="utf-8")
    assert resume_plus._unarchive_rollout_no_prompt(src) == "Not archived."


def test_rename_blank_keeps_existing_title(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    session_id = "019b0000-0000-7000-8000-000000000301"
    monkeypatch.setattr(resume_plus, "load_titles", lambda: {session_id: "Old Title"})

    set_calls: list[tuple[str, str]] = []
    unset_calls: list[str] = []
    monkeypatch.setattr(
        resume_plus, "set_title", lambda sid, title: set_calls.append((sid, title))
    )
    monkeypatch.setattr(
        resume_plus, "unset_title", lambda sid: unset_calls.append(sid) or True
    )
    monkeypatch.setattr(resume_plus, "_read_from_tty", lambda _prompt: "")

    rc = resume_plus._rename_one(session_id=session_id, title=None, unset=False)
    assert rc == 0
    assert set_calls == []
    assert unset_calls == []
    assert "No change." in capsys.readouterr().err


def test_rename_clear_requires_confirmation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    session_id = "019b0000-0000-7000-8000-000000000302"
    monkeypatch.setattr(resume_plus, "load_titles", lambda: {session_id: "Old Title"})

    prompts = iter(["-", "n"])
    monkeypatch.setattr(resume_plus, "_read_from_tty", lambda _prompt: next(prompts))
    unset_calls: list[str] = []
    monkeypatch.setattr(
        resume_plus, "unset_title", lambda sid: unset_calls.append(sid) or True
    )

    rc = resume_plus._rename_one(session_id=session_id, title=None, unset=False)
    assert rc == 0
    assert unset_calls == []
    assert "Rename canceled." in capsys.readouterr().err


def test_rename_clear_yes_unsets_title(monkeypatch: pytest.MonkeyPatch):
    session_id = "019b0000-0000-7000-8000-000000000303"
    monkeypatch.setattr(resume_plus, "load_titles", lambda: {session_id: "Old Title"})
    prompts = iter(["-", "y"])
    monkeypatch.setattr(resume_plus, "_read_from_tty", lambda _prompt: next(prompts))

    unset_calls: list[str] = []
    monkeypatch.setattr(
        resume_plus, "unset_title", lambda sid: unset_calls.append(sid) or True
    )

    rc = resume_plus._rename_one(session_id=session_id, title=None, unset=False)
    assert rc == 0
    assert unset_calls == [session_id]


def test_rename_keyboard_interrupt_is_clean(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    session_id = "019b0000-0000-7000-8000-000000000304"
    monkeypatch.setattr(resume_plus, "load_titles", lambda: {})

    def _raise(_prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(resume_plus, "_read_from_tty", _raise)
    rc = resume_plus._rename_one(session_id=session_id, title=None, unset=False)
    assert rc == 0
    assert "Rename canceled." in capsys.readouterr().err


def test_run_picker_binds_unarchive_only_when_visible(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(resume_plus, "require_fzf", lambda: None)
    monkeypatch.setattr(resume_plus, "load_cwd_overrides", lambda: {})

    captured: dict[str, list[str]] = {}

    def _fake_run(argv, **kwargs):
        captured["argv"] = list(argv)
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(resume_plus.subprocess, "run", _fake_run)

    rc = resume_plus._run_picker(
        _picker_args(all=False, include_archived=False, preview="summary")
    )
    assert rc == 0
    argv = captured["argv"]
    assert not any("ctrl-u:" in str(item) for item in argv)

    rc = resume_plus._run_picker(
        _picker_args(all=False, include_archived=True, preview="summary")
    )
    assert rc == 0
    argv = captured["argv"]
    assert any("ctrl-u:" in str(item) for item in argv)
    _ = capsys.readouterr()


def test_run_picker_does_not_resume_archived_selection(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    monkeypatch.setattr(resume_plus, "require_fzf", lambda: None)
    monkeypatch.setattr(resume_plus, "load_cwd_overrides", lambda: {})

    archived = tmp_path / "archived" / "rollout.jsonl"
    archived.parent.mkdir(parents=True, exist_ok=True)
    archived.write_text("{}", encoding="utf-8")

    selected_line = resume_plus.FZF_DELIM.join(
        [
            str(archived),
            "2026-01-01 00:00",
            "deadbeef",
            "Title",
            "-",
            "019b0000-0000-7000-8000-000000000401",
            "first",
        ]
    )

    monkeypatch.setattr(
        resume_plus.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(
            returncode=0, stdout=selected_line, stderr=""
        ),
    )
    monkeypatch.setattr(resume_plus, "_is_archived_rollout", lambda _p: True)

    called = {"execvp": False}

    def _execvp(_file, _args):
        called["execvp"] = True

    monkeypatch.setattr(resume_plus.os, "execvp", _execvp)

    rc = resume_plus._run_picker(
        _picker_args(all=False, include_archived=True, preview="summary")
    )
    assert rc == 0
    assert called["execvp"] is False
    assert "archived" in capsys.readouterr().err.lower()


def test_run_picker_execs_codex_on_unarchived_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(resume_plus, "require_fzf", lambda: None)
    monkeypatch.setattr(resume_plus, "load_cwd_overrides", lambda: {})
    monkeypatch.setattr(resume_plus, "lsof_pids", lambda _p: (True, []))

    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text("{}", encoding="utf-8")
    selected_line = resume_plus.FZF_DELIM.join(
        [
            str(rollout),
            "2026-01-01 00:00",
            "deadbeef",
            "Title",
            "-",
            "019b0000-0000-7000-8000-000000000402",
            "first",
        ]
    )

    monkeypatch.setattr(
        resume_plus.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(
            returncode=0, stdout=selected_line, stderr=""
        ),
    )
    monkeypatch.setattr(resume_plus, "_is_archived_rollout", lambda _p: False)

    called: dict[str, object] = {}

    def _execvp(file: str, argv: list[str]) -> None:
        called["file"] = file
        called["argv"] = argv
        raise SystemExit(0)

    monkeypatch.setattr(resume_plus.os, "execvp", _execvp)

    with pytest.raises(SystemExit):
        resume_plus._run_picker(
            _picker_args(all=False, include_archived=False, preview="summary")
        )

    assert called["file"] == "codex"
    assert called["argv"] == [
        "codex",
        "resume",
        "019b0000-0000-7000-8000-000000000402",
    ]


def test_run_picker_execs_codex_with_cd_override_when_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(resume_plus, "require_fzf", lambda: None)
    monkeypatch.setattr(resume_plus, "lsof_pids", lambda _p: (True, []))

    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text("{}", encoding="utf-8")
    session_id = "019b0000-0000-7000-8000-000000000403"
    selected_line = resume_plus.FZF_DELIM.join(
        [
            str(rollout),
            "2026-01-01 00:00",
            "deadbeef",
            "Title",
            "-",
            session_id,
            "first",
        ]
    )

    monkeypatch.setattr(
        resume_plus.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(
            returncode=0, stdout=selected_line, stderr=""
        ),
    )
    monkeypatch.setattr(resume_plus, "_is_archived_rollout", lambda _p: False)
    monkeypatch.setattr(
        resume_plus, "load_cwd_overrides", lambda: {session_id: str(tmp_path)}
    )

    called: dict[str, object] = {}

    def _execvp(file: str, argv: list[str]) -> None:
        called["file"] = file
        called["argv"] = argv
        raise SystemExit(0)

    monkeypatch.setattr(resume_plus.os, "execvp", _execvp)

    with pytest.raises(SystemExit):
        resume_plus._run_picker(
            _picker_args(all=False, include_archived=False, preview="summary")
        )

    assert called["file"] == "codex"
    assert called["argv"] == [
        "codex",
        "resume",
        "--add-dir",
        str(tmp_path),
        "-C",
        str(tmp_path),
        session_id,
    ]


def test_run_picker_blocks_when_rollout_is_open(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    monkeypatch.setattr(resume_plus, "require_fzf", lambda: None)
    monkeypatch.setattr(resume_plus, "load_cwd_overrides", lambda: {})
    monkeypatch.setattr(resume_plus, "lsof_pids", lambda _p: (True, ["1234"]))

    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text("{}", encoding="utf-8")
    selected_line = resume_plus.FZF_DELIM.join(
        [
            str(rollout),
            "2026-01-01 00:00",
            "deadbeef",
            "Title",
            "-",
            "019b0000-0000-7000-8000-000000000499",
            "first",
        ]
    )

    monkeypatch.setattr(
        resume_plus.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(
            returncode=0, stdout=selected_line, stderr=""
        ),
    )
    monkeypatch.setattr(resume_plus, "_is_archived_rollout", lambda _p: False)

    called = {"execvp": False}

    def _execvp(_file, _args):
        called["execvp"] = True

    monkeypatch.setattr(resume_plus.os, "execvp", _execvp)

    rc = resume_plus._run_picker(
        _picker_args(all=False, include_archived=False, preview="summary")
    )
    assert rc == 0
    assert called["execvp"] is False
    assert "open" in capsys.readouterr().err.lower()


def test_is_path_trusted_is_true_for_subdir_of_trusted_root(
    codex_home: Path, tmp_path: Path
):
    root = tmp_path / "proj"
    child = root / "subdir"
    child.mkdir(parents=True, exist_ok=True)

    config = resume_plus.get_codex_config_path()
    config.write_text(
        f'[projects."{root}"]\ntrust_level = "trusted"\n',
        encoding="utf-8",
    )

    assert resume_plus.is_path_trusted(child, config_path=config) is True


def test_ensure_path_trusted_appends_project_when_missing(
    codex_home: Path, tmp_path: Path
):
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)

    config = resume_plus.get_codex_config_path()
    config.write_text('[misc]\nfoo = "bar"\n', encoding="utf-8")

    resume_plus.ensure_path_trusted(root, config_path=config)
    text = config.read_text(encoding="utf-8")
    assert f'[projects."{root}"]' in text
    assert 'trust_level = "trusted"' in text


def test_ensure_path_trusted_updates_existing_project_entry(
    codex_home: Path, tmp_path: Path
):
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)

    config = resume_plus.get_codex_config_path()
    config.write_text(
        f'[projects."{root}"]\ntrust_level = "untrusted"\n',
        encoding="utf-8",
    )

    resume_plus.ensure_path_trusted(root, config_path=config)
    text = config.read_text(encoding="utf-8")
    assert f'[projects."{root}"]' in text
    assert 'trust_level = "trusted"' in text


def test_ensure_path_trusted_inserts_trust_level_when_missing(
    codex_home: Path, tmp_path: Path
):
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)

    config = resume_plus.get_codex_config_path()
    config.write_text(
        f'[projects."{root}"]\nfoo = "bar"\n',
        encoding="utf-8",
    )

    resume_plus.ensure_path_trusted(root, config_path=config)
    text = config.read_text(encoding="utf-8")

    header = f'[projects."{root}"]'
    assert (
        text.index(header)
        < text.index('trust_level = "trusted"')
        < text.index('foo = "bar"')
    )


def test_ensure_path_trusted_preserves_comment_and_file_mode(
    codex_home: Path, tmp_path: Path
):
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)

    config = resume_plus.get_codex_config_path()
    config.write_text(
        f'[projects."{root}"]\ntrust_level = "untrusted"  # keep\n',
        encoding="utf-8",
    )
    os.chmod(config, 0o644)

    resume_plus.ensure_path_trusted(root, config_path=config)
    text = config.read_text(encoding="utf-8")
    assert 'trust_level = "trusted"  # keep' in text
    assert (config.stat().st_mode & 0o777) == 0o644


def test_ensure_path_trusted_creates_config_with_0600(codex_home: Path, tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)

    config = resume_plus.get_codex_config_path()
    assert not config.exists()

    resume_plus.ensure_path_trusted(root, config_path=config)
    assert (config.stat().st_mode & 0o777) == 0o600


def test_ensure_path_trusted_does_not_create_config_lock_file(
    codex_home: Path, tmp_path: Path
):
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)

    config = resume_plus.get_codex_config_path()
    resume_plus.ensure_path_trusted(root, config_path=config)

    assert not (codex_home / "config.toml.lock").exists()


def test_ensure_path_trusted_escapes_quotes_and_backslashes(
    codex_home: Path, tmp_path: Path
):
    root = tmp_path / 'proj "weird" \\\\ name'
    root.mkdir(parents=True, exist_ok=True)

    config = resume_plus.get_codex_config_path()
    resume_plus.ensure_path_trusted(root, config_path=config)

    escaped = resume_plus._toml_escape_basic_string(str(root.resolve()))
    text = config.read_text(encoding="utf-8")
    assert f'[projects."{escaped}"]' in text


def test_build_fzf_header_includes_trust_hint_only_when_untrusted():
    assert "^Y trust pwd×2" in resume_plus._build_fzf_header(
        include_archived=False, cwd_trusted=False
    )
    assert "^Y trust pwd×2" not in resume_plus._build_fzf_header(
        include_archived=False, cwd_trusted=True
    )


def test_preview_summary_shows_first_user_without_repeating_single_user_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    session_id = "019b0000-0000-7000-8000-000000000901"
    rollout = tmp_path / "rollout.jsonl"
    _write_jsonl(
        rollout,
        [
            _session_meta(session_id=session_id, cwd=str(tmp_path)),
            _response_item(role="user", text="# AGENTS.md instructions for /..."),
            _response_item(
                role="user",
                text="\n".join(
                    [
                        resume_plus.IDE_CONTEXT_HEADER,
                        "## Documents:",
                        "- doc1.md",
                        "## Notes:",
                        "Keep this line.",
                        "",
                        "Main question: hello",
                    ]
                ),
            ),
            _response_item(role="assistant", text="Assistant says hi."),
        ],
    )

    assert resume_plus._preview(rollout, mode="summary") == 0
    out = capsys.readouterr().out

    assert "First user:" in out
    assert "Keep this line." in out
    assert "Main question: hello" in out
    assert "doc1.md" not in out

    # Only one meaningful user message; do not repeat it as "Last user".
    assert "Last user:" not in out
    assert "Last assistant:" in out
    assert "Assistant says hi." in out


def test_preview_full_skips_boilerplate_user_messages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    session_id = "019b0000-0000-7000-8000-000000000902"
    rollout = tmp_path / "rollout.jsonl"
    _write_jsonl(
        rollout,
        [
            _session_meta(session_id=session_id, cwd=str(tmp_path)),
            _response_item(role="user", text="<environment_context>"),
            _response_item(role="assistant", text="Assistant message 1."),
            _response_item(role="user", text="Real user message."),
        ],
    )

    assert resume_plus._preview(rollout, mode="full") == 0
    out = capsys.readouterr().out
    assert "AGENTS.md instructions" not in out
    assert "<environment_context>" not in out
    assert "Assistant message 1." in out
    assert "Real user message." in out


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_arm_state(path: Path, *, session_id: str, armed_at: float) -> None:
    path.write_text(
        json.dumps({"session_id": session_id, "armed_at": armed_at}),
        encoding="utf-8",
    )


def test_fzf_archive_first_press_arms_without_archiving(
    codex_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    now = 1000.0
    monkeypatch.setattr(resume_plus.time, "time", lambda: now)

    called: dict[str, object] = {"archive": False}
    monkeypatch.setattr(
        resume_plus,
        "_archive_rollout_no_prompt",
        lambda _p: called.__setitem__("archive", True),
    )

    session_id = "019b0000-0000-7000-8000-00000000aaaa"
    rollout = tmp_path / "rollout.jsonl"
    rc = resume_plus._fzf_archive(
        session_id=session_id,
        rollout_path=str(rollout),
        show_all=False,
        include_archived=False,
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "Confirm archive" in out
    assert session_id[-8:] in out
    assert called["archive"] is False

    state_path = resume_plus._archive_arm_state_path()
    assert state_path.exists()
    data = _read_json(state_path)
    assert data["session_id"] == session_id
    assert data["armed_at"] == now


def test_fzf_archive_second_press_within_ttl_archives_and_refreshes(
    codex_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    now = 1000.0
    monkeypatch.setattr(resume_plus.time, "time", lambda: now)

    session_id = "019b0000-0000-7000-8000-00000000aaaa"
    _write_arm_state(
        resume_plus._archive_arm_state_path(),
        session_id=session_id,
        armed_at=now - 1.0,
    )

    called: dict[str, object] = {}

    def _archive(p: Path):
        called["path"] = p
        return None

    monkeypatch.setattr(resume_plus, "_archive_rollout_no_prompt", _archive)
    monkeypatch.setattr(resume_plus, "is_path_trusted", lambda _p: True)

    rollout = tmp_path / "rollout.jsonl"
    rc = resume_plus._fzf_archive(
        session_id=session_id,
        rollout_path=str(rollout),
        show_all=False,
        include_archived=False,
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "reload-sync(" in out
    assert "change-header(" in out
    assert "Confirm archive" not in out
    assert called["path"] == rollout
    assert not resume_plus._archive_arm_state_path().exists()


def test_fzf_archive_ttl_expired_rearms_instead_of_archiving(
    codex_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    now = 1000.0
    monkeypatch.setattr(resume_plus.time, "time", lambda: now)

    session_id = "019b0000-0000-7000-8000-00000000aaaa"
    _write_arm_state(
        resume_plus._archive_arm_state_path(),
        session_id=session_id,
        armed_at=now - resume_plus._ARCHIVE_ARM_TTL_S - 1.0,
    )

    called: dict[str, object] = {"archive": False}
    monkeypatch.setattr(
        resume_plus,
        "_archive_rollout_no_prompt",
        lambda _p: called.__setitem__("archive", True),
    )

    rollout = tmp_path / "rollout.jsonl"
    rc = resume_plus._fzf_archive(
        session_id=session_id,
        rollout_path=str(rollout),
        show_all=False,
        include_archived=False,
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "Confirm archive" in out
    assert called["archive"] is False

    data = _read_json(resume_plus._archive_arm_state_path())
    assert data["session_id"] == session_id
    assert data["armed_at"] == now


def test_fzf_archive_armed_for_different_session_requires_new_confirm(
    codex_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    now = 1000.0
    monkeypatch.setattr(resume_plus.time, "time", lambda: now)

    _write_arm_state(
        resume_plus._archive_arm_state_path(),
        session_id="019b0000-0000-7000-8000-00000000aaaa",
        armed_at=now - 1.0,
    )

    session_id_b = "019b0000-0000-7000-8000-00000000bbbb"
    rollout = tmp_path / "rollout.jsonl"
    rc = resume_plus._fzf_archive(
        session_id=session_id_b,
        rollout_path=str(rollout),
        show_all=False,
        include_archived=False,
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "Confirm archive" in out
    assert session_id_b[-8:] in out

    data = _read_json(resume_plus._archive_arm_state_path())
    assert data["session_id"] == session_id_b


def test_fzf_archive_blocked_message_is_condensed_and_truncated(
    codex_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    now = 1000.0
    monkeypatch.setattr(resume_plus.time, "time", lambda: now)

    session_id = "019b0000-0000-7000-8000-00000000aaaa"
    _write_arm_state(
        resume_plus._archive_arm_state_path(),
        session_id=session_id,
        armed_at=now - 1.0,
    )

    long_error = "Line1\n" + ("x" * 500) + "\nLine3"
    monkeypatch.setattr(
        resume_plus, "_archive_rollout_no_prompt", lambda _p: long_error
    )

    rollout = tmp_path / "rollout.jsonl"
    rc = resume_plus._fzf_archive(
        session_id=session_id,
        rollout_path=str(rollout),
        show_all=False,
        include_archived=False,
    )
    assert rc == 0

    out = capsys.readouterr().out.strip()
    assert "Archive blocked:" in out
    assert "\n" not in out
    assert out.endswith("…)")
    assert "reload-sync(" not in out
    assert not resume_plus._archive_arm_state_path().exists()


def test_fzf_archive_corrupted_arm_file_is_treated_as_unarmed(
    codex_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    now = 1000.0
    monkeypatch.setattr(resume_plus.time, "time", lambda: now)

    state_path = resume_plus._archive_arm_state_path()
    state_path.write_text("not json", encoding="utf-8")

    session_id = "019b0000-0000-7000-8000-00000000aaaa"
    rollout = tmp_path / "rollout.jsonl"
    rc = resume_plus._fzf_archive(
        session_id=session_id,
        rollout_path=str(rollout),
        show_all=False,
        include_archived=False,
    )
    assert rc == 0
    assert "Confirm archive" in capsys.readouterr().out

    data = _read_json(state_path)
    assert data["session_id"] == session_id
    assert data["armed_at"] == now


def test_fzf_unarchive_first_press_arms_without_unarchiving(
    codex_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    now = 1000.0
    monkeypatch.setattr(resume_plus.time, "time", lambda: now)

    called: dict[str, object] = {"unarchive": False}
    monkeypatch.setattr(
        resume_plus,
        "_unarchive_rollout_no_prompt",
        lambda _p: called.__setitem__("unarchive", True),
    )

    session_id = "019b0000-0000-7000-8000-00000000aaaa"
    rollout = tmp_path / "rollout.jsonl"
    rc = resume_plus._fzf_unarchive(
        session_id=session_id,
        rollout_path=str(rollout),
        show_all=False,
        include_archived=True,
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "Confirm unarchive" in out
    assert session_id[-8:] in out
    assert called["unarchive"] is False

    state_path = resume_plus._unarchive_arm_state_path()
    assert state_path.exists()
    data = _read_json(state_path)
    assert data["session_id"] == session_id
    assert data["armed_at"] == now


def test_fzf_unarchive_second_press_within_ttl_unarchives_and_refreshes(
    codex_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    now = 1000.0
    monkeypatch.setattr(resume_plus.time, "time", lambda: now)

    session_id = "019b0000-0000-7000-8000-00000000aaaa"
    _write_arm_state(
        resume_plus._unarchive_arm_state_path(),
        session_id=session_id,
        armed_at=now - 1.0,
    )

    called: dict[str, object] = {}

    def _unarchive(p: Path):
        called["path"] = p
        return None

    monkeypatch.setattr(resume_plus, "_unarchive_rollout_no_prompt", _unarchive)
    monkeypatch.setattr(resume_plus, "is_path_trusted", lambda _p: True)

    rollout = tmp_path / "rollout.jsonl"
    rc = resume_plus._fzf_unarchive(
        session_id=session_id,
        rollout_path=str(rollout),
        show_all=False,
        include_archived=True,
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "reload-sync(" in out
    assert "change-header(" in out
    assert "Confirm unarchive" not in out
    assert called["path"] == rollout
    assert not resume_plus._unarchive_arm_state_path().exists()


def test_fzf_trust_pwd_already_trusted_is_noop_and_clears_arm(
    codex_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(resume_plus, "require_macos", lambda: None)

    resume_plus._write_trust_arm(str(tmp_path))
    monkeypatch.setattr(resume_plus, "is_path_trusted", lambda _p: True)

    called: dict[str, object] = {"ensure": False}
    monkeypatch.setattr(
        resume_plus,
        "ensure_path_trusted",
        lambda _p: called.__setitem__("ensure", True),
    )

    rc = resume_plus.main(["fzf-trust-pwd"])
    assert rc == 0
    out = capsys.readouterr().out

    assert "change-header(" in out
    assert "change-prompt(" in out
    assert called["ensure"] is False
    assert not resume_plus._trust_arm_state_path().exists()


def test_fzf_trust_pwd_double_press_trusts(
    codex_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(resume_plus, "require_macos", lambda: None)
    monkeypatch.setattr(resume_plus, "is_path_trusted", lambda _p: False)

    ensured: list[Path] = []
    monkeypatch.setattr(resume_plus, "ensure_path_trusted", lambda p: ensured.append(p))

    rc = resume_plus.main(["fzf-trust-pwd"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Confirm trust" in out

    rc = resume_plus.main(["fzf-trust-pwd"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "change-header(" in out
    assert "change-prompt(" in out
    assert ensured == [tmp_path.resolve()]
    assert not resume_plus._trust_arm_state_path().exists()


def test_fzf_trust_pwd_second_press_failure_is_single_line_and_clears_arm(
    codex_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(resume_plus, "require_macos", lambda: None)
    monkeypatch.setattr(resume_plus, "is_path_trusted", lambda _p: False)

    resume_plus._write_trust_arm(str(tmp_path.resolve()))

    def _fail(_p: Path) -> None:
        raise RuntimeError("Line1\n" + ("x" * 500) + "\nLine3")

    monkeypatch.setattr(resume_plus, "ensure_path_trusted", _fail)

    rc = resume_plus.main(["fzf-trust-pwd"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert "Trust failed:" in out
    assert "\n" not in out
    assert out.endswith("…)")
    assert not resume_plus._trust_arm_state_path().exists()
