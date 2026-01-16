from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from scripts import codex_resume_plus as common


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


def test_sanitize_message_for_preview_strips_ide_documents_and_images():
    msg = "\n".join(
        [
            common.IDE_CONTEXT_HEADER,
            "## Documents:",
            "- doc1.md",
            "- doc2.md",
            "## Images:",
            "- img1.png",
            "## Notes:",
            "Keep this line.",
            "## My request for Codex:",
            "The heading above should be removed, but this line stays.",
        ]
    )
    out = common.sanitize_message_for_preview(msg)
    assert "Documents:" not in out
    assert "doc1.md" not in out
    assert "Images:" not in out
    assert "img1.png" not in out
    assert "## Notes:" in out
    assert "Keep this line." in out
    assert "## My request for Codex:" not in out
    assert "The heading above should be removed" in out


def test_read_first_user_message_text_skips_boilerplate(tmp_path: Path):
    session_id = "019b0000-0000-7000-8000-000000000001"
    rollout = (
        tmp_path
        / "rollout-2026-01-01T00-00-00-019b0000-0000-7000-8000-000000000001.jsonl"
    )
    _write_jsonl(
        rollout,
        [
            _session_meta(session_id=session_id, cwd="/tmp"),
            _response_item(role="user", text="# AGENTS.md instructions for /..."),
            _response_item(
                role="user",
                text="<environment_context>\n  <cwd>/</cwd>\n</environment_context>",
            ),
            _response_item(role="user", text="Hello from the real first user message."),
        ],
    )
    assert (
        common.read_first_user_message_text(rollout)
        == "Hello from the real first user message."
    )


def test_read_first_user_message_text_none_when_only_boilerplate(tmp_path: Path):
    session_id = "019b0000-0000-7000-8000-000000000002"
    rollout = (
        tmp_path
        / "rollout-2026-01-01T00-00-00-019b0000-0000-7000-8000-000000000002.jsonl"
    )
    _write_jsonl(
        rollout,
        [
            _session_meta(session_id=session_id, cwd="/tmp"),
            _response_item(role="user", text="# AGENTS.md instructions for /..."),
            _response_item(
                role="user", text="<environment_context>...</environment_context>"
            ),
        ],
    )
    assert common.read_first_user_message_text(rollout) is None


def test_read_last_message_scans_from_end(tmp_path: Path):
    rollout = (
        tmp_path
        / "rollout-2026-01-01T00-00-00-019b0000-0000-7000-8000-000000000003.jsonl"
    )
    _write_jsonl(
        rollout,
        [
            _session_meta(
                session_id="019b0000-0000-7000-8000-000000000003", cwd="/tmp"
            ),
            _response_item(role="user", text="First"),
            {"this": "is not jsonl message"},
            _response_item(role="assistant", text="Last assistant message."),
        ],
    )
    msg_type, msg = common.read_last_message(rollout)
    assert msg_type == "agent_message"
    assert msg == "Last assistant message."


def test_titles_sidecar_is_resilient_to_corrupt_json(codex_home: Path):
    titles_path = common.get_titles_path()
    titles_path.parent.mkdir(parents=True, exist_ok=True)
    titles_path.write_text("{not valid json", encoding="utf-8")
    assert common.load_titles() == {}


def test_set_and_unset_title_writes_atomic_file(codex_home: Path):
    session_id = "019b0000-0000-7000-8000-000000000004"
    common.set_title(session_id, "My Title")
    assert common.load_titles()[session_id] == "My Title"

    titles_path = common.get_titles_path()
    mode = titles_path.stat().st_mode & 0o777
    assert mode == 0o600

    assert common.unset_title(session_id) is True
    assert session_id not in common.load_titles()


def test_cwd_overrides_sidecar_is_resilient_to_corrupt_json(codex_home: Path):
    overrides_path = common.get_cwd_overrides_path()
    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    overrides_path.write_text("{not valid json", encoding="utf-8")
    assert common.load_cwd_overrides() == {}


def test_set_and_unset_cwd_override_writes_atomic_file(
    codex_home: Path, tmp_path: Path
):
    session_id = "019b0000-0000-7000-8000-000000000005"
    common.set_cwd_override(session_id, str(tmp_path))
    assert common.load_cwd_overrides()[session_id] == str(tmp_path.resolve())

    overrides_path = common.get_cwd_overrides_path()
    mode = overrides_path.stat().st_mode & 0o777
    assert mode == 0o600

    assert common.unset_cwd_override(session_id) is True
    assert session_id not in common.load_cwd_overrides()


def test_assert_rollout_not_open_blocks_without_lsof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(common, "lsof_pids", lambda _path: (False, []))
    with pytest.raises(RuntimeError, match="lsof"):
        common.assert_rollout_not_open(rollout, action="archive")


def test_build_session_summaries_filters_to_scope_by_default(
    codex_home: Path, tmp_path: Path
):
    scope = tmp_path / "project"
    scope.mkdir()
    inside = scope / "subdir"
    inside.mkdir()
    outside = tmp_path / "other"
    outside.mkdir()

    sessions_dir = common.get_codex_sessions_dir()
    p1 = (
        sessions_dir
        / "2026"
        / "01"
        / "01"
        / "rollout-2026-01-01T00-00-00-019b0000-0000-7000-8000-000000000101.jsonl"
    )
    _write_jsonl(
        p1,
        [
            _session_meta(
                session_id="019b0000-0000-7000-8000-000000000101", cwd=str(scope)
            ),
            _response_item(role="user", text="Inside scope"),
        ],
    )
    os.utime(p1, (1000, 1000))

    p2 = (
        sessions_dir
        / "2026"
        / "01"
        / "02"
        / "rollout-2026-01-02T00-00-00-019b0000-0000-7000-8000-000000000102.jsonl"
    )
    _write_jsonl(
        p2,
        [
            _session_meta(
                session_id="019b0000-0000-7000-8000-000000000102", cwd=str(inside)
            ),
            _response_item(role="user", text="Inside subdir"),
        ],
    )
    os.utime(p2, (2000, 2000))

    p3 = (
        sessions_dir
        / "2026"
        / "01"
        / "03"
        / "rollout-2026-01-03T00-00-00-019b0000-0000-7000-8000-000000000103.jsonl"
    )
    _write_jsonl(
        p3,
        [
            _session_meta(
                session_id="019b0000-0000-7000-8000-000000000103", cwd=str(outside)
            ),
            _response_item(role="user", text="Outside scope"),
        ],
    )
    os.utime(p3, (3000, 3000))

    p4 = (
        sessions_dir
        / "2026"
        / "01"
        / "04"
        / "rollout-2026-01-04T00-00-00-019b0000-0000-7000-8000-000000000104.jsonl"
    )
    _write_jsonl(
        p4,
        [
            _session_meta(session_id="019b0000-0000-7000-8000-000000000104", cwd=None),
            _response_item(role="user", text="No cwd"),
        ],
    )
    os.utime(p4, (4000, 4000))

    # Soft-move: map an outside-scope session into this scope.
    common.set_cwd_override("019b0000-0000-7000-8000-000000000103", str(scope))

    summaries = common.build_session_summaries(
        include_archived=False,
        include_all_cwds=False,
        current_scope=scope,
    )

    # Only sessions whose *effective* cwd is inside the scope are shown by default.
    assert [s.session_id for s in summaries] == [
        "019b0000-0000-7000-8000-000000000103",
        "019b0000-0000-7000-8000-000000000102",
        "019b0000-0000-7000-8000-000000000101",
    ]

    all_summaries = common.build_session_summaries(
        include_archived=False,
        include_all_cwds=True,
        current_scope=scope,
    )
    assert {s.session_id for s in all_summaries} == {
        "019b0000-0000-7000-8000-000000000101",
        "019b0000-0000-7000-8000-000000000102",
        "019b0000-0000-7000-8000-000000000103",
        "019b0000-0000-7000-8000-000000000104",
    }


def test_shell_join_fzf_command_quotes_fixed_parts_but_not_placeholders():
    cmd = common.shell_join_fzf_command(
        [
            "/path with spaces/python",
            "/path/with spaces/common script.py",
            "preview",
            "--path",
            "{1}",
            "{q}",
        ]
    )
    assert "'/path with spaces/python'" in cmd
    assert "'/path/with spaces/common script.py'" in cmd
    assert "{1}" in cmd
    assert "{q}" in cmd
