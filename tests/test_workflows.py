"""Workflow hygiene (tests/test_release.py checks each workflow's own purpose).

* Every action is pinned to a 40-hex commit SHA, with its tag in a comment on the same line, so a moved tag
  can't change what runs. One action is pinned to one commit across the workflows, and a commit always carries
  the same tag.
* No run script holds a ``${{ ... }}`` expression: a value reaches the shell only through an environment
  variable, so nothing a workflow is handed (an input, a ref, a job's result) is ever pasted into a script.
* Every workflow's token is read-only (``contents: read``) and no job asks for more; no workflow runs on
  ``pull_request_target``, reads a stored secret, or keeps the checkout's credentials.
* Every workflow runs one uv version, the one the plugin's skill installs: each ``astral-sh/setup-uv`` step
  names it exactly.
* ``.gitattributes`` turns line-ending conversion off for every file, so a checkout is byte for byte the repo.
* **The offline job proves a whole run, not just its edges.** test.yml's ``offline`` job runs the
  ``cli_subprocess`` tests under ``sandbox-exec`` with the network denied. One of them is here: a transcription,
  a relabel and notes, each through the real command line in a child process, on the fake engines. The job
  fails when its selection no longer holds this test, and this file checks that the job says so.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
WORKFLOWS = ROOT / ".github" / "workflows"
FILES = sorted(WORKFLOWS.glob("*.yml"))
SKILL = ROOT / "claude-plugin" / "skills" / "audiointake" / "SKILL.md"

USES_LINE = re.compile(r"^\s*(?:-\s+)?uses:\s*(?P<ref>\S+)(?P<rest>.*)$")
PINNED = re.compile(r"^(?P<action>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_./-]+)?)@(?P<sha>[0-9a-f]{40})$")
TAG_COMMENT = re.compile(r"^\s+#\s*(?P<tag>v\d+(?:\.\d+){0,2})\s*$")
UV_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
SKILL_UV = re.compile(r"github\.com/astral-sh/uv/releases/download/(\d+\.\d+\.\d+)/uv-aarch64-apple-darwin\.tar\.gz")


def _uses_lines(files: list[Path] = FILES) -> list[tuple[str, int, str, str]]:
    """(workflow, line number, ref, rest of the line) for every ``uses:`` line, read as text."""
    out = []
    for path in files:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            m = USES_LINE.match(line)
            if m:
                out.append((path.name, n, m["ref"].strip("'\""), m["rest"]))
    return out


def pin_problems(lines: list[tuple[str, int, str, str]]) -> list[str]:
    problems = []
    for name, n, ref, rest in lines:
        if ref.startswith("./"):                          # an action in this repo: the checkout pins it
            continue
        if not PINNED.match(ref):
            problems.append(f"{name}:{n}: {ref} isn't pinned to a 40-hex commit SHA")
        elif not TAG_COMMENT.match(rest):
            problems.append(f"{name}:{n}: {ref} needs its tag in a comment (# vX.Y.Z)")
    return problems


def test_there_are_the_four_workflows():
    assert {p.name for p in FILES} == {"test.yml", "resolve.yml", "install-smoke.yml", "mlx-probe.yml"}


def test_every_action_is_pinned_to_a_commit_with_its_tag_in_a_comment():
    lines = _uses_lines()
    assert lines
    assert pin_problems(lines) == []


def test_one_commit_per_action_and_one_tag_per_commit():
    actions: dict[str, set[str]] = {}
    tags: dict[str, set[str]] = {}
    for _name, _n, ref, rest in _uses_lines():
        m, c = PINNED.match(ref), TAG_COMMENT.match(rest)
        if m and c:
            actions.setdefault(m["action"], set()).add(m["sha"])
            tags.setdefault(m["sha"], set()).add(c["tag"])
    assert {a: s for a, s in actions.items() if len(s) > 1} == {}, "an action is pinned to two commits"
    assert {s: t for s, t in tags.items() if len(t) > 1} == {}, "a commit carries two different tags"


def test_the_pin_rule_catches_what_it_is_for(tmp_path: Path):
    wf = tmp_path / "x.yml"
    wf.write_text("steps:\n  - uses: actions/checkout@v7\n  - uses: actions/cache@" + "a" * 40 + "\n"
                  "  - uses: actions/cache@" + "a" * 40 + " # v4.3.0\n  - uses: ./local\n", encoding="utf-8")
    assert pin_problems(_uses_lines([wf])) == [
        "x.yml:2: actions/checkout@v7 isn't pinned to a 40-hex commit SHA",
        "x.yml:3: actions/cache@" + "a" * 40 + " needs its tag in a comment (# vX.Y.Z)"]


# --- read as YAML (PyYAML comes with huggingface-hub, a runtime dependency)

def _wf(path: Path) -> dict[str, Any]:
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["on"] = data.pop(True, data.get("on"))           # YAML 1.1 reads the key `on` as true
    return data


def _steps(wf: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for job in wf["jobs"].values() for step in job.get("steps", [])]


def test_the_text_scan_sees_every_uses_the_yaml_holds():
    for path in FILES:
        wf = _wf(path)
        parsed = [str(s["uses"]) for s in _steps(wf) if "uses" in s]
        parsed += [str(j["uses"]) for j in wf["jobs"].values() if "uses" in j]   # reusable workflows
        scanned = [ref for name, _n, ref, _rest in _uses_lines() if name == path.name]
        assert sorted(parsed) == sorted(scanned), path.name


def test_no_run_script_holds_an_expression():
    for path in FILES:
        for job_id, job in _wf(path)["jobs"].items():
            for step in job.get("steps", []):
                assert "${{" not in str(step.get("run", "")), (
                    f"{path.name} job {job_id}, step {step.get('name') or step.get('uses')}: pass the value "
                    "through env: instead of an expression inside the script")


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_every_token_is_read_only(path: Path):
    wf = _wf(path)
    assert wf["permissions"] == {"contents": "read"}, path.name
    for job_id, job in wf["jobs"].items():
        perms = job.get("permissions")
        if perms is None:
            continue
        assert isinstance(perms, dict), f"{job_id}: permissions: {perms!r} is wider than contents: read"
        wider = {k: v for k, v in perms.items() if v != "none" and not (k == "contents" and v == "read")}
        assert wider == {}, f"{job_id} asks for more than contents: read: {wider}"


def test_no_workflow_uses_pull_request_target_secrets_or_kept_credentials():
    for path in FILES:
        text = "\n".join(l for l in path.read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith("#"))
        assert "pull_request_target" not in text, path.name
        assert "secrets." not in text, f"{path.name} reads a stored secret"
        for step in _steps(_wf(path)):
            if str(step.get("uses", "")).startswith("actions/checkout"):
                assert step.get("with", {}).get("persist-credentials") is False, f"{path.name}: checkout keeps its token"


def _uv_versions() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in FILES:
        for job_id, job in _wf(path)["jobs"].items():
            for step in job.get("steps", []):
                if str(step.get("uses", "")).startswith("astral-sh/setup-uv@"):
                    where = f"{path.name} job {job_id}"
                    version = str((step.get("with") or {}).get("version", ""))
                    assert UV_VERSION.match(version), f"{where}: setup-uv needs an exact version: (got {version!r})"
                    found.setdefault(version, set()).add(where)
    return found


def test_every_workflow_runs_one_uv_version():
    """A setup-uv step without a version takes whatever uv is newest that day."""
    found = _uv_versions()
    assert found, "no setup-uv step"
    assert len(found) == 1, f"workflows run more than one uv version: {found}"


def test_the_skill_installs_the_uv_the_workflows_run():
    if not SKILL.is_file():
        pytest.skip("SKILL.md isn't written yet")
    pins = set(SKILL_UV.findall(SKILL.read_text(encoding="utf-8")))
    assert pins == set(_uv_versions()), f"the skill installs uv {sorted(pins)}, the workflows run {sorted(_uv_versions())}"


def test_every_file_checks_out_byte_for_byte():
    rules = [ln.split("#", 1)[0].split() for ln in (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()]
    assert ["*", "-text"] in rules


# --------------------------------------------------------------------------- the offline job's end-to-end run

OFFLINE_E2E = "tests/test_workflows.py::test_the_offline_job_transcribes_end_to_end"


def test_the_offline_job_refuses_a_selection_without_a_whole_run():
    text = (WORKFLOWS / "test.yml").read_text(encoding="utf-8")
    check = text.index("--collect-only -m \"cli_subprocess and not network\"")
    assert f"grep -q '{OFFLINE_E2E}'" in text[check:] and check < text.index("sandbox-exec -p")
    tree = __import__("ast").parse(Path(__file__).read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if getattr(n, "name", "") == OFFLINE_E2E.rsplit("::", 1)[1])
    marks = [__import__("ast").unparse(d) for d in fn.decorator_list]
    assert marks == ["pytest.mark.cli_subprocess"], marks


def _cli(tmp_path: Path, *args: str) -> tuple[int, dict, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AUDIOINTAKE_", "HF_"))}
    env.update(HOME=str(tmp_path / "home"), AUDIOINTAKE_HOME=str(tmp_path / "data"),
               AUDIOINTAKE_CACHE=str(tmp_path / "cache"), HF_HUB_CACHE=str(tmp_path / "hf"),
               PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-I", "-B", str(TESTS / "fakes" / "run_cli.py"), str(tmp_path / "cfg.json"),
                           *args, "--json"], env=env, capture_output=True, text=True, timeout=120,
                          stdin=subprocess.DEVNULL)
    out = json.loads(proc.stdout) if proc.stdout.strip() else {}
    return proc.returncode, out, proc.stderr


@pytest.mark.cli_subprocess
def test_the_offline_job_transcribes_end_to_end(tmp_path: Path):
    if not (TESTS / "fakes" / "run_cli.py").is_file():
        pytest.skip("the fakes' command-line runner isn't written yet")
    from fakes import audio
    for sub in ("home", "data", "cache", "hf"):
        (tmp_path / sub).mkdir()
    (tmp_path / "cfg.json").write_text(json.dumps({"root": str(tmp_path / "fake"), "setup": {}}), encoding="utf-8")
    src = tmp_path / "in"
    src.mkdir()
    rec = audio(src / "Team sync.m4a")

    code, out, err = _cli(tmp_path, "transcribe", str(rec))
    assert code == 0 and out["exit"] == 0, err
    (record,) = out["records"]
    assert record["exit"] == 0 and record["error"] is None, record
    md, js = src / "Team sync (transcript).md", src / "Team sync (transcript).json"
    assert md.is_file() and js.is_file()
    assert json.loads(js.read_text(encoding="utf-8"))["format"] == "audiointake-transcript"
    labels = [s["label"] for s in record["speakers"]]
    assert labels, record

    code, out, err = _cli(tmp_path, "relabel", str(rec), "--label", labels[0], "--name", "Jane Doe")
    assert code == 0 and out["record"]["exit"] == 0, err
    assert "Jane Doe" in md.read_text(encoding="utf-8")

    notes = src / "Team sync (notes).md"
    notes.write_text("# Team sync\n\n## Summary\n\nA made-up meeting.\n", encoding="utf-8")
    code, out, err = _cli(tmp_path, "notes", str(rec), "--into", str(notes))
    assert code == 0, err
    assert "## Transcript" in notes.read_text(encoding="utf-8")
    code, out, err = _cli(tmp_path, "notes", str(rec), "--check", str(notes))
    assert code == 0, err
