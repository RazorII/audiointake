"""The command, the plugin and the skill carry one version, and a release can only go out whole.

Claude Code only hands out a new copy of a plugin when its version changes, and the skill upgrades the command
when it's older than the version the skill names. So all three move together on every release: plugin.json's
``version``, the skill's "audiointake X or newer" and its install line's archive URL, and the README's install
line, all equal ``audiointake.__version__``. The skill installs uv from one pinned release with a pinned SHA-256
(a ``network`` test checks it against the checksum uv publishes).

The rest checks the release machinery itself, never against GitHub:

* the workflows keep their purpose (test.yml's suite, offline and ci-ok jobs; install-smoke installs the
  release's archive by hand only; resolve runs weekly, wheels only; mlx-probe by hand only);
* CODEOWNERS and the pull-request template;
* ``scripts/release.sh`` runs against a fake ``git``, ``gh`` and ``scripts/models-local.sh`` in a temporary
  repo. It refuses to tag on every gate: a version that isn't X.Y.Z or isn't ``__version__``, a dirty tree, a
  checkout that isn't on main, an unset git identity, any author, committer or config name or email other than
  RazorII's noreply identity on the commits GitHub's main doesn't have yet (all commits when there is no main;
  a failure to ask GitHub stops instead of counting as "no main"), a main this checkout lacks, a skipped or failed
  repo-level test, a failed models gate, immutable releases off (a 404 or ``enabled: false``), a gh failure
  reported as itself, a tag elsewhere or a lightweight one, and a tagger other than RazorII's noreply identity;
  after the tag, a failed push, a tag that didn't reach GitHub, and a release that isn't immutable. A rerun
  accepts its own annotated tag at HEAD and an already-published release;
* ``scripts/dev-install.sh`` makes the environment at ``$VIRTUAL_ENV`` (never inside the source folder) and
  installs editable with ``[test]``, or with ``--tool`` installs the tool the way a user gets it.

Files other units write skip while missing and fail under AUDIOINTAKE_RELEASE_STRICT=1, which release.sh sets.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path
from typing import Any

import pytest

from audiointake import __version__

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "claude-plugin"
SKILL = PLUGIN / "skills" / "audiointake" / "SKILL.md"
MANIFEST = PLUGIN / ".claude-plugin" / "plugin.json"
README = ROOT / "README.md"
WORKFLOWS = ROOT / ".github" / "workflows"
SCRIPTS = ROOT / "scripts"
SLUG = "RazorII/audiointake"
NOREPLY = "1577315+RazorII@users.noreply.github.com"
ARCHIVE = re.compile(r"github\.com/RazorII/audiointake/archive/refs/tags/v(\d+\.\d+\.\d+)\.tar\.gz")
UV_URL = re.compile(r"https://github\.com/astral-sh/uv/releases/download/(\d+\.\d+\.\d+)/uv-aarch64-apple-darwin\.tar\.gz")
SHA_LINE = re.compile(r'echo "([0-9a-f]{64})  \$HOME/\.local/bin/uv-aarch64-apple-darwin\.tar\.gz" \| shasum -a 256 -c -')


def _strict() -> bool:
    return os.environ.get("AUDIOINTAKE_RELEASE_STRICT") == "1"


def _need(path: Path) -> Path:
    if not path.exists():
        msg = f"{path.relative_to(ROOT)} isn't there yet"
        if _strict():
            pytest.fail(msg)
        pytest.skip(msg)
    return path


# --- one version everywhere

def test_plugin_version_matches_package():
    manifest = json.loads(_need(MANIFEST).read_text(encoding="utf-8"))
    assert manifest["version"] == __version__


def test_skill_names_the_current_version():
    skill = _need(SKILL).read_text(encoding="utf-8")
    named = (set(re.findall(r"audiointake (\d+\.\d+\.\d+) or newer", skill)) | set(re.findall(r"older than (\d+\.\d+\.\d+)", skill))
             | set(re.findall(r"Installing audiointake (\d+\.\d+\.\d+)", skill)))
    assert named == {__version__}, named


@pytest.mark.parametrize("doc", [SKILL, README, PLUGIN / "README.md"], ids=["SKILL", "README", "plugin-README"])
def test_every_install_line_is_this_releases_archive(doc: Path):
    text = _need(doc).read_text(encoding="utf-8")
    assert set(ARCHIVE.findall(text)) == {__version__}
    assert "git+" not in text, f"{doc.name}: a git+ install needs git, which a Mac without the developer tools lacks"
    assert not re.search(r"github\.com/RazorII/audiointake(?:\.git)?@", text), f"{doc.name}: an install pinned the git way"
    for line in text.splitlines():
        if "uv tool install" in line:
            assert "--python 3.14" in line, f"{doc.name}: an install line without --python 3.14: {line.strip()}"


def _skill_uv() -> tuple[str, str]:
    skill = _need(SKILL).read_text(encoding="utf-8")
    versions, shas = set(UV_URL.findall(skill)), set(SHA_LINE.findall(skill))
    assert len(versions) == 1 and len(shas) == 1, (versions, shas)
    return versions.pop(), shas.pop()


def test_the_skill_installs_uv_from_one_pinned_release():
    version, sha = _skill_uv()
    skill = SKILL.read_text(encoding="utf-8")
    assert re.fullmatch(r"\d+\.\d+\.\d+", version) and re.fullmatch(r"[0-9a-f]{64}", sha)
    block = next(b for b in re.findall(r"```\n(.*?)```", skill, flags=re.S) if "astral-sh/uv" in b)
    # Check, then unpack: the fingerprint is checked before tar runs, and a bad download is deleted.
    assert block.index("shasum -a 256 -c -") < block.index("tar -xzf"), block
    assert "rm -f ~/.local/bin/uv-aarch64-apple-darwin.tar.gz" in block
    assert not re.search(r"\|\s*(?:sh|bash|zsh)\b", skill), "never pipe a download into a shell"


@pytest.mark.network
def test_the_skills_uv_checksum_is_the_one_uv_publishes():
    import urllib.request

    version, sha = _skill_uv()
    url = f"https://github.com/astral-sh/uv/releases/download/{version}/uv-aarch64-apple-darwin.tar.gz.sha256"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            published = r.read().decode("ascii").split()[0]
    except OSError as exc:                                  # URLError, a timeout, no route: no network here
        if _strict():
            pytest.fail(f"couldn't read uv's published checksum ({exc}); a release needs this check")
        pytest.skip(f"no network to read uv's published checksum ({exc})")
    assert published == sha


# --- the workflows keep their purpose

yaml = pytest.importorskip("yaml")


def _wf(name: str) -> dict[str, Any]:
    data = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    data["on"] = data.pop(True, data.get("on"))           # YAML 1.1 reads the key `on` as true
    return data


def _runs(job: dict[str, Any]) -> str:
    return "\n".join(str(s.get("run", "")) for s in job.get("steps", []))


def test_the_test_workflow_has_suite_offline_and_ci_ok():
    wf = _wf("test.yml")
    assert set(wf["jobs"]) == {"suite", "offline", "ci-ok"}
    assert set(wf["on"]) == {"push", "pull_request", "workflow_dispatch"}
    matrix = wf["jobs"]["suite"]["strategy"]["matrix"]["include"]
    assert {(r["os"], r["python"]) for r in matrix} == {("macos-15", "3.12"), ("macos-latest", "3.14")}
    for name in ("suite", "offline"):
        runs = _runs(wf["jobs"][name])
        assert "scripts/dev-install.sh" in runs and "pytest" in runs, name
        assert runs.index("scripts/dev-install.sh") < runs.index("pytest"), name
    offline = _runs(wf["jobs"]["offline"])
    assert "sandbox-exec -p '(version 1)(allow default)(deny network*)'" in offline
    assert '-m "cli_subprocess and not network"' in offline
    assert wf["env"] == {"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def test_ci_ok_sums_up_both_jobs_through_env():
    job = _wf("test.yml")["jobs"]["ci-ok"]
    assert job["if"] == "always()"
    assert job["needs"] == ["suite", "offline"]
    (step,) = job["steps"]
    assert step["env"] == {"SUITE": "${{ needs.suite.result }}", "OFFLINE": "${{ needs.offline.result }}"}
    assert '[ "$SUITE" = success ] && [ "$OFFLINE" = success ]' in step["run"]


def test_install_smoke_installs_the_release_archive_by_hand_only():
    wf = _wf("install-smoke.yml")
    assert list(wf["on"]) == ["workflow_dispatch"]
    assert wf["on"]["workflow_dispatch"]["inputs"]["tag"]["required"] is True
    (job,) = wf["jobs"].values()
    assert job["env"]["TAG"] == "${{ inputs.tag }}"
    runs = _runs(job)
    assert '[[ "$TAG" =~ ^v[0-9]+\\.[0-9]+\\.[0-9]+$ ]]' in runs
    assert 'uv tool install --python 3.14 "https://github.com/RazorII/audiointake/archive/refs/tags/$TAG.tar.gz"' in runs
    assert "audiointake --version" in runs and "audiointake doctor --json" in runs
    assert '[ "$out" = "audiointake ${TAG#v}" ]' in runs, "the version must match exactly, not as a substring"
    assert '[ "$rc" = 3 ]' in runs and "not_set_up" in runs
    assert "dev-install.sh" not in runs, "install-smoke must install the release, not this checkout"
    assert runs.index('[[ "$TAG"') < runs.index("uv tool install")
    assert set(job["strategy"]["matrix"]["os"]) == {"macos-15", "macos-latest"}


def test_resolve_runs_weekly_wheels_only_for_every_python():
    wf = _wf("resolve.yml")
    assert set(wf["on"]) == {"schedule", "workflow_dispatch"}
    (cron,) = [c["cron"] for c in wf["on"]["schedule"]]
    _minute, _hour, dom, month, dow = cron.split()
    assert dom == "*" and month == "*" and dow.isdigit()
    (job,) = wf["jobs"].values()
    assert job["strategy"]["matrix"]["python"] == ["3.12", "3.13", "3.14", "3.15"]
    runs = _runs(job)
    assert "uv pip compile" in runs and "--only-binary :all:" in runs and "--python-platform aarch64-apple-darwin" in runs


def test_mlx_probe_is_by_hand_and_pins_the_projects_mlx():
    wf = _wf("mlx-probe.yml")
    assert list(wf["on"]) == ["workflow_dispatch"]
    (job,) = wf["jobs"].values()
    runs = _runs(job)
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    mlx = next(d for d in deps if d.startswith("mlx=="))
    assert f'"{mlx}"' in runs
    for pkg in re.findall(r'"([A-Za-z0-9_.-]+[=<>!~][^"]*)"', runs):
        assert re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][0-9A-Za-z.]*", pkg), f"not an exact pin: {pkg}"
    assert "mx.ones(4).sum()" in runs and "model.predict" in runs


def test_codeowners_require_razorii():
    lines = [l.split() for l in (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    owners = {l[0]: l[1:] for l in lines}
    assert owners["*"] == ["@RazorII"]
    for path in ("/src/audiointake/setup/", "/src/audiointake/proc.py", "/.github/", "/scripts/", "/pyproject.toml"):
        assert owners.get(path) == ["@RazorII"], path


def test_pull_request_template():
    code = (ROOT / ".github" / "PULL_REQUEST_TEMPLATE" / "code.md").read_text(encoding="utf-8")
    for words in ("No audio in the repo", "tests/test_hygiene.py", "proc.py", "argv.py", "errors.py",
                  "Jane Doe", "network connection", "docs/add-a-platform.md"):
        assert words in code, words


# --- release.sh, read

def test_release_sh_is_gdocouts_core_with_the_design_gates():
    text = (SCRIPTS / "release.sh").read_text(encoding="utf-8")
    for line in ('[[ "$v" =~ ^(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$ ]]',
                 'src/audiointake/__init__.py',
                 'publisher="RazorII"',
                 f'noreply="{NOREPLY}"',
                 'range=FETCH_HEAD..HEAD',
                 "git log --format='%an%n%cn' \"$range\"",
                 "git log --format='%ae%n%ce' \"$range\"",
                 'git config user.email',
                 'AUDIOINTAKE_RELEASE_STRICT=1',
                 'tests/test_release.py tests/test_docs.py tests/test_plugin_folder.py tests/test_workflows.py',
                 'scripts/models-local.sh',
                 'repos/$slug/immutable-releases',
                 'git tag -a "$tag" -m "audiointake $v"',
                 "%(taggername)|%(taggeremail)",
                 'git push --atomic origin HEAD "refs/tags/$tag"',
                 'git ls-remote --exit-code --tags origin "refs/tags/$tag"',
                 'gh release create "$tag" --repo "$slug" --verify-tag',
                 '[ -z "$(git status --porcelain)" ]',
                 'git symbolic-ref --short -q HEAD',
                 'git ls-remote --exit-code --heads origin main'):
        assert line in text, line
    low = text.lower()
    for word in ("pypi", "twine", "pdfintake"):
        assert word not in low, word


# --- release.sh against fakes

FAKE_GIT = r'''
import json, os, sys
state_path = os.environ["FAKE_STATE"]
state = json.load(open(state_path, encoding="utf-8"))
a = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as log:
    log.write("git " + " ".join(a) + "\n")
def save():
    json.dump(state, open(state_path, "w", encoding="utf-8"))
FIELDS = {"%an": "an", "%cn": "cn", "%ae": "ae", "%ce": "ce"}
if a == ["rev-parse", "--show-toplevel"]:
    print(state["root"])
elif a == ["status", "--porcelain"]:
    sys.stdout.write(state["porcelain"])
elif a[:3] == ["ls-remote", "--exit-code", "--heads"]:
    if state["remote_main"] == "error":
        print("fatal: unable to access 'https://github.com/RazorII/audiointake.git/': Could not resolve host",
              file=sys.stderr)
        sys.exit(128)
    sys.exit(0 if state["remote_main"] else 2)
elif a == ["symbolic-ref", "--short", "-q", "HEAD"]:
    if state["branch"] is None:
        sys.exit(1)
    print(state["branch"])
elif a[:2] == ["ls-remote", "--exit-code"] and a[2] == "--tags":
    sys.exit(0 if a[-1].rsplit("/", 1)[1] in state["remote_tags"] else 2)
elif a[0] == "fetch":
    state["fetched"] = True
    save()
elif a[:2] == ["merge-base", "--is-ancestor"]:
    sys.exit(0 if state["fast_forward"] else 1)
elif a[0] == "log":
    fmt = a[1].split("=", 1)[1].split("%n")
    commits = state["commits"] if a[2] == "HEAD" else [c for c in state["commits"] if not c["pushed"]]
    assert a[2] in ("HEAD", "FETCH_HEAD..HEAD"), a
    print("\n".join(c[FIELDS[f]] for c in commits for f in fmt))
elif a == ["config", "user.name"]:
    if state["user"] is None:
        sys.exit(1)
    print(state["user"])
elif a == ["config", "user.email"]:
    if state["email"] is None:
        sys.exit(1)
    print(state["email"])
elif a[:2] == ["cat-file", "-t"]:
    tag = a[2].rsplit("/", 1)[1]
    if tag not in state["tags"]:
        sys.exit(128)
    print(state["tags"][tag]["type"])
elif a == ["rev-parse", "HEAD"]:
    print(state["head"])
elif a[0] == "rev-parse":
    tag = a[1].split("/")[-1].split("^")[0]
    if tag not in state["tags"]:
        sys.exit(128)
    print(state["tags"][tag]["commit"])
elif a[:2] == ["tag", "-a"]:
    tag = a[2]
    assert a[3] == "-m", a
    if tag in state["tags"]:
        print(f"fatal: tag '{tag}' already exists", file=sys.stderr)
        sys.exit(128)
    state["tags"][tag] = {"type": "tag", "commit": state["head"], "message": a[4],
                          "tagger": state.get("tagger_name") or state["user"],
                          "tagger_email": state.get("tagger_email") or state["email"]}
    save()
elif a[0] == "for-each-ref":
    tag = a[-1].rsplit("/", 1)[1]
    t = state["tags"][tag]
    if t["type"] == "tag":
        print(f"{t['tagger']}|<{t['tagger_email']}>")
    else:
        print("|")
elif a[:2] == ["push", "--atomic"]:
    if not state["push_ok"]:
        print("fake git: push refused", file=sys.stderr)
        sys.exit(1)
    if state["push_lands"]:
        state["remote_tags"].append(a[4].rsplit("/", 1)[1])
    save()
else:
    print("fake git: unexpected " + " ".join(a), file=sys.stderr)
    sys.exit(99)
'''

FAKE_GH = r'''
import json, os, sys
state_path = os.environ["FAKE_STATE"]
state = json.load(open(state_path, encoding="utf-8"))
a = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as log:
    log.write("gh " + " ".join(a) + "\n")
def save():
    json.dump(state, open(state_path, "w", encoding="utf-8"))
def not_found():
    print(json.dumps({"message": "Not Found", "status": "404"}))
    print("gh: Not Found (HTTP 404)", file=sys.stderr)
    sys.exit(1)
slug = "repos/RazorII/audiointake"
if a[0] == "api":
    path = a[1]
    if path == f"{slug}/immutable-releases":
        kind = state.get("gh_fails")
        if kind:
            print({"network": "error connecting to api.github.com", "auth": "gh: Bad credentials (HTTP 401)"}[kind],
                  file=sys.stderr)
            sys.exit(1)
        if state["immutable"] == "disabled":
            print(json.dumps({"enabled": False, "enforced_by_owner": False}))
        elif state["immutable"]:
            print(json.dumps({"enabled": True, "enforced_by_owner": False}))
        else:
            not_found()
    elif path.startswith(f"{slug}/releases/tags/"):
        tag = path.rsplit("/", 1)[1]
        rel = state["releases"].get(tag)
        if rel is None or state.get("lag", 0) > 0:
            if rel is not None:
                state["lag"] -= 1
                save()
            not_found()
        print(json.dumps(rel))
    else:
        sys.exit(99)
elif a[:2] == ["release", "create"]:
    tag = a[2]
    assert "--verify-tag" in a and a[a.index("--repo") + 1] == "RazorII/audiointake", a
    if tag not in state["remote_tags"]:
        print("fake gh: tag not on the remote", file=sys.stderr)
        sys.exit(1)
    state["releases"][tag] = {"tag_name": tag, "draft": False, "immutable": state["publish_immutable"],
                              "assets": []}
    save()
else:
    print("fake gh: unexpected " + " ".join(a), file=sys.stderr)
    sys.exit(99)
'''

STUB_TEST = '''
import os
import pytest


def test_gate():
    with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as log:
        log.write("pytest {name} strict=" + os.environ.get("AUDIOINTAKE_RELEASE_STRICT", "") + "\\n")
    if os.environ.get("FAKE_SKIP") == "{name}":
        pytest.skip("planted skip")
    assert os.environ.get("FAKE_FAIL") != "{name}"
'''


def _tool(bin_dir: Path, name: str, code: str) -> None:
    script = bin_dir / f"{name}_fake.py"
    script.write_text(code, encoding="utf-8")
    wrapper = bin_dir / name
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)


def _shell(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(0o755)


def _bash() -> str:
    # The release Mac's /bin/bash is 3.2: the scripts must run there.
    return "/bin/bash" if sys.platform == "darwin" and Path("/bin/bash").exists() else (shutil.which("bash") or "bash")


posix = pytest.mark.skipif(sys.platform == "win32", reason="the release scripts are bash; releases are cut on a Mac")


def _commit(name: str = "RazorII", email: str = NOREPLY, *, pushed: bool = False, **over: str) -> dict[str, Any]:
    c = {"an": name, "cn": name, "ae": email, "ce": email, "pushed": pushed}
    c.update(over)
    return c


class FakeRelease:
    """A temporary repo with release.sh, fake git and gh on PATH, and a state file both share."""

    VERSION = "0.2.0"

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.root = tmp / "my répo"
        self.bin = tmp / "bin"
        self.log = tmp / "calls.log"
        self.state_path = tmp / "state.json"
        self.bin.mkdir()
        for d in ("src/audiointake", "scripts", "tests"):
            (self.root / d).mkdir(parents=True)
        (self.root / "src/audiointake/__init__.py").write_text(
            f'__version__ = "{self.VERSION}"          # the only version source; release.sh greps this exact line\n',
            encoding="utf-8")
        shutil.copy(SCRIPTS / "release.sh", self.root / "scripts/release.sh")
        _shell(self.root / "scripts/models-local.sh", 'echo "models-local" >> "$FAKE_LOG"\nexit "${FAKE_MODELS_RC:-0}"\n')
        for name in ("test_release", "test_docs", "test_plugin_folder", "test_workflows"):
            (self.root / "tests" / f"{name}.py").write_text(STUB_TEST.replace("{name}", name), encoding="utf-8")
        _tool(self.bin, "git", FAKE_GIT)
        _tool(self.bin, "gh", FAKE_GH)
        self.state: dict[str, Any] = {
            "root": str(self.root), "porcelain": "", "user": "RazorII", "email": NOREPLY, "branch": "main",
            "commits": [_commit(pushed=True), _commit()], "remote_main": True, "fast_forward": True,
            "tags": {}, "head": "c0ffee", "remote_tags": [], "push_ok": True, "push_lands": True,
            "immutable": True, "releases": {}, "publish_immutable": True,
        }

    def run(self, version: str | None = None, **env: str) -> subprocess.CompletedProcess[str]:
        self.state_path.write_text(json.dumps(self.state), encoding="utf-8")
        e = {k: v for k, v in os.environ.items() if k not in {"AUDIOINTAKE_RELEASE_STRICT", "VIRTUAL_ENV"}}
        e.update(PATH=f"{self.bin}{os.pathsep}{e['PATH']}", FAKE_STATE=str(self.state_path), FAKE_LOG=str(self.log),
                 PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE="1", **env)
        proc = subprocess.run([_bash(), str(self.root / "scripts/release.sh"), version or self.VERSION],
                              cwd=self.tmp, env=e, capture_output=True, text=True, timeout=300)
        self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        return proc

    def calls(self) -> list[str]:
        return self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []

    def index_of(self, prefix: str) -> int:
        return next(i for i, c in enumerate(self.calls()) if c.startswith(prefix))

    def tagged(self) -> bool:
        return any(c.startswith(("git tag", "git push")) for c in self.calls())

    def released(self) -> bool:
        return any(c.startswith("gh release create") for c in self.calls())


@pytest.fixture
def rel(tmp_path: Path) -> FakeRelease:
    return FakeRelease(tmp_path)


@posix
def test_release_happy_path(rel: FakeRelease):
    proc = rel.run()
    assert proc.returncode == 0, proc.stderr
    assert "v0.2.0 is published as an immutable release" in proc.stdout
    order = [rel.index_of(p) for p in ("pytest test_release strict=", "git status --porcelain",
                                       "git log --format=%an%n%cn FETCH_HEAD..HEAD",
                                       "pytest test_docs strict=1", "models-local",
                                       "gh api repos/RazorII/audiointake/immutable-releases",
                                       "git tag -a v0.2.0 -m audiointake 0.2.0", "git for-each-ref",
                                       "git push --atomic origin HEAD refs/tags/v0.2.0",
                                       "git ls-remote --exit-code --tags origin refs/tags/v0.2.0",
                                       "gh release create v0.2.0")]
    assert order == sorted(order), rel.calls()
    assert rel.state["releases"]["v0.2.0"]["immutable"] is True
    assert rel.state["tags"]["v0.2.0"]["tagger_email"] == NOREPLY


@posix
def test_gate_1_runs_every_repo_level_test_strictly(rel: FakeRelease):
    assert rel.run().returncode == 0
    strict = [c for c in rel.calls() if c.startswith("pytest ") and c.endswith("strict=1")]
    assert sorted(c.split()[1] for c in strict) == ["test_docs", "test_plugin_folder", "test_release", "test_workflows"]


@posix
@pytest.mark.parametrize("case, says", [
    ("not-a-version", "not a version"),
    ("version", "is not at 0.3.0"),
    ("dirty", "commit everything first"),
])
def test_the_basic_gates(rel: FakeRelease, case: str, says: str):
    if case == "dirty":
        rel.state["porcelain"] = " M README.md\n"
    proc = rel.run({"not-a-version": "v0.2", "version": "0.3.0"}.get(case))
    assert proc.returncode != 0 and says in proc.stderr, proc.stderr
    assert not rel.tagged() and not rel.released()


@posix
@pytest.mark.parametrize("who, says", [
    ({"an": "Someone Else"}, "must be by RazorII"),
    ({"cn": "Someone Else"}, "must be by RazorII"),
    ({"ae": "someone@example.com"}, f"must carry {NOREPLY}"),
    ({"ce": "someone@example.com"}, f"must carry {NOREPLY}"),
])
def test_an_unpushed_commit_by_anyone_else_stops_the_release(rel: FakeRelease, who: dict[str, str], says: str):
    rel.state["commits"].append(_commit(**who))
    proc = rel.run()
    assert proc.returncode != 0 and says in proc.stderr, proc.stderr
    assert not rel.tagged() and not any(c.startswith("models-local") for c in rel.calls())


@posix
@pytest.mark.parametrize("key, value, says", [("user", "Someone Else", "must be by RazorII"),
                                              ("email", "someone@example.com", f"must carry {NOREPLY}")])
def test_the_identity_the_tag_would_carry_is_checked_first(rel: FakeRelease, key: str, value: str, says: str):
    rel.state[key] = value
    proc = rel.run()
    assert proc.returncode != 0 and says in proc.stderr, proc.stderr
    assert not rel.tagged()


@posix
def test_a_merged_contribution_on_main_keeps_its_own_names(rel: FakeRelease):
    rel.state["commits"].insert(0, _commit("Contributor", "contributor@example.com", pushed=True))
    proc = rel.run()
    assert proc.returncode == 0, proc.stderr
    assert "git fetch --quiet origin main" in rel.calls()


@posix
def test_without_a_main_on_github_every_commit_is_checked(rel: FakeRelease):
    rel.state["remote_main"] = False
    rel.state["commits"].insert(0, _commit("Contributor", "contributor@example.com", pushed=True))
    proc = rel.run()
    assert proc.returncode != 0 and "must be by RazorII" in proc.stderr
    assert "git log --format=%an%n%cn HEAD" in rel.calls()
    assert not any(c.startswith("git fetch") for c in rel.calls())


@posix
def test_a_failure_to_ask_github_for_main_stops_instead_of_counting_as_no_main(rel: FakeRelease):
    rel.state["remote_main"] = "error"
    rel.state["commits"].insert(0, _commit("Contributor", "contributor@example.com", pushed=True))
    proc = rel.run()
    assert proc.returncode != 0 and "couldn't ask GitHub for its main branch (git ls-remote exit 128)" in proc.stderr
    assert "Could not resolve host" in proc.stderr and "must be by RazorII" not in proc.stderr
    assert not any(c.startswith("git log") for c in rel.calls()) and not rel.tagged()


@posix
@pytest.mark.parametrize("unset", ["user", "email"])
def test_an_unset_git_identity_says_so(rel: FakeRelease, unset: str):
    rel.state[unset] = None
    proc = rel.run()
    assert proc.returncode != 0 and "git config user.name and user.email must be set" in proc.stderr, proc.stderr
    assert not rel.tagged()


@posix
@pytest.mark.parametrize("branch, says", [("feature", "this checkout is on feature"),
                                          (None, "this checkout is on a detached HEAD")])
def test_a_release_is_cut_from_main_only(rel: FakeRelease, branch: str | None, says: str):
    rel.state["branch"] = branch
    proc = rel.run()
    assert proc.returncode != 0 and says in proc.stderr, proc.stderr
    assert not rel.tagged() and not any(c.startswith("models-local") for c in rel.calls())


@posix
def test_a_main_this_checkout_lacks_stops_the_release(rel: FakeRelease):
    rel.state["fast_forward"] = False
    proc = rel.run()
    assert proc.returncode != 0 and "git pull --ff-only origin main first" in proc.stderr
    assert not rel.tagged()


@posix
@pytest.mark.parametrize("plant", ["FAKE_SKIP", "FAKE_FAIL"])
@pytest.mark.parametrize("test", ["test_docs", "test_plugin_folder", "test_workflows"])
def test_gate_1_refuses_a_skipped_or_failed_repo_test(rel: FakeRelease, plant: str, test: str):
    proc = rel.run(**{plant: test})
    assert proc.returncode != 0 and "the repo-level tests must all run and pass" in proc.stderr, proc.stderr
    assert not rel.tagged() and not any(c.startswith("models-local") for c in rel.calls())


@posix
def test_gate_2_refuses_a_failed_models_run(rel: FakeRelease):
    proc = rel.run(FAKE_MODELS_RC="1")
    assert proc.returncode != 0
    assert "models-local" in rel.calls() and not rel.tagged()
    assert not any(c.startswith("gh ") for c in rel.calls())


@posix
def test_gate_2_refuses_without_the_models_script(rel: FakeRelease):
    (rel.root / "scripts/models-local.sh").unlink()
    proc = rel.run()
    assert proc.returncode != 0 and "scripts/models-local.sh is missing" in proc.stderr
    assert not rel.tagged()


@posix
@pytest.mark.parametrize("answer", ["404", "disabled"])
def test_gate_3_refuses_while_immutable_releases_are_off(rel: FakeRelease, answer: str):
    rel.state["immutable"] = False if answer == "404" else "disabled"
    proc = rel.run()
    assert proc.returncode != 0 and "immutable releases are off" in proc.stderr, proc.stderr
    assert not rel.tagged() and not rel.released()


@posix
@pytest.mark.parametrize("kind, said", [("network", "error connecting to api.github.com"),
                                        ("auth", "Bad credentials (HTTP 401)")])
def test_a_failed_settings_read_says_so_instead_of_calling_it_off(rel: FakeRelease, kind: str, said: str):
    rel.state["gh_fails"] = kind
    proc = rel.run()
    assert proc.returncode != 0 and not rel.tagged()
    assert "couldn't read the immutable-releases setting for RazorII/audiointake: " in proc.stderr
    assert said in proc.stderr and "immutable releases are off" not in proc.stderr


@posix
@pytest.mark.parametrize("tagger", [{"tagger_email": "someone@example.com"}, {"tagger_name": "Someone Else"}])
def test_a_tag_by_anyone_else_is_never_pushed(rel: FakeRelease, tagger: dict[str, str]):
    # git takes the tagger from GIT_COMMITTER_NAME/EMAIL when set, which the config check can't see.
    rel.state.update(tagger)
    proc = rel.run()
    assert proc.returncode != 0 and f"must be by RazorII <{NOREPLY}>" in proc.stderr, proc.stderr
    assert "git tag -a v0.2.0 -m audiointake 0.2.0" in rel.calls()
    assert not any(c.startswith("git push") for c in rel.calls()) and not rel.released()


@posix
def test_a_rerun_accepts_its_own_annotated_tag_at_head(rel: FakeRelease):
    rel.state["tags"]["v0.2.0"] = {"type": "tag", "commit": "c0ffee", "tagger": "RazorII", "tagger_email": NOREPLY}
    proc = rel.run()
    assert proc.returncode == 0, proc.stderr
    assert not any(c.startswith("git tag") for c in rel.calls())
    assert "git push --atomic origin HEAD refs/tags/v0.2.0" in rel.calls()


@posix
def test_a_rerun_refuses_its_tag_at_head_by_anyone_else(rel: FakeRelease):
    rel.state["tags"]["v0.2.0"] = {"type": "tag", "commit": "c0ffee", "tagger": "RazorII",
                                   "tagger_email": "someone@example.com"}
    proc = rel.run()
    assert proc.returncode != 0 and "it is by RazorII <someone@example.com>" in proc.stderr
    assert not any(c.startswith("git push") for c in rel.calls())


@posix
@pytest.mark.parametrize("tag", [{"type": "tag", "commit": "0ld"}, {"type": "commit", "commit": "c0ffee"}],
                         ids=["elsewhere", "lightweight"])
def test_a_tag_elsewhere_or_a_lightweight_one_is_never_replaced(rel: FakeRelease, tag: dict[str, str]):
    rel.state["tags"]["v0.2.0"] = dict(tag, tagger="RazorII", tagger_email=NOREPLY)
    proc = rel.run()
    assert proc.returncode != 0 and "is never replaced" in proc.stderr
    assert not any(c.startswith(("git tag", "git push")) for c in rel.calls())


@posix
def test_a_failed_push_publishes_nothing(rel: FakeRelease):
    rel.state["push_ok"] = False
    proc = rel.run()
    assert proc.returncode != 0 and not rel.released()


@posix
def test_a_tag_that_did_not_reach_github_publishes_nothing(rel: FakeRelease):
    rel.state["push_lands"] = False
    proc = rel.run()
    assert proc.returncode != 0 and "v0.2.0 did not reach GitHub" in proc.stderr
    assert not rel.released()


@posix
def test_the_release_listing_lag_is_waited_out(rel: FakeRelease):
    rel.state["lag"] = 1
    proc = rel.run()
    assert proc.returncode == 0, proc.stderr
    looks = [c for c in rel.calls() if c == "gh api repos/RazorII/audiointake/releases/tags/v0.2.0"]
    assert len(looks) == 3, rel.calls()           # before the create, once lagging, once found


@posix
def test_a_release_github_did_not_make_immutable_is_reported(rel: FakeRelease):
    rel.state["publish_immutable"] = False
    proc = rel.run()
    assert proc.returncode != 0 and "isn't immutable" in proc.stderr


@posix
def test_a_rerun_after_a_published_release_creates_nothing(rel: FakeRelease):
    rel.state["tags"]["v0.2.0"] = {"type": "tag", "commit": "c0ffee", "tagger": "RazorII", "tagger_email": NOREPLY}
    rel.state["remote_tags"] = ["v0.2.0"]
    rel.state["releases"]["v0.2.0"] = {"tag_name": "v0.2.0", "draft": False, "immutable": True, "assets": []}
    proc = rel.run()
    assert proc.returncode == 0, proc.stderr
    assert not rel.released()


# --- dev-install.sh against a fake uv

FAKE_UV = r'''
import os, sys
from pathlib import Path
a = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as log:
    log.write("uv " + " ".join(a) + "\n")
if a[0] == "venv":
    py = Path(a[-1]) / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("#!/bin/sh\n", encoding="utf-8")
    py.chmod(0o755)
elif a[:3] == ["tool", "dir", "--bin"]:
    print(os.environ["FAKE_TOOL_BIN"])
elif a[0] in ("pip", "tool"):
    pass
else:
    sys.exit(99)
'''


class FakeInstall:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.root = tmp / "my répo"
        self.bin = tmp / "bin"
        self.log = tmp / "calls.log"
        self.bin.mkdir()
        (self.root / "scripts").mkdir(parents=True)
        shutil.copy(SCRIPTS / "dev-install.sh", self.root / "scripts/dev-install.sh")
        _tool(self.bin, "uv", FAKE_UV)
        self.github_path = tmp / "github_path"

    def run(self, *args: str, cwd: Path | None = None, **env: str) -> subprocess.CompletedProcess[str]:
        e = {k: v for k, v in os.environ.items() if k not in {"VIRTUAL_ENV", "UV", "GITHUB_PATH"}}
        e.update(PATH=f"{self.bin}{os.pathsep}{e['PATH']}", FAKE_LOG=str(self.log),
                 FAKE_TOOL_BIN=str(self.tmp / "toolbin"), PYTHONDONTWRITEBYTECODE="1", **env)
        return subprocess.run([_bash(), str(self.root / "scripts/dev-install.sh"), *args], cwd=cwd or self.tmp,
                              env=e, capture_output=True, text=True, timeout=120)

    def calls(self) -> list[str]:
        return self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []


@pytest.fixture
def inst(tmp_path: Path) -> FakeInstall:
    return FakeInstall(tmp_path)


@posix
def test_dev_install_makes_the_environment_and_installs_editable_with_test(inst: FakeInstall):
    venv = inst.tmp / "dev env"
    proc = inst.run(VIRTUAL_ENV=str(venv), GITHUB_PATH=str(inst.github_path))
    assert proc.returncode == 0, proc.stderr
    assert inst.calls() == [f"uv venv --python 3.14 {venv}", f"uv pip install --python {venv}/bin/python -e .[test]"]
    assert inst.github_path.read_text(encoding="utf-8").strip() == f"{venv}/bin"


@posix
def test_dev_install_reuses_an_environment_and_takes_python_and_extras(inst: FakeInstall):
    venv = inst.tmp / "env"
    assert inst.run("--python", "3.12", VIRTUAL_ENV=str(venv)).returncode == 0
    assert inst.calls()[0] == f"uv venv --python 3.12 {venv}"
    inst.log.unlink()
    assert inst.run("--extras", "test,bench", VIRTUAL_ENV=str(venv)).returncode == 0
    assert inst.calls() == [f"uv pip install --python {venv}/bin/python -e .[test,bench]"]
    inst.log.unlink()
    assert inst.run("--extras", "", VIRTUAL_ENV=str(venv)).returncode == 0
    assert inst.calls() == [f"uv pip install --python {venv}/bin/python -e ."]


@posix
def test_dev_install_resolves_a_relative_environment_from_where_it_was_started(inst: FakeInstall):
    (inst.tmp / "work").mkdir()
    proc = inst.run(cwd=inst.tmp / "work", VIRTUAL_ENV="envs/dev")
    assert proc.returncode == 0, proc.stderr
    assert inst.calls()[0] == f"uv venv --python 3.14 {inst.tmp / 'work' / 'envs/dev'}"


@posix
def test_dev_install_needs_an_environment_outside_the_source_folder(inst: FakeInstall):
    proc = inst.run()
    assert proc.returncode == 2 and "set VIRTUAL_ENV" in proc.stderr
    proc = inst.run(VIRTUAL_ENV=str(inst.root / ".venv"))
    assert proc.returncode == 2 and "inside the source folder" in proc.stderr
    assert inst.calls() == []


@posix
def test_dev_install_tool_mode_is_the_users_install(inst: FakeInstall):
    proc = inst.run("--tool", GITHUB_PATH=str(inst.github_path))
    assert proc.returncode == 0, proc.stderr
    assert inst.calls() == ["uv tool install --force --python 3.14 .", "uv tool dir --bin"]
    assert inst.github_path.read_text(encoding="utf-8").strip() == str(inst.tmp / "toolbin")


@posix
def test_dev_install_needs_uv_and_known_arguments(inst: FakeInstall):
    proc = inst.run(UV="no-such-uv-command", VIRTUAL_ENV=str(inst.tmp / "env"))
    assert proc.returncode == 3 and "needs uv" in proc.stderr
    proc = inst.run("--frobnicate")
    assert proc.returncode == 2 and "unknown argument --frobnicate" in proc.stderr
