"""The public docs agree with the code they describe, and none of them describes a run that asks.

* **Nothing is asked during a run.** No doc names AskUserQuestion, a "decision" exit, or a question flow: asking
  the user, a numbered option list, waiting for a reply, prompting the user. A sentence that says it *doesn't*
  happen ("never asks the user") is fine, as long as the negation governs the match itself and not a condition
  before it. Every Markdown file in the repo and the plugin manifest are scanned.
* **Exit codes.** The exit tables of README.md, SKILL.md and docs/decision-record.md give exactly ``errors``'
  exits 0 to 5 (a row for the signal exits 129, 130 and 143 is allowed); each of the three names SIGTERM, SIGHUP
  and SIGINT and their exits, 143, 129 and 130, and where one gives the order several records decide the process exit in, it is
  ``errors.EXIT_PRIORITY``.
* **Every code is explained** in SKILL.md and docs/decision-record.md, each in backticks; in SKILL.md each error
  under the heading of its exit, and the warnings, notes and offers under theirs. SKILL.md relays every offer in
  core's words: ``errors.OFFERS``' templates appear in it verbatim, and so do the messages it gives before
  anything is installed (``platform_not_apple_silicon``, ``macos_too_old``).
* **Consent, word for word**: the skill's line before an enroll and its line after one, and the
  README's first-screen paragraph on voiceprints. The README's first screen, before its first ``##`` heading,
  says what stays on the Mac, what Claude sees, that voiceprints are biometric data, and that it records nothing.
* **The skill follows the design**: the one-command platform check, the setup line,
  the "just transcribe" statement, `--lang en,es` for a mixed recording, a redo only over outputs proven
  unchanged, the `remember` offer only for the user's own voice, the Command Line Tools flow, `brctl download`
  with a cap, notes written in place and finished by ``notes --into``, no temporary folder and no ``rm -rf``.
* **PRIVACY.md** names every host setup can contact (``HOSTS``, plus every host the setup code and the pack
  manifests spell out) and every file and folder audiointake keeps (``DATA_FILES``, plus every name the code
  joins onto ``data_dir()`` or ``cache_dir()``). The plugin README names every host too.
* **THIRD-PARTY.md** has a row for every runtime dependency in pyproject.toml, its six sections,
  the CAM++, BUT-RTTM and NemoTextProcessing findings, the ported key map's MIT notice, and every model repo a
  pack manifest pins.
* **No ``{{TBD: …}}`` marker is left** in a doc. Markers hold the places measurements fill in; they fail only
  under AUDIOINTAKE_RELEASE_STRICT=1, which scripts/release.sh sets.

A doc or module that isn't written yet skips, and fails under AUDIOINTAKE_RELEASE_STRICT=1.
"""
from __future__ import annotations

import ast
import json
import os
import re
import tomllib
from pathlib import Path
from urllib.parse import urlparse

import pytest

from audiointake import errors

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "audiointake"
README = ROOT / "README.md"
PRIVACY = ROOT / "PRIVACY.md"
THIRD_PARTY = ROOT / "THIRD-PARTY.md"
PLUGIN_README = ROOT / "claude-plugin" / "README.md"
SKILL = ROOT / "claude-plugin" / "skills" / "audiointake" / "SKILL.md"
RECORD_DOC = ROOT / "docs" / "decision-record.md"
PLUGIN_MANIFEST = ROOT / "claude-plugin" / ".claude-plugin" / "plugin.json"

MAIN_CODES = {errors.EXIT_OK, errors.EXIT_FAILED, errors.EXIT_USAGE, errors.EXIT_SETUP, errors.EXIT_REFUSED,
              errors.EXIT_STALE}
SIGNAL_CODES = {errors.EXIT_SIGHUP, errors.EXIT_SIGINT, errors.EXIT_SIGTERM}
ALL_CODES = {getattr(errors, n) for n in dir(errors) if n.startswith("EXIT_") and isinstance(getattr(errors, n), int)}

# Every host setup, an added pack and installing can contact.
HOSTS = ("huggingface.co", "hf.co", "github.com", "codeload.github.com", "objects.githubusercontent.com",
         "release-assets.githubusercontent.com", "groups.inf.ed.ac.uk", "pypi.org", "files.pythonhosted.org",
         "releases.astral.sh")                    # uv fetches Python 3.14 from Astral's mirror when the Mac has none
NOT_CONTACTED = {"localhost", "example.com", "example.org", "example.net"}
# What <data>, the cache and the logs hold.
DATA_FILES = ("records/", "voices/", "models/", "tools/", "packs/", "canary/", "speed.json", "downloads.json",
              "run.lock", "work/", "build/", "~/Library/Logs/audiointake/")
TBD = re.compile(r"\{\{TBD:[^}]*\}\}")

CONSENT_BEFORE = ("Voiceprints are biometric data: only keep one for someone who has agreed to having a voiceprint "
                  "kept.")
CONSENT_AFTER = ("Saved. `audiointake voices remove 'NAME'` deletes the voiceprint; transcripts and records already "
                 "made keep the name they show.")
README_VOICEPRINTS = ("**Voiceprints are biometric data.** audiointake makes one only when you ask, from recordings "
                      "you give it, and keeps it on your Mac (not in iCloud; Time Machine backs it up like the rest of "
                      "your Library). Only enroll people who have agreed to having a voiceprint kept. `audiointake "
                      "voices remove NAME` deletes one; transcripts already made keep the name they show, and so does "
                      "a Time Machine backup.")
FIRST_SCREEN = ("**What stays on your Mac.**", "**What Claude sees.**", "**Voiceprints are biometric data.**",
                "**It records nothing.**")
SETUP_LINE = ("First I'll download audiointake's core models, about {fetch} GB, which takes about {low}–{high} "
              "minutes on a typical connection; I'll transcribe as soon as it's done. Keep the Mac plugged in with "
              "the lid open.")
NOTES_STATEMENT = ("When it's done I'll read the transcript to write notes, which sends its text to Anthropic; say "
                   "'just transcribe' and I won't.")


def _strict() -> bool:
    return os.environ.get("AUDIOINTAKE_RELEASE_STRICT") == "1"


def _not_yet(msg: str) -> None:
    if _strict():
        pytest.fail(msg)
    pytest.skip(msg)


def _read(path: Path) -> str:
    if not path.is_file():
        _not_yet(f"{path.relative_to(ROOT)} isn't written yet")
    return path.read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """Text with blockquote markers dropped and all whitespace runs one space, for word-for-word checks."""
    return re.sub(r"\s+", " ", re.sub(r"(?m)^\s*>\s?", "", text)).strip()


# --------------------------------------------------------------------------- parsers

def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _plain(cell: str) -> str:
    cell = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", cell)
    return cell.replace("`", "").replace("*", "").strip()


def tables(text: str) -> list[list[list[str]]]:
    """Every Markdown table: a list of rows (header first, separator dropped), each a list of cells."""
    out, block = [], []
    for line in text.splitlines() + [""]:
        if line.lstrip().startswith("|"):
            block.append(line)
            continue
        if len(block) >= 2 and re.fullmatch(r"[\s|:-]+", block[1]):
            out.append([_cells(block[0])] + [_cells(r) for r in block[2:]])
        block = []
    return out


EXIT_TABLE_HEADERS = {"code", "exit", "exit code", "exit codes", "exit status", "outcome", "return code"}


def exit_codes_in(text: str) -> set[int]:
    codes: set[int] = set()
    for table in tables(text):
        if _plain(table[0][0]).lower() in EXIT_TABLE_HEADERS:
            for row in table[1:]:
                codes |= {int(n) for n in re.findall(r"\d+", _plain(row[0]))}
    if not codes:
        for m in re.finditer(r"^\s*(?:[-*]|\d+\.)\s+\*\*([0-9][0-9, or/]*?)[.:]?\*\*", text, flags=re.M):
            codes |= {int(n) for n in re.findall(r"\d+", m.group(1))}
    return codes


def titled_section(text: str, title: str) -> str:
    """The body under the first heading that starts with ``title`` (with or without a number, "## 3. Title"),
    up to the next heading of the same or a higher level."""
    m = re.search(rf"^(#+)\s+(?:\d+\.\s+)?{re.escape(title)}[^\n]*$", text, flags=re.M)
    if not m:
        return ""
    level = len(m.group(1))
    end = re.search(rf"^#{{1,{level}}}\s", text[m.end():], flags=re.M)
    return text[m.end(): m.end() + end.start()] if end else text[m.end():]


def codes_in(text: str) -> set[str]:
    """Every snake_case code written in backticks (``pack_not_installed``; not ``EXIT_PRIORITY``)."""
    return {c for c in re.findall(r"`([^`\n]+)`", text) if re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", c)}


def _dist_name(requirement: str) -> str:
    m = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return re.sub(r"[-_.]+", "-", m.group(1)).lower() if m else ""


def third_party_names(text: str) -> set[str]:
    names = set()
    for table in tables(text):
        for row in table:
            m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", _plain(row[0]))
            if m:
                names.add(re.sub(r"[-_.]+", "-", m.group(1)).lower())
    return names


_CODE_RUN = re.compile(r"(?<!\d)(?<!\d\.)\d+(?:\s*(?:,\s*(?:then\s+|and\s+)?|>|\bthen\b|\band\b)\s*\d+(?!\d|\.\d)){2,}")
_PRIORITY_WORDS = re.compile(r"priorit|most\s+important|\border\b", re.I)


def priority_sequences(text: str) -> list[tuple[int, ...]]:
    """Every run of three or more exit codes whose sentence speaks of their priority or order."""
    out = []
    for m in _CODE_RUN.finditer(text):
        sentence_start = max(text.rfind(". ", 0, m.start()), text.rfind("\n", 0, m.start())) + 1
        if _PRIORITY_WORDS.search(text[sentence_start: m.start()]):
            out.append(tuple(int(n) for n in re.findall(r"\d+", m.group())))
    return out


# --------------------------------------------------------------------------- nothing is asked during a run

ALWAYS_FORBIDDEN = (("AskUserQuestion", re.compile(r"AskUserQuestion")),)
QUESTION_FLOW = (
    ("a decision exit", re.compile(r"\bEXIT_DECISION\b")),
    ("a question flow", re.compile(r"\bask(?:s|ed)?\s+(?:the\s+user|them|the\s+question|it\s+like|again|"
                                   r"(?:its|the|a|each|one)\s+questions?)\b", re.I)),
    ("a question flow", re.compile(r"\basks?\s+you\b", re.I)),
    ("a question flow", re.compile(r"\bquestions?\s+(?:audiointake|claude|the\s+skill|it|a\s+run)\s+asks?\b", re.I)),
    ("a question flow", re.compile(r"\bhands?\s+back\s+a\s+question\b", re.I)),
    ("a question flow", re.compile(r"\banswer(?:s|ing)?\s+(?:a|the|its|each)\s+question", re.I)),
    ("a question flow", re.compile(r"\b(?:one\s+question\s+per|(?:report|show)s?\s+(?:each|the|its)\s+question)\b", re.I)),
    ("a question flow", re.compile(r"\bnumbered\s+(?:option\s+)?list\b", re.I)),
    ("a question flow", re.compile(r"\bwait(?:s|ing)?\s+for\s+(?:the|their|an?|your)\s+(?:reply|answer|user|confirmation)", re.I)),
    ("a question flow", re.compile(r"\bprompt(?:s|ed|ing)?\s+(?:the\s+user|them|you)\b", re.I)),
    ("a question flow", re.compile(r"\bwhen\s+(?:you|the\s+user)\s+agrees?\b", re.I)),
    ("a question flow", re.compile(r"\bdo\s+you\s+want\s+me\s+to\b", re.I)),
    ("a question flow", re.compile(r"\bdecision\.(?:question|apply_flag|options|kind|answer)\b")),
    ("a question flow", re.compile(r"\bconfirm\s+with\b", re.I)),
    ("a question flow", re.compile(r"\bcheck\s+with\s+(?:the\s+user|them|you)\b", re.I)),
    ("a question flow", re.compile(r"\b(?:should|shall|may)\s+I\b[^.?!\n]*\?", re.I)),
    ("a question flow", re.compile(r"\b(?:would|do)\s+you\s+(?:like|want|prefer)\b[^.?!\n]*\?", re.I)),
    ("a question flow", re.compile(r"\blet\s+(?:the\s+user|them|you)\s+(?:choose|pick|decide|select)\b", re.I)),
    ("a question flow", re.compile(r"\b(?:get|request|obtain|seek)\s+(?:the\s+user['’]s\s+|their\s+|your\s+)?"
                                   r"(?:ok|okay|confirmation|approval|go-ahead|consent)\b", re.I)),
    ("a question flow", re.compile(r"\bwait\s+(?:until|till|for)\s+(?:the\s+user|they|them|you)\b", re.I)),
    ("a question flow", re.compile(r"\binquire\b", re.I)),
    ("a question flow", re.compile(r"\bas\s+(?:numbered\s+)?(?:options|choices)\b|\boptions?\s+\(?1\)?\s*(?:,|and|or)\s*\(?2\b",
                                   re.I)),
)
# "Ask for / whether / which …" is a question flow only when Claude, audiointake or the skill does the asking (an
# imperative, or one of `_ASKERS` as the subject): "when you ask for it" is the user's own request, which is fine,
# and so is the passive "asked for with `--lang`" (what the user requested).
ASK_FOR = re.compile(r"\bask(?:s|ing)?\s+for\b", re.I)
ASK_WHETHER = re.compile(r"\bask(?:s|ed|ing)?\s+(?:whether|which|if|what|about)\b(?!\s+you\s+want)", re.I)
_ASKERS = {"claude", "audiointake", "the skill", "skill", "it", "run", "the run", "the cli", "we", "i"}
_FILLERS = {"then", "first", "also", "now", "just", "always", "only", "please", "should", "will", "can",
            "must", "may", "would", "could", "later", "so", "and"}
_CLAUSE_END = re.compile(r"[.!?;,](?=\s|$)|:\s|\s(?:and|but|then|so|or|while|when)\s|\n")
_NEGATION_WORD = re.compile(r"^\W*(?:never|no|not|nor|nothing|none|nobody|without)\W*$|n['’]t\W*$", re.I)
_CONDITIONAL = {"if", "unless", "when", "whenever", "once"}
_NEGATION_REACH = 4


def _clause(text: str, start: int, end: int) -> tuple[str, str]:
    before = [m.end() for m in _CLAUSE_END.finditer(text, 0, start)]
    left = before[-1] if before else 0
    after = _CLAUSE_END.search(text, end)
    return text[left:start], text[end: after.start() if after else len(text)]


def _words(before: str) -> list[str]:
    return [w for w in (re.sub(r"[`*_>\"“”]", "", w).lower() for w in before.split()) if w and w != "-"]


def _negated(text: str, start: int, end: int) -> bool:
    """True when the match itself is negated: a negation among the few words right before it ("never asks the
    user", "without stopping to ask you"). In a clause that opens with a condition only the word right before the
    match counts: "If nothing opens it ask the user" asks."""
    before, _after = _clause(text, start, end)
    words = _words(before)
    conditional = any(w.strip("()") in _CONDITIONAL for w in words) or (words[:1] == ["without"] and len(words) > 1)
    reach = words[-1:] if conditional else words[-_NEGATION_REACH:]
    return any(_NEGATION_WORD.search(w) for w in reach)


def _asked_by_the_skill(text: str, start: int, end: int, *, you_count: bool) -> bool:
    before, _ = _clause(text, start, end)
    words = [w for w in _words(before) if not re.fullmatch(r"\d+\.|\(?[a-z0-9]\)", w)]
    subject = " ".join(w for w in words if w not in _FILLERS and not _NEGATION_WORD.search(w))
    return subject == "" or subject in _ASKERS or (you_count and subject == "you")


def question_flow_problems(text: str) -> list[str]:
    """Each place ``text`` describes asking during a run, as ``line N: what: excerpt``."""
    hits = []
    for what, pattern in ALWAYS_FORBIDDEN:
        hits += [(m.start(), what) for m in pattern.finditer(text)]
    for what, pattern in QUESTION_FLOW:
        hits += [(m.start(), what) for m in pattern.finditer(text) if not _negated(text, m.start(), m.end())]
    for pattern, you_count in ((ASK_FOR, False), (ASK_WHETHER, True)):
        hits += [(m.start(), "a question flow") for m in pattern.finditer(text)
                 if not _negated(text, m.start(), m.end())
                 and _asked_by_the_skill(text, m.start(), m.end(), you_count=you_count)]
    problems = []
    for start, what in sorted(set(hits)):
        line_no = text.count("\n", 0, start) + 1
        line = text.splitlines()[line_no - 1].strip()
        problems.append(f"line {line_no}: {what}: {line[:100]}")
    for code in sorted(exit_codes_in(text) - MAIN_CODES - SIGNAL_CODES):
        problems.append(f"an exit-code table or bullet lists {code}, which audiointake doesn't exit with")
    return problems


def scanned_docs() -> list[Path]:
    """Every Markdown file in the repo (git data, tests and virtualenvs aside), and the plugin manifest."""
    skip = {".git", "tests", ".venv", "venv", "node_modules", "build", "dist"}
    docs = [p for p in ROOT.rglob("*.md") if not skip & set(p.relative_to(ROOT).parts[:-1])]
    if PLUGIN_MANIFEST.is_file():
        docs.append(PLUGIN_MANIFEST)
    return sorted(docs)


@pytest.mark.parametrize("doc", scanned_docs(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_doc_describes_a_run_that_asks(doc: Path):
    problems = question_flow_problems(doc.read_text(encoding="utf-8"))
    assert problems == [], f"{doc.relative_to(ROOT)} describes asking during a run: " + " | ".join(problems)


def test_the_skill_and_the_readmes_are_scanned():
    for doc in (SKILL, README, PLUGIN_README):
        _read(doc)
        assert doc in scanned_docs(), doc


def test_errors_has_no_decision_exit():
    assert not hasattr(errors, "EXIT_DECISION")
    assert ALL_CODES == MAIN_CODES | SIGNAL_CODES, sorted(ALL_CODES)


# --------------------------------------------------------------------------- exit codes

@pytest.mark.parametrize("doc", [SKILL, README, RECORD_DOC], ids=["SKILL", "README", "decision-record"])
def test_the_exit_table_is_errors(doc: Path):
    codes = exit_codes_in(_read(doc))
    assert codes in (MAIN_CODES, MAIN_CODES | SIGNAL_CODES), (
        f"{doc.relative_to(ROOT)} documents exits {sorted(codes)}; errors.py has {sorted(MAIN_CODES)} "
        f"(and {sorted(SIGNAL_CODES)} for the signals)")


@pytest.mark.parametrize("doc,required", [(SKILL, True), (RECORD_DOC, True), (README, False)],
                         ids=["SKILL", "decision-record", "README"])
def test_the_exit_priority_order_is_errors(doc: Path, required: bool):
    found = priority_sequences(_read(doc))
    if required:
        assert found, f"{doc.relative_to(ROOT)} doesn't give the exit priority order"
    wrong = [seq for seq in found if seq != tuple(errors.EXIT_PRIORITY)]
    assert wrong == [], f"{doc.relative_to(ROOT)} gives the exit priority as {wrong}; errors has {errors.EXIT_PRIORITY}"


@pytest.mark.parametrize("doc", [SKILL, README, RECORD_DOC], ids=["SKILL", "README", "decision-record"])
def test_every_exit_doc_gives_the_stop_signal_exit(doc: Path):
    text = _read(doc)
    for word in ("SIGTERM", "SIGHUP", "SIGINT", "143", "129", "130"):
        assert word in text, f"{doc.relative_to(ROOT)} doesn't say {word!r}"


@pytest.mark.parametrize("doc", [SKILL, README], ids=["SKILL", "README"])
def test_each_exit_row_names_its_codes(doc: Path):
    rows: dict[int, str] = {}
    for table in tables(_read(doc)):
        if _plain(table[0][0]).lower() in EXIT_TABLE_HEADERS:
            for row in table[1:]:
                for n in re.findall(r"\d+", _plain(row[0])):
                    rows[int(n)] = " | ".join(row)
    for code, (kind, ex) in errors.CODES.items():
        if kind == "error" and ex in (errors.EXIT_FAILED, errors.EXIT_REFUSED, errors.EXIT_STALE):
            assert f"`{code}`" in rows.get(ex, ""), f"{doc.relative_to(ROOT)}: the exit {ex} row doesn't name {code}"
    assert "`input_not_file`" in rows.get(errors.EXIT_USAGE, "")


# --------------------------------------------------------------------------- every code explained

def _codes(kind: str) -> set[str]:
    return {c for c, (k, _ex) in errors.CODES.items() if k == kind}


@pytest.mark.parametrize("doc", [SKILL, RECORD_DOC], ids=["SKILL", "decision-record"])
def test_every_code_is_explained(doc: Path):
    text = _read(doc)
    missing = sorted(c for c in errors.CODES if f"`{c}`" not in text)
    assert missing == [], f"{doc.relative_to(ROOT)} doesn't explain {missing}"


def test_the_skill_explains_each_error_under_its_exit():
    skill = _read(SKILL)
    for code, (kind, ex) in errors.CODES.items():
        if kind != "error":
            continue
        body = titled_section(skill, f"Exit {ex}:")
        assert body, f"SKILL.md has no 'Exit {ex}: …' heading"
        assert f"`{code}`" in body, f"SKILL.md explains {code} outside its 'Exit {ex}' section"


@pytest.mark.parametrize("heading,kind", [("Warnings", "warning"), ("Notes", "note"), ("Offers", "offer")])
def test_the_skill_explains_warnings_notes_and_offers_in_their_sections(heading: str, kind: str):
    body = titled_section(_read(SKILL), heading)
    assert body, f"SKILL.md has no '{heading}' section"
    expected = _codes(kind) | ({c for c in errors.ALSO_WARNING} if kind == "warning" else set())
    missing = sorted(c for c in expected if f"`{c}`" not in body)
    assert missing == [], f"SKILL.md → {heading} doesn't explain {missing}"


@pytest.mark.parametrize("kind", sorted(errors.OFFERS))
def test_the_skill_relays_each_offer_in_cores_words(kind: str):
    assert _flat(errors.OFFERS[kind]) in _flat(titled_section(_read(SKILL), "Offers")), kind


@pytest.mark.parametrize("code", ["platform_not_apple_silicon", "macos_too_old"])
def test_the_skill_gives_cores_platform_messages(code: str):
    assert _flat(errors.MESSAGES[code]) in _flat(titled_section(_read(SKILL), "1. Where it runs")), code


# --------------------------------------------------------------------------- consent and the first screen

def test_the_skill_carries_the_consent_lines_word_for_word():
    flat = _flat(_read(SKILL))
    assert CONSENT_BEFORE in flat and CONSENT_AFTER in flat
    later = _flat(titled_section(_read(SKILL), "8. Later"))
    assert later.index(CONSENT_BEFORE) < later.index("audiointake enroll 'Jane' --from") < later.index(CONSENT_AFTER)


def _first_screen(text: str) -> str:
    m = re.search(r"^## ", text, flags=re.M)
    return text[: m.start()] if m else text


def test_the_readme_first_screen():
    first = _first_screen(_read(README))
    for lead in FIRST_SCREEN:
        assert lead in first, f"README.md's first screen doesn't have {lead}"
    assert _flat(README_VOICEPRINTS) in _flat(first)
    assert "transcribing never goes online" in first and '"just transcribe"' in first


# --------------------------------------------------------------------------- the skill follows the design

def test_the_skill_checks_the_platform_in_one_command_first():
    where = titled_section(_read(SKILL), "1. Where it runs")
    assert "uname -s; sysctl -n hw.optional.arm64" in where and "sw_vers -productVersion" in where
    assert "Linux" in where and "then stop" in where.lower()
    assert "Anything but `Darwin`" in where and "MINGW64_NT" in where, "Windows must stop too, not reach the install"
    skill = _read(SKILL)
    assert skill.index("## 1. Where it runs") < skill.index("## 2. Find audiointake")


def test_the_skill_announces_setup_and_the_notes():
    skill = _flat(_read(SKILL))
    assert _flat(SETUP_LINE) in skill
    assert _flat(NOTES_STATEMENT) in skill
    assert "audiointake setup --plan --json" in skill and "audiointake setup --json" in skill


def test_the_skill_maps_languages_as_d48_says():
    flags = titled_section(_read(SKILL), "4. Turn the request into flags")
    assert "`--lang en,es`" in flags and "only when the whole recording is in that language" in flags
    assert '"mostly Spanish"' in flags and '"we switch"' in flags


def test_the_skill_redoes_only_over_outputs_proven_unchanged():
    flags = _flat(titled_section(_read(SKILL), "4. Turn the request into flags"))
    for words in ('"redo it", "again", "with X instead"', "audiointake record show <record id> --json",
                  "shasum -a 256", "If every file matches, pass `--overwrite`"):
        assert words in flags, words


def test_the_skill_relays_remember_only_for_the_users_own_voice():
    summary = _flat(titled_section(_read(SKILL), "7. One summary"))
    assert "A `remember` offer only when the user named a speaker as themselves" in summary
    assert "never for anyone else" in summary


def test_the_skill_handles_missing_command_line_tools_as_d42_says():
    setup = _flat(titled_section(_read(SKILL), "3. First use"))
    for words in ("`xcode-select --install`", "Transcribe at once", "`xcode-select -p`",
                  "audiointake apply <record id> --choice speakers --json"):
        assert words in setup, words
    assert setup.index("`xcode-select --install`") < setup.index("Transcribe at once") < setup.index("`xcode-select -p`")


def test_the_skill_brings_icloud_placeholders_down_with_a_cap():
    flags = _flat(titled_section(_read(SKILL), "4. Turn the request into flags"))
    # The flag, not ls -lO's whole line (a file name may hold the word); 110 checks fit a 10-minute timeout.
    assert ("brctl download 'FILE'; for i in $(seq 110); do stat -f %Sf 'FILE' | grep -q dataless || break; "
            "sleep 5; done") in flags
    assert "10-minute timeout" in flags and "input_not_downloaded" in flags
    assert "ls -lO" not in flags


def test_the_skill_names_where_a_paths_list_goes_and_quotes_names():
    flags = _flat(titled_section(_read(SKILL), "4. Turn the request into flags"))
    assert "`~/Library/Caches/audiointake/paths-from.txt`" in flags
    assert "--paths-from ~/Library/Caches/audiointake/paths-from.txt" in flags
    assert "**Names** (for `relabel --name` and `enroll`) go in single quotes too" in flags
    assert "A name you guessed from the transcript goes into a command only once the user has said it." in flags


def test_the_skill_relays_cores_rendered_words():
    skill = _flat(_read(SKILL))
    assert "each offer's `summary` from the record, word for word" in skill
    assert "its `message` in the record says it" in skill
    assert "Fill `{" not in skill, "the record already carries core's words, filled in"


def test_the_skill_announces_installing_audiointake_and_gives_it_a_timeout():
    find = _flat(titled_section(_read(SKILL), "2. Find audiointake"))
    assert "Installing audiointake 0.1.0 with uv into ~/.local (about 1.2 GB with its Python libraries)." in find
    assert "10-minute timeout" in find
    assert find.index("`audiointake --version`") < find.index("`~/.local/bin/audiointake --version`")


@pytest.mark.parametrize("doc", [README, SKILL, PLUGIN_README], ids=["README", "SKILL", "plugin-README"])
def test_every_format_example_keeps_md_and_json(doc: Path):
    # --format lists every file to write: without md and json, relabel, notes and apply have nothing to work from.
    for value in re.findall(r"--format[ =]([a-z,]+)", _read(doc)):
        assert value == "sidecars" or {"md", "json"} <= set(value.split(",")), f"{doc.name}: --format {value}"


def test_the_skill_writes_notes_in_place_and_lets_audiointake_append_the_transcript():
    notes = _flat(titled_section(_read(SKILL), "6. Meeting notes"))
    for words in ("audiointake notes <record id> --into '<name> (notes).md' --json",
                  "audiointake notes <record id> --check '<name> (notes).md' --json",
                  "Never type out the transcript yourself", "Never a `## Transcript` heading"):
        assert words in notes, words


def test_the_skill_uses_no_temporary_folder():
    skill = _read(SKILL)
    assert "mktemp" not in skill
    rm_rf = [ln for ln in skill.splitlines() if "rm -rf" in ln]
    assert rm_rf and all(ln.lstrip().startswith("- Never") for ln in rm_rf), rm_rf


def test_the_skill_names_the_thai_models_terms_when_it_offers_the_pack():
    """Both ways the skill puts the Thai pack in front of the user name the terms its model comes with beyond its
    licence: installing it (section 8, after the plan) and relaying a `language` offer whose `th` line carries the
    `apply --lang th` command that downloads it (section 9). The offer's own words stay core's, so the terms are a
    line of the skill's after them, never inside the quoted template. The address is the one THIRD-PARTY.md and
    the README link to."""
    terms = "the Thai model comes with OpenTyphoon's terms (opentyphoon.ai/tac)"
    skill = _read(SKILL)
    later = _flat(titled_section(skill, "8. Later"))
    start = later.index('"Install the Hungarian pack"')
    offer = later[start: later.index("- **", start)]
    assert terms in offer
    offers = titled_section(skill, "Offers")
    assert f"After the line for a `language` offer's `th` option, add one more: {terms}." in _flat(offers)
    assert [ln for ln in offers.splitlines() if terms in ln and not ln.lstrip().startswith(">")], \
        "the terms line is inside a quoted offer template"
    assert terms not in errors.OFFERS["language"]
    for doc in (THIRD_PARTY, README):
        assert "https://opentyphoon.ai/tac" in _read(doc), doc.name


def test_the_readme_says_which_macos_brctl_was_tried_on():
    limits = titled_section(_read(README), "Known limits")
    assert "`brctl download`" in limits and "macOS 27.2" in limits


# --------------------------------------------------------------------------- hosts and data files

_BARE_HOST = re.compile(r"^\.?((?:[a-z0-9-]+\.)+(?:co|com|org|net|io|uk|ai|dev))$")
_HOST_NAME = re.compile(r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}")


def _string_constants(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def hosts_in_code(src: Path = SRC) -> set[str]:
    """Every host the setup code, about, doctor and the pack manifests spell out: a URL's host, or a string
    that is a host name on its own (an allowlist entry)."""
    found: set[str] = set()
    files = sorted((src / "setup").rglob("*.py")) + [src / "about.py", src / "doctor.py"]
    texts: list[str] = []
    for f in files:
        if f.is_file():
            texts += _string_constants(f)
    for f in sorted((src / "setup" / "packs").glob("*.json")):
        texts += re.findall(r'"(https?://[^"]+)"', f.read_text(encoding="utf-8"))
    for s in texts:
        for url in re.findall(r"https?://[^\s\"'<>)]+", s):
            try:
                host = urlparse(url).hostname
            except ValueError:                        # a pattern, not a URL (r"https?://([a-z0-9.-]+)…")
                continue
            if host and _HOST_NAME.fullmatch(host.lower()):
                found.add(host.lower())
        m = _BARE_HOST.match(s.strip().lower())
        if m:
            found.add(m.group(1))
    return {h for h in found if h not in NOT_CONTACTED and not h.startswith("127.")}


def test_hosts_in_code_reads_urls_and_allowlists(tmp_path: Path):
    setup = tmp_path / "setup"
    (setup / "packs").mkdir(parents=True)
    (setup / "download.py").write_text('URL = "https://Example.org/x"\nALLOWED = {"huggingface.co", ".hf.co"}\n'
                                       'LOCAL = "http://127.0.0.1:8080/x"\nWORD = "speed.json"\n', encoding="utf-8")
    (setup / "packs" / "core.json").write_text('{"canary": {"url": "https://groups.inf.ed.ac.uk/ami/x.wav"}}',
                                               encoding="utf-8")
    assert hosts_in_code(tmp_path) == {"huggingface.co", "hf.co", "groups.inf.ed.ac.uk"}


@pytest.mark.parametrize("doc", [PRIVACY, PLUGIN_README], ids=["PRIVACY", "plugin-README"])
def test_every_host_is_named(doc: Path):
    text = _read(doc)
    missing = sorted(h for h in set(HOSTS) | hosts_in_code() if f"`{h}`" not in text)
    assert missing == [], f"{doc.relative_to(ROOT)} doesn't name {missing}"


def data_names_in_code(src: Path = SRC) -> set[str]:
    """The first name the code joins onto ``data_dir()`` or ``cache_dir()`` (``data_dir() / "records"``)."""
    names: set[str] = set()
    for f in sorted(src.rglob("*.py")):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and isinstance(node.left, ast.Call)
                    and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)):
                func = node.left.func
                fname = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
                if fname in ("data_dir", "cache_dir"):
                    names.add(node.right.value.split("/")[0])
    return names


def test_data_names_in_code_reads_joins(tmp_path: Path):
    (tmp_path / "m.py").write_text("from x import platforms, data_dir\na = platforms.data_dir() / 'records' / 'x'\n"
                                   "b = data_dir() / 'speed.json'\nc = cache_dir() / 'work/run'\nd = other() / 'no'\n",
                                   encoding="utf-8")
    assert data_names_in_code(tmp_path) == {"records", "speed.json", "work"}


def test_privacy_names_every_file_audiointake_keeps():
    text = _read(PRIVACY)
    expected = set(DATA_FILES) | {n + ("" if "." in n else "/") for n in data_names_in_code()}
    missing = sorted(n for n in expected if f"`{n}`" not in text)
    assert missing == [], f"PRIVACY.md doesn't name {missing}"


def test_privacy_says_what_the_design_promises():
    text = _flat(_read(PRIVACY))
    for words in ("No Hugging Face token is ever sent", "`ORT_DISABLE_TELEMETRY=1`", "`HF_HUB_DISABLE_TELEMETRY=1`",
                  "deleted after 90 days", "Time Machine", "Anthropic's privacy policy", "It never holds transcript text",
                  "`audiointake transcribe` makes no network connection"):
        assert words in text, words


# --------------------------------------------------------------------------- THIRD-PARTY.md

def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_every_runtime_dependency_has_a_third_party_row():
    rows = third_party_names(_read(THIRD_PARTY))
    deps = sorted({_dist_name(d) for d in _pyproject()["project"]["dependencies"]})
    missing = [d for d in deps if d not in rows]
    assert missing == [], f"THIRD-PARTY.md has no row for {missing}"


def test_third_party_has_the_designs_sections_and_findings():
    text = _read(THIRD_PARTY)
    for heading in ("Runtime dependencies", "Models", "Built at setup on your Mac", "Code ported from elsewhere",
                    "Test and bench data", "Never used"):
        assert titled_section(text, heading), f"THIRD-PARTY.md has no '{heading}' section"
    models = titled_section(text, "Models")
    cam = next(ln for ln in models.splitlines() if "CAM++" in ln)
    assert "Apache-2.0" in cam and "Checked at all three sources on 2026-11-17" in cam
    built = titled_section(text, "Built at setup on your Mac")
    nemo = next(ln for ln in built.splitlines() if "NemoTextProcessing" in ln)
    assert "v0.3.1" in nemo and "v0.3.0" in nemo and "Apache-2.0" in nemo
    data = titled_section(text, "Test and bench data")
    but = next(ln for ln in data.splitlines() if "BUT" in ln)
    assert "Apache-2.0" in but and "re-read on 2026-11-17" in but
    ported = titled_section(text, "Code ported from elsewhere")
    assert "Copyright © 2023 Apple Inc." in ported and "e52c128d113f10546f0fa391f87edcc58d3880cb" in ported


def test_third_party_names_every_model_a_manifest_pins():
    manifests = sorted((SRC / "setup" / "packs").glob("*.json"))
    if not manifests:
        _not_yet("the pack manifests (src/audiointake/setup/packs/*.json) aren't written yet")
    text = _read(THIRD_PARTY)
    repos = set()
    for m in manifests:
        for model in json.loads(m.read_text(encoding="utf-8")).get("models", []):
            if model.get("repo"):
                repos.add(model["repo"])
    missing = sorted(r for r in repos if r not in text)
    assert missing == [], f"THIRD-PARTY.md doesn't name the model repos {missing}"


# --------------------------------------------------------------------------- markers

def tbd_markers() -> list[str]:
    found = []
    for doc in scanned_docs():
        for n, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for m in TBD.finditer(line):
                found.append(f"{doc.relative_to(ROOT)}:{n}: {m.group(0)}")
    return found


def test_no_tbd_marker_is_left():
    found = tbd_markers()
    if found and not _strict():
        pytest.skip(f"{len(found)} TBD markers wait for measurements (strict mode fails on them)")
    assert found == [], "fill these before a release:\n" + "\n".join(found)


# --------------------------------------------------------------------------- the parsers do what they claim

def test_exit_codes_from_a_table_and_from_bullets():
    text = "| Exit | Meaning |\n|---|---|\n| 0 | done |\n| `1` | failed |\n| 129 / 143 | signals |\n"
    assert exit_codes_in(text) == {0, 1, 129, 143}
    assert exit_codes_in("- **0.** Report.\n- **1, 2 or 3:** Report the error.\n") == {0, 1, 2, 3}
    assert exit_codes_in("| Exit | x |\n|:-:|---|\n| 5 | stale |\n\n- **7.** other\n") == {5}


def test_priority_sequences():
    assert priority_sequences("the most important code among them, in the order 2, 3, 1, 4, 5, 0.") == [(2, 3, 1, 4, 5, 0)]
    assert priority_sequences("The exit priority is 2 > 3 > 1 > 4 > 5 > 0.") == [(2, 3, 1, 4, 5, 0)]
    assert priority_sequences("Python 3.12, 3.13 and 3.14 in that order.") == []
    assert priority_sequences("Speakers 1, 5, 7 were named.") == []


def test_titled_section_and_codes_in():
    text = ("# Page\n\n## 9. What the codes mean\n\n### Exit 2: usage\n\n- `input_missing`: x\n\n"
            "### Exit 3: setup\n\n- `not_set_up` and `EXIT_SETUP`\n\n## Never\n")
    assert codes_in(titled_section(text, "Exit 2:")) == {"input_missing"}
    assert codes_in(titled_section(text, "Exit 3:")) == {"not_set_up"}
    assert "Exit 3" in titled_section(text, "What the codes mean")
    assert titled_section(text, "Exit 4:") == ""


def test_third_party_names_normalise():
    text = ("| Package | Licence |\n|---|---|\n| `parakeet-mlx` | Apache-2.0 |\n| [mlx_whisper](https://example.com) | MIT |\n"
            "| **huggingface_hub** ==2.1.1 | Apache-2.0 |\n")
    assert {"parakeet-mlx", "mlx-whisper", "huggingface-hub"} <= third_party_names(text)
    assert _dist_name("huggingface-hub==2.1.1") == "huggingface-hub"


def test_the_marker_pattern():
    assert TBD.findall("x {{TBD: U5}} y {{TBD: U5, Spanish}} z {{ TBD }}") == ["{{TBD: U5}}", "{{TBD: U5, Spanish}}"]


OLD_FLOWS = [
    "Use AskUserQuestion, one question per recording.",
    "No AskUserQuestion here.",
    "Ask the user which language the recording is in, then run apply.",
    "Otherwise Claude asks you first, and names the download size.",
    "### Questions audiointake asks",
    "It hands back a question in a JSON record.",
    "Show the packs as a short numbered list.",
    "Then wait for the reply.",
    "Then wait for the user to confirm.",
    "Report each question and its options, then stop.",
    "Prompt the user for the speaker's name.",
    "Ask for the name, then run enroll.",
    "Ask whether the recording is in Spanish before running.",
    "Ask which speaker is Jane.",
    "Claude asks which pack to install.",
    "Do you want me to install the Hungarian pack?",
    "Install the pack when the user agrees.",
    "If the recording is long ask the user before running it.",
    "The CLI uses `EXIT_DECISION` for that.",
    "| Exit | Meaning |\n|---|---|\n| 0 | done |\n| 10 | a question |\n",
    "Run `audiointake apply <id> <decision.apply_flag> <option>`.",
    "Confirm with the user before running the batch.",
    "Check with the user which language it is.",
    "Should I install the Hungarian pack?",
    "Would you like me to add speaker labels?",
    "Do you want the Thai pack as well?",
    "Let the user choose a pack, then run apply.",
    "Offer the options and let them pick one.",
    "Before transcribing, get the user's OK.",
    "Request confirmation from the user.",
    "Stop and wait until the user replies.",
    "Present the three packs as options 1, 2 and 3.",
    "Inquire whether the recording is in Spanish.",
]

ALLOWED = [
    "**A run never waits for an answer.** audiointake doesn't ask anything during a run.",
    "It works through a whole batch of recordings without stopping to ask you anything.",
    "The skill never asks the user anything during a batch.",
    "Never ask anything before or during a batch.",
    "Ask questions in [Discussions](https://github.com/RazorII/audiointake/discussions).",
    "A pack is downloaded only when you ask for it.",
    "If the user later asks for the speakers, run apply.",
    "Say 'just transcribe' and Claude won't read it.",
    "| Exit | Meaning |\n|---|---|\n| 0 | done |\n| 129 / 143 | stopped by a signal |\n",
    "With several records, the process exits with the most important code among them, in the order 2, 3, 1, 4, 5, 0.",
    "Claude Code asks before each new command it runs.",
    "macOS shows a privacy window for it.",
    "Nothing in a record waits for an answer.",
    "the recording is in a language whose pack isn't installed: asked for with `--lang`, or detected.",
    "So don't ask questions, don't offer a menu of choices, and don't stop to check \"should I…\".",
    "Never confirm with the user before a batch.",
    "Stop and check with the person who maintains it.",
    "The user's own words decide; a guess waits until the user says the name.",
    "Before transcribing, never get the user's OK: the batch runs.",
]


@pytest.mark.parametrize("text", OLD_FLOWS)
def test_the_scanner_catches_a_run_that_asks(text: str):
    assert question_flow_problems(text)


@pytest.mark.parametrize("text", ALLOWED)
def test_the_scanner_lets_through_what_says_it_does_not_ask(text: str):
    assert question_flow_problems(text) == []


def test_the_scanner_names_the_line():
    problems = question_flow_problems("# Title\n\nFine.\n\nUse AskUserQuestion.\n")
    assert problems == ["line 5: AskUserQuestion: Use AskUserQuestion."]


def test_the_readme_quotes_the_same_hungarian_figures_as_the_warning():
    """errors.PACK_WER is what the detected-Hungarian warning quotes; the README's table must say the same."""
    core, pack = errors.PACK_WER["hu"]
    row = next(ln for ln in _read(README).splitlines() if ln.startswith("| Word error rate, FLEURS Hungarian"))
    want = f"{errors.format_number('core_wer', core)} % / {errors.format_number('pack_wer', pack)} %"
    assert want in row, (want, row)
