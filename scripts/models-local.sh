#!/usr/bin/env bash
# Run the real-model tests on this Mac, and pass only when every one of them ran and passed.
#
# GitHub's macOS runners may have no Metal device, so the release runs this on the release Mac before it tags:
# `pytest -m models` (tests/models/) against a data folder where `audiointake setup` has installed every pack,
# with the pinned public test audio fetched. It refuses a run that collected no test, any skip, any failure or error,
# and a run that didn't run every models test: the count is taken first from an unfiltered collection, so extra
# arguments may change how pytest reports (-v, -x) but never how many tests the gate needs.
# The suite includes the runs with the network denied (sandbox-exec) and the privacy check with real children.
#
# usage: scripts/models-local.sh [extra pytest arguments]
# Environment:
#   AUDIOINTAKE_TEST_HOME   the set-up data folder (required; every pack installed there)
#   AUDIOINTAKE_TEST_AUDIO  the pinned test audio (default ~/.audiointake-dev/audio; python3 tools/fetch_test_audio.py)
#   PYTHON                  the interpreter with audiointake[test,bench] installed (default python3)
# The peak-memory and error-rate figures the tests measure are printed at the end, one JSON line each (and kept in
# $AUDIOINTAKE_MODELS_REPORT when it names a file).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "the models gate runs on a Mac with Apple silicon (this is $(uname -s) $(uname -m))" >&2
  exit 1
fi
if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = 1 ]; then
  echo "the models gate can't run under Rosetta: use an Apple-silicon Python" >&2
  exit 1
fi
if [ -z "${AUDIOINTAKE_TEST_HOME:-}" ]; then
  echo "set AUDIOINTAKE_TEST_HOME to a data folder where 'audiointake setup' has installed every pack" >&2
  exit 1
fi

py="${PYTHON:-python3}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
export AUDIOINTAKE_MODELS_REPORT="${AUDIOINTAKE_MODELS_REPORT:-$work/models-report.jsonl}"

"$py" tools/fetch_test_audio.py --check >/dev/null || {
  echo "the pinned test audio is missing or wrong: run $py tools/fetch_test_audio.py" >&2
  exit 1
}

# How many models tests exist, counted without the caller's arguments (a -k, -m, --deselect or node id would only
# narrow the run, and deselected tests aren't skips).
expected="$(PYTHONDONTWRITEBYTECODE=1 "$py" -m pytest -m models -p no:cacheprovider --collect-only -q tests/models 2>/dev/null \
  | "$py" -c 'import re, sys
m = re.search(r"(\d+)(?:/\d+)? tests? (?:collected|selected)", sys.stdin.read())
print(m.group(1) if m else 0)')"

rc=0
PYTHONDONTWRITEBYTECODE=1 "$py" -m pytest -m models -p no:cacheprovider -q -rs --junitxml "$work/models.xml" \
  tests/models "$@" || rc=$?

"$py" - "$work/models.xml" "$rc" "$expected" <<'PY'
import sys
import xml.etree.ElementTree as ET

path, rc, expected = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
try:
    root = ET.parse(path).getroot()
except (OSError, ET.ParseError):
    print(f"the models gate failed: pytest wrote no report (exit {rc})", file=sys.stderr)
    sys.exit(1)
suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
count = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "skipped", "failures", "errors")}
ran = count["tests"] - count["skipped"]
problems = []
if count["tests"] == 0 or rc == 5:
    problems.append("no models test was collected")
if expected == 0:
    problems.append("the unfiltered collection found no models test")
elif count["tests"] != expected:
    problems.append(f"{count['tests']} of the {expected} models tests ran (the gate needs all of them)")
if count["skipped"]:
    problems.append(f"{count['skipped']} skipped")
if count["failures"] or count["errors"]:
    problems.append(f"{count['failures']} failed, {count['errors']} errors")
if rc not in (0, 5) and not problems:
    problems.append(f"pytest exited {rc}")
if problems:
    print("the models gate failed: " + "; ".join(problems), file=sys.stderr)
    sys.exit(1)
print(f"the models gate passed: {ran} models tests ran, none skipped")
PY
[ -f "$AUDIOINTAKE_MODELS_REPORT" ] && cat "$AUDIOINTAKE_MODELS_REPORT"
