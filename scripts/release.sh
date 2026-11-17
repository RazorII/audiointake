#!/usr/bin/env bash
# Tag an audiointake release, push it, publish it as an immutable GitHub release, and prove it.
#
# The plugin's skill installs audiointake from the release's source archive, pinned to its tag, so
# a tag that never reached GitHub breaks every new install. `git push --follow-tags` skips
# lightweight tags, so a tag can stay local without anyone noticing.
#
# Before anything is tagged this script refuses unless:
#   - the version is X.Y.Z and src/audiointake/__init__.py's __version__ line says it;
#   - tests/test_release.py passes, the tree is clean, and this checkout is on main;
#   - every commit that isn't on GitHub's main yet, and the name and email git will put on the tag,
#     are RazorII's, with RazorII's GitHub noreply address as both author and committer email. What
#     GitHub's main already has stays as it is: a contributor's merged pull request keeps its own
#     names, under the merge commit GitHub wrote. This checkout must contain GitHub's main;
#   - Gate 1: the repo-level tests pass in strict mode with nothing skipped (a missing document or a
#     left-over TBD marker fails instead of skipping);
#   - Gate 2: scripts/models-local.sh passes on this Mac: the real models end to end, nothing
#     skipped, including the run with the network denied;
#   - Gate 3: GitHub's immutable releases are on for the repo, so a published release and its tag
#     can never change.
# Then it makes the annotated tag (a rerun accepts this release's own annotated tag at HEAD), checks
# the tagger like the commits, pushes the branch and the tag in one --atomic push, checks the tag is
# on GitHub, and publishes the release (no assets: the archive GitHub makes from the tag is what
# users install). A published release is never edited to change what it ships: a mistake is fixed
# by the next patch version.
#
# usage: scripts/release.sh X.Y.Z
# Environment: PYTHON (default python3) has audiointake[test] installed.
set -euo pipefail
v="${1:?usage: scripts/release.sh X.Y.Z}"
[[ "$v" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] || { echo "not a version: $v" >&2; exit 1; }
cd "$(git rev-parse --show-toplevel)"
grep -qE "^__version__ = \"${v//./\\.}\"([[:space:]]|\$)" src/audiointake/__init__.py \
  || { echo "src/audiointake/__init__.py is not at $v" >&2; exit 1; }
slug="RazorII/audiointake"
tag="v$v"
publisher="RazorII"
noreply="1577315+RazorII@users.noreply.github.com"
py="${PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

"$py" -m pytest -q -p no:cacheprovider tests/test_release.py
[ -z "$(git status --porcelain)" ] || { echo "commit everything first" >&2; exit 1; }
# A release is cut from main: pushing HEAD from another branch would publish that branch and tag a commit
# main doesn't have.
branch="$(git symbolic-ref --short -q HEAD || true)"
[ "$branch" = main ] || { echo "release from main: this checkout is on ${branch:-a detached HEAD}" >&2; exit 1; }

# audiointake is published as RazorII. A fresh clone takes the global git identity, so check every
# commit GitHub doesn't have yet, and the identity the tag will carry, before anything is pushed.
cfg_name="$(git config user.name || true)"
cfg_email="$(git config user.email || true)"
[ -n "$cfg_name" ] && [ -n "$cfg_email" ] || { echo "git config user.name and user.email must be set in this checkout ($publisher, $noreply); found: '${cfg_name}' '${cfg_email}'" >&2; exit 1; }
# ls-remote --exit-code: 0 means GitHub has a main, 2 means it has none; anything else is a failure to ask,
# which must never pass for "no main" (that would check a different set of commits).
range=HEAD
rc=0
git ls-remote --exit-code --heads origin main > /dev/null 2> "$work/ls-remote.err" || rc=$?
case "$rc" in
  0)
    git fetch --quiet origin main
    git merge-base --is-ancestor FETCH_HEAD HEAD || { echo "GitHub's main has commits this checkout lacks (a merged pull request?): git pull --ff-only origin main first" >&2; exit 1; }
    range=FETCH_HEAD..HEAD ;;
  2) ;;
  *) echo "couldn't ask GitHub for its main branch (git ls-remote exit $rc): $(head -n 1 "$work/ls-remote.err")" >&2; exit 1 ;;
esac
names="$( { git log --format='%an%n%cn' "$range"; echo "$cfg_name"; } | sort -u)"
[ "$names" = "$publisher" ] || { echo "commits and the tag must be by $publisher (git config user.name $publisher); found: $(echo $names)" >&2; exit 1; }
emails="$( { git log --format='%ae%n%ce' "$range"; echo "$cfg_email"; } | sort -u)"
[ "$emails" = "$noreply" ] || { echo "commits and the tag must carry $noreply (git config user.email $noreply); found: $(echo $emails)" >&2; exit 1; }

cat > "$work/helper.py" <<'PY'
"""release.sh's helper: read pytest's report and GitHub's answers."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SLUG = "RazorII/audiointake"


def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


def no_skips(report, rc):
    root = ET.parse(report).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    count = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "skipped", "failures", "errors")}
    if int(rc) != 0 or count["skipped"] or count["failures"] or count["errors"] or not count["tests"]:
        die(f"the repo-level tests must all run and pass: {count['tests']} tests, {count['skipped']} skipped, "
            f"{count['failures']} failed, {count['errors']} errors (pytest exit {rc})")


def gh_answer(path, err, rc, what):
    """The JSON gh api printed, or None for a 404. Any other failure (auth, network, rate limit)
    stops here with gh's own words, so a setting that is on is never reported as off."""
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        body = None
    try:
        said = [ln.strip() for ln in Path(err).read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    except OSError:
        said = []
    if int(rc) == 0:
        if body is None:
            die(f"couldn't read {what} for {SLUG}: GitHub's answer isn't JSON")
        return body
    if any("HTTP 404" in ln for ln in said) or (
            isinstance(body, dict) and (str(body.get("status")) == "404" or body.get("message") == "Not Found")):
        return None
    die(f"couldn't read {what} for {SLUG}: " + (said[0] if said else f"gh api exited {rc}"))


def immutable(path, err, rc):
    """The endpoint answers 404 while immutable releases are off, and 200 {enabled: false} too."""
    answer = gh_answer(path, err, rc, "the immutable-releases setting")
    if not (isinstance(answer, dict) and answer.get("enabled") is True):
        die(f"immutable releases are off for {SLUG}: turn them on in the repo's settings "
            "(Settings, General, Releases) before tagging")


def release(path, err, rc, tag):
    """missing, or published: this tag's release, not a draft, immutable."""
    answer = gh_answer(path, err, rc, f"the {tag} release")
    if answer is None:
        print("missing")
        return
    if answer.get("tag_name") != tag:
        die(f"GitHub answered with the release for {answer.get('tag_name')!r}, not {tag}")
    if answer.get("draft"):
        die(f"the {tag} release is a draft: publish or delete it on GitHub, then run this again")
    if answer.get("immutable") is not True:
        die(f"the {tag} release isn't immutable")
    print("published")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"no-skips": no_skips, "immutable": immutable, "release": release}[cmd](*args)
PY
helper() { "$py" "$work/helper.py" "$@"; }

# Gate 1: the repo-level tests, strict. A missing document or a left-over TBD marker fails instead of skipping.
rc=0
AUDIOINTAKE_RELEASE_STRICT=1 "$py" -m pytest -q -p no:cacheprovider -rs --junitxml "$work/repo.xml" \
  tests/test_release.py tests/test_docs.py tests/test_plugin_folder.py tests/test_workflows.py || rc=$?
[ -f "$work/repo.xml" ] || { echo "the repo-level tests didn't run (pytest exit $rc)" >&2; exit 1; }
helper no-skips "$work/repo.xml" "$rc"

# Gate 2: the real models on this Mac, nothing skipped, the network-denied run included.
[ -x scripts/models-local.sh ] || { echo "scripts/models-local.sh is missing or not executable" >&2; exit 1; }
PYTHON="$py" scripts/models-local.sh

# Gate 3: immutable releases on. The endpoint answers 404 while they're off; any other gh failure
# is reported as itself.
rc=0
gh api "repos/$slug/immutable-releases" > "$work/immutable.json" 2> "$work/immutable.err" || rc=$?
helper immutable "$work/immutable.json" "$work/immutable.err" "$rc"

# The tag. A rerun after a failed push finds its own annotated tag already at HEAD and pushes it
# again. A tag anywhere else, or a lightweight one, stops here, and git tag -a never replaces one.
tag_type="$(git cat-file -t "refs/tags/$tag" 2>/dev/null || true)"
if [ -n "$tag_type" ]; then
  if [ "$tag_type" != tag ] || [ "$(git rev-parse "refs/tags/$tag^{commit}")" != "$(git rev-parse HEAD)" ]; then
    echo "fatal: tag '$tag' already exists and isn't this release's annotated tag at HEAD; it is never replaced" >&2
    exit 1
  fi
else
  git tag -a "$tag" -m "audiointake $v"
fi
tagger="$(git for-each-ref --format='%(taggername)|%(taggeremail)' "refs/tags/$tag")"
tagger_name="${tagger%%|*}"; tagger_email="${tagger#*|}"; tagger_email="${tagger_email#<}"; tagger_email="${tagger_email%>}"
if [ "$tagger_name" != "$publisher" ] || [ "$tagger_email" != "$noreply" ]; then
  echo "the tag $tag must be by $publisher <$noreply>; it is by $tagger_name <$tagger_email>. Delete it (git tag -d $tag), fix the identity, and run this again" >&2
  exit 1
fi

# --atomic: main and the tag land together or not at all, so a refused tag can't leave main published
# without its release (installs pin to the tag).
git push --atomic origin HEAD "refs/tags/$tag"
git ls-remote --exit-code --tags origin "refs/tags/$tag" >/dev/null || { echo "$tag did not reach GitHub" >&2; exit 1; }
echo "$tag is on GitHub"

# The release. A rerun finds it already published and only checks it.
rc=0
gh api "repos/$slug/releases/tags/$tag" > "$work/release.json" 2> "$work/release.err" || rc=$?
state="$(helper release "$work/release.json" "$work/release.err" "$rc" "$tag")"
if [ "$state" = missing ]; then
  gh release create "$tag" --repo "$slug" --verify-tag --title "audiointake $v" \
    --notes "audiointake $v. Release notes follow."
  # GitHub's release listing can lag the create by a few seconds, so look again before giving up.
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    rc=0
    gh api "repos/$slug/releases/tags/$tag" > "$work/release.json" 2> "$work/release.err" || rc=$?
    state="$(helper release "$work/release.json" "$work/release.err" "$rc" "$tag")"
    [ "$state" = missing ] || break
    sleep 3
  done
  [ "$state" = published ] || { echo "the $tag release didn't appear on GitHub" >&2; exit 1; }
fi
echo "$tag is published as an immutable release"
