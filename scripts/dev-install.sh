#!/usr/bin/env bash
# Install audiointake for development or CI.
#
# usage: scripts/dev-install.sh [--python PY] [--extras LIST] [--tool]
#   (default)  an editable install with the [test] extra into the virtual environment at
#              $VIRTUAL_ENV, made with `uv venv` if it isn't there yet
#   --tool     `uv tool install --force --python 3.14 .`: a non-editable install, the way a user
#              gets it; VIRTUAL_ENV isn't needed
#   --python   the Python uv installs for (default 3.14, the measured version; 3.12 to 3.14 work)
#   --extras   the extras of the editable install (default test; empty for none; test,bench adds
#              what the public bench needs)
#
# VIRTUAL_ENV must be outside this source folder: the environment holds about 1.2 GB, and a source
# folder inside a synced folder (iCloud Drive, Dropbox) would upload it. For example:
#   uv venv --python 3.14 ~/.audiointake-dev/venv
#   VIRTUAL_ENV=~/.audiointake-dev/venv scripts/dev-install.sh
# Environment: UV is the uv command (default uv). Inside GitHub Actions the install's bin folder is
# added to GITHUB_PATH, so later steps find `python` and `audiointake`.
set -euo pipefail

here="$PWD"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="3.14"; extras="test"; tool=0
while [ $# -gt 0 ]; do
  case "$1" in
    --python) py="${2:?--python needs a value}"; shift 2 ;;
    --extras) [ $# -ge 2 ] || { echo "dev-install.sh: --extras needs a value" >&2; exit 2; }; extras="$2"; shift 2 ;;
    --tool) tool=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "dev-install.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

uv="${UV:-uv}"
command -v "$uv" >/dev/null 2>&1 || { echo "dev-install.sh needs uv (https://docs.astral.sh/uv/)" >&2; exit 3; }
cd "$root"

if [ "$tool" = 1 ]; then
  "$uv" tool install --force --python "$py" .
  if [ -n "${GITHUB_PATH:-}" ]; then "$uv" tool dir --bin >> "$GITHUB_PATH"; fi
  exit 0
fi

venv="${VIRTUAL_ENV:-}"
[ -n "$venv" ] || {
  echo "dev-install.sh: set VIRTUAL_ENV to the environment's folder, outside this source folder" \
       "(for example VIRTUAL_ENV=~/.audiointake-dev/venv scripts/dev-install.sh)" >&2
  exit 2
}
case "$venv" in /*) ;; *) venv="$here/$venv" ;; esac
case "$venv/" in
  "$root"/*) echo "dev-install.sh: VIRTUAL_ENV ($venv) is inside the source folder; put it outside, for example ~/.audiointake-dev/venv" >&2; exit 2 ;;
esac

if [ ! -x "$venv/bin/python" ]; then
  "$uv" venv --python "$py" "$venv"
fi
spec="."
[ -z "$extras" ] || spec=".[$extras]"
"$uv" pip install --python "$venv/bin/python" -e "$spec"
if [ -n "${GITHUB_PATH:-}" ]; then echo "$venv/bin" >> "$GITHUB_PATH"; fi
echo "audiointake is installed in $venv (editable)."
