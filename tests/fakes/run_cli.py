"""Run the real ``audiointake`` CLI in a child process with the fakes in place (for the signal tests).

``python -I run_cli.py CONFIG.json ARGS...``: CONFIG names the fake setup's root and its keyword arguments
(``{"root": ..., "setup": {...}}``); the environment carries the scratch ``HOME``, ``AUDIOINTAKE_HOME`` and
``AUDIOINTAKE_CACHE``. Then ``cli.main(ARGS)`` runs exactly as the installed command would.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    from audiointake import cli, pipeline

    from fakes import stub_setup, stub_steps
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    fs = stub_setup.FakeSetup(Path(cfg["root"]), **cfg.get("setup", {}))
    pipeline.setup_api = fs.namespace
    pipeline.engine = stub_steps.namespace
    cli.CAFFEINATE = Path(cfg["root"]) / "no-caffeinate"
    return cli.main(sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
