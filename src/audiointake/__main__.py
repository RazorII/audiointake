"""``python -m audiointake``: the same entry point as the ``audiointake`` command."""

if __name__ == "__main__":
    from .cli import main          # inside the guard, so importing this module never needs the CLI

    raise SystemExit(main())
