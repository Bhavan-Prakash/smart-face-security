"""Compatibility entry point; identity matching is database-backed in main.py."""
from main import main


if __name__ == "__main__":
    raise SystemExit(main())
