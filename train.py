"""Backward-compatible entry point. Prefer: uv run cleandift-train"""
from cleandift.train import main

if __name__ == "__main__":
    main()
