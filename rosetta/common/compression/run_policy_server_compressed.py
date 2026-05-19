"""Entry point that wraps lerobot.async_inference.policy_server with the
image-decompression hook installed.

Usage (drop-in replacement for `python -m lerobot.async_inference.policy_server`):

    python -m rosetta.common.compression.run_policy_server_compressed \
        --host=127.0.0.1 --port=8080
"""
from __future__ import annotations

import runpy
import sys

from .integration import wrap_policy_server


def main() -> None:
    wrap_policy_server()
    runpy.run_module(
        "lerobot.async_inference.policy_server",
        run_name="__main__",
        alter_sys=True,
    )


if __name__ == "__main__":
    sys.exit(main())
