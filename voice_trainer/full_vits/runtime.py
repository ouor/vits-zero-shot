from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parent


def monotonic_align_root() -> Path:
    return package_root() / "monotonic_align"


def ensure_monotonic_align_built() -> None:
    root = monotonic_align_root()
    if any(root.glob("core*.so")) or any(root.glob("core*.pyd")):
        return
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=root,
        check=True,
    )
