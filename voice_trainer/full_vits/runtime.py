from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import warnings


def package_root() -> Path:
    return Path(__file__).resolve().parent


def monotonic_align_root() -> Path:
    return package_root() / "monotonic_align"


def ensure_monotonic_align_built() -> None:
    root = monotonic_align_root()
    if any(root.glob("core*.so")) or any(root.glob("core*.pyd")):
        return
    try:
        subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=root,
            check=True,
        )
    except subprocess.CalledProcessError:
        warnings.warn(
            "monotonic_align extension build failed; falling back to the pure Python implementation.",
            RuntimeWarning,
        )
