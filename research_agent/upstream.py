"""Access to A-RAG code that lives in ``scripts/`` (not importable as a package)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` once and cache it in ``sys.modules`` as ``arag_scripts.<name>``."""
    module_name = f"arag_scripts.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load upstream script {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_env() -> None:
    """Load ``.env`` from the repo root without overriding variables already set."""
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
