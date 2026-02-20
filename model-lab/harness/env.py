"""
Small helpers for environment loading.

We keep this minimal and optional so scripts can run even if dotenv isn't installed,
but when it is installed it prevents a common footgun: CLI runs not inheriting
HF_TOKEN/HUGGINGFACE_HUB_TOKEN because the shell didn't `source .env`.
"""

from __future__ import annotations

from pathlib import Path


def load_dotenv_if_present(dotenv_path: str | Path = ".env") -> bool:
    """
    Load a .env file into process environment variables if python-dotenv is available.

    Returns True if a dotenv file was found and loaded, else False.
    """
    path = Path(dotenv_path)
    if not path.exists():
        return False

    try:
        from dotenv import load_dotenv
    except Exception:
        return False

    # Don't override values already provided by the shell/CI.
    load_dotenv(dotenv_path=path, override=False)
    return True

