"""Test layering boundaries to prevent architectural violations."""
import pytest
from pathlib import Path
import re


def test_server_does_not_import_harness():
    """Server layer must not import harness except for intentional integration points."""
    
    server_dir = Path(__file__).parent.parent.parent / "server"
    violations = []
    
    # Allowlist: server/api/workbench.py is the intentional execution surface
    allowlist = {
        "server/api/workbench.py",
    }
    
    for py_file in server_dir.rglob("*.py"):
        if py_file.name.startswith("__"):
            continue
            
        rel_path = str(py_file.relative_to(server_dir.parent))
        
        # Skip allowlisted files
        if rel_path in allowlist:
            continue
            
        content = py_file.read_text(encoding="utf-8")
        
        # Check for harness imports
        if re.search(r'^\s*(from harness|import harness)', content, re.MULTILINE):
            violations.append(rel_path)
    
    if violations:
        msg = (
            "Server layer must not import harness (except allowlisted integration points).\n"
            f"Violations found in:\n" + "\n".join(f"  - {v}" for v in violations)
        )
        pytest.fail(msg)
