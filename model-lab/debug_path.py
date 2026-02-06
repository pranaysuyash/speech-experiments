import sys
from pathlib import Path


def debug_find(run_id):
    runs_root = Path("runs").resolve() / "sessions"
    print(f"Searching in: {runs_root}")

    found_run_dir = None
    if runs_root.exists():
        # Search for the run_id directory
        # logic from experiments.py
        for path in runs_root.glob(f"*/{run_id}"):
            print(f"Checking {path}")
            if path.is_dir():
                found_run_dir = path
                print(f"FOUND: {found_run_dir}")
                break

    if not found_run_dir:
        print("NOT FOUND via glob")
        return

    # Check for transcript
    # logic from artifacts.py
    candidates = [
        "bundle/transcript.json",
        "bundle/transcript.txt",
        "artifacts/transcript.txt",
        "asr/transcript.txt",
    ]

    for rel in candidates:
        p = found_run_dir / rel
        print(f"Checking candidate: {p} -> exists? {p.exists()}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        debug_find(sys.argv[1])
    else:
        print("Usage: python3 debug_path.py <run_id>")
