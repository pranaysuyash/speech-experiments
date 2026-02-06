from typing import Any


class RunsRepo:
    @staticmethod
    def get_run_manifest(run_id: str) -> dict[str, Any] | None:
        """Get manifest for a run (placeholder for future DB)."""
        # Currently, manifests are loaded via index, but this could centralize file ops
        return None  # Implement if needed

    @staticmethod
    def save_run_result(run_id: str, result: dict[str, Any]) -> None:
        """Save result to file (placeholder)."""
        # Currently handled in harness, but centralize here
        pass

    @staticmethod
    def list_run_files(run_id: str) -> list[str]:
        """List files for a run."""
        # Placeholder for file operations
        return []
