from unittest.mock import patch

from fastapi.testclient import TestClient

from server.main import app


class TestLifecycleAPI:
    def test_kill_endpoint_idempotence(self):
        client = TestClient(app)
        run_id = "test_run_kill"

        # 1. First Kill (Success)
        with patch("server.api.lifecycle.kill_run") as mock_kill:
            mock_kill.return_value = (True, "killed")

            resp = client.post(f"/api/runs/{run_id}/kill")
            assert resp.status_code == 200
            assert resp.json() == {"status": "cancelled", "outcome": "killed"}

        # 2. Second Kill (Already Stopped) -> Should still be 200
        with patch("server.api.lifecycle.kill_run") as mock_kill:
            # Service layer returns True for "already_terminal" or "already_dead"
            mock_kill.return_value = (True, "already_terminal")

            resp = client.post(f"/api/runs/{run_id}/kill")
            assert resp.status_code == 200
            assert resp.json() == {"status": "cancelled", "outcome": "already_terminal"}

        # 3. Third Kill (Forced Cancel / Zombie) -> Should be 200
        with patch("server.api.lifecycle.kill_run") as mock_kill:
            mock_kill.return_value = (True, "forced_cancel")

            resp = client.post(f"/api/runs/{run_id}/kill")
            assert resp.status_code == 200
            assert resp.json() == {"status": "cancelled", "outcome": "forced_cancel"}

    def test_kill_not_found(self):
        client = TestClient(app)
        with patch("server.api.lifecycle.kill_run") as mock_kill:
            mock_kill.return_value = (False, "not_found")

            resp = client.post("/api/runs/missing/kill")
            assert resp.status_code == 404
