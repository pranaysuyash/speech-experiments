"""Integration tests for /api/pipelines/* endpoints."""

from fastapi.testclient import TestClient
from server.main import app

client = TestClient(app)


class TestPipelinesStepsEndpoint:
    def test_list_steps_returns_all_steps(self):
        response = client.get("/api/pipelines/steps")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        step_names = [s["name"] for s in data]
        assert "ingest" in step_names
        assert "asr" in step_names
        assert "diarization" in step_names
        assert "alignment" in step_names

    def test_step_has_required_fields(self):
        response = client.get("/api/pipelines/steps")
        data = response.json()
        for step in data:
            assert "name" in step
            assert "deps" in step
            assert "description" in step


class TestPipelinesPreprocessingEndpoint:
    def test_list_preprocessing_ops(self):
        response = client.get("/api/pipelines/preprocessing")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        op_names = [op["name"] for op in data]
        assert "trim_silence" in op_names
        assert "normalize_loudness" in op_names

    def test_preprocessing_has_params(self):
        response = client.get("/api/pipelines/preprocessing")
        data = response.json()
        for op in data:
            assert "name" in op
            assert "description" in op
            assert "params" in op


class TestPipelinesTemplatesEndpoint:
    def test_list_templates(self):
        response = client.get("/api/pipelines/templates")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        template_names = [t["name"] for t in data]
        assert "fast_asr" in template_names
        assert "full_meeting" in template_names

    def test_get_template_detail(self):
        response = client.get("/api/pipelines/templates/fast_asr")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "fast_asr"
        assert "resolved_steps" in data
        assert "ingest" in data["resolved_steps"]

    def test_get_unknown_template_returns_404(self):
        response = client.get("/api/pipelines/templates/nonexistent")
        assert response.status_code == 404


class TestPipelinesValidateEndpoint:
    def test_validate_valid_config(self):
        response = client.post("/api/pipelines/validate", json={
            "steps": ["ingest", "asr"],
            "preprocessing": ["trim_silence"]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["errors"] == []
        assert "ingest" in data["resolved_steps"]

    def test_validate_invalid_step(self):
        response = client.post("/api/pipelines/validate", json={
            "steps": ["ingest", "invalid_step"]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0

    def test_validate_empty_steps(self):
        response = client.post("/api/pipelines/validate", json={
            "steps": []
        })
        data = response.json()
        assert data["valid"] is False


class TestPipelinesResolveEndpoint:
    def test_resolve_adds_dependencies(self):
        response = client.post("/api/pipelines/resolve", json={
            "steps": ["alignment"]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["requested_steps"] == ["alignment"]
        assert "ingest" in data["resolved_steps"]
        assert "asr" in data["resolved_steps"]
        assert "diarization" in data["resolved_steps"]
        assert "alignment" in data["resolved_steps"]
        assert "ingest" in data["added_dependencies"]

    def test_resolve_preserves_order(self):
        response = client.post("/api/pipelines/resolve", json={
            "steps": ["chapters"]
        })
        data = response.json()
        resolved = data["resolved_steps"]
        assert resolved.index("ingest") < resolved.index("asr")
        assert resolved.index("alignment") < resolved.index("chapters")


class TestPipelinesStepInfoEndpoint:
    def test_get_step_info(self):
        response = client.get("/api/pipelines/step/asr")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "asr"
        assert "deps" in data
        assert "dependents" in data
        assert "description" in data

    def test_get_unknown_step_returns_404(self):
        response = client.get("/api/pipelines/step/nonexistent")
        assert response.status_code == 404


class TestPipelinesPreprocessingInfoEndpoint:
    def test_get_preprocessing_info(self):
        response = client.get("/api/pipelines/preprocessing/trim_silence")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "trim_silence"
        assert "params" in data

    def test_get_unknown_preprocessing_returns_404(self):
        response = client.get("/api/pipelines/preprocessing/nonexistent")
        assert response.status_code == 404


class TestUserTemplatesEndpoints:
    """Tests for user-defined pipeline templates API."""
    
    def test_list_user_templates_empty(self):
        response = client.get("/api/pipelines/user-templates")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_create_user_template(self):
        import uuid
        unique_name = f"test_template_{uuid.uuid4().hex[:8]}"
        response = client.post("/api/pipelines/user-templates", json={
            "name": unique_name,
            "steps": ["ingest", "asr"],
            "preprocessing": ["trim_silence"],
            "description": "Test template for unit tests"
        })
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == unique_name
        assert data["created"] is True
    
    def test_get_user_template(self):
        # First create it
        client.post("/api/pipelines/user-templates", json={
            "name": "test_get_template",
            "steps": ["ingest", "diarization"],
        })
        
        response = client.get("/api/pipelines/user-templates/test_get_template")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_get_template"
        assert data["steps"] == ["ingest", "diarization"]
        assert "resolved_steps" in data
    
    def test_update_user_template(self):
        # Create
        client.post("/api/pipelines/user-templates", json={
            "name": "test_update_template",
            "steps": ["ingest", "asr"],
        })
        
        # Update
        response = client.post("/api/pipelines/user-templates", json={
            "name": "test_update_template",
            "steps": ["ingest", "asr", "diarization"],
        })
        assert response.status_code == 200
        data = response.json()
        assert data["updated"] is True
    
    def test_delete_user_template(self):
        # Create
        client.post("/api/pipelines/user-templates", json={
            "name": "test_delete_template",
            "steps": ["ingest"],
        })
        
        # Delete
        response = client.delete("/api/pipelines/user-templates/test_delete_template")
        assert response.status_code == 200
        assert response.json()["deleted"] == "test_delete_template"
        
        # Verify deleted
        response = client.get("/api/pipelines/user-templates/test_delete_template")
        assert response.status_code == 404
    
    def test_create_template_invalid_steps(self):
        response = client.post("/api/pipelines/user-templates", json={
            "name": "invalid_template",
            "steps": ["ingest", "nonexistent_step"],
        })
        assert response.status_code == 400
    
    def test_get_unknown_template_returns_404(self):
        response = client.get("/api/pipelines/user-templates/nonexistent")
        assert response.status_code == 404

