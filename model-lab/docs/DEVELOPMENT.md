# Development Environment

## Quickstart (Recommended)

To start the full stack (backend + frontend) with automatic environment handling, use the helper script:

```bash
./dev.sh
```

### Features of `dev.sh`
*   **Port Cleanup**: Automatically kills any zombie processes on port `8000` (Backend) and `5174` (Frontend) before starting.
*   **Venv Force**: Detects `.venv` or `venv` directories and activates the Python virtual environment to prevent system python conflicts.
*   **Unified Logging**:
    *   Backend output -> `server.log` (Streamed to console by default)
    *   Frontend output -> `frontend.log` (Backgrounded)
*   **Clean Exit**: Traps `Ctrl+C` to shut down both servers simultaneously.

## Manual Setup

If you prefer to run services individually:

### 1. Backend (Port 8000)
```bash
# Activate the project venv first
source .venv/bin/activate

# Preferred: use `uv` (from venv) to run the app. Example:
uv run uvicorn server.main:app --host 127.0.0.1 --port 8000 --reload

# Or call the venv uvicorn binary directly:
.venv/bin/uvicorn server.main:app --host 127.0.0.1 --port 8000 --reload

# Quick health check:
curl http://127.0.0.1:8000/health
```

## Commit workflow

When preparing local changes, follow this simple workflow to avoid missing files:

- Stage everything: `git add -A`
- Commit: `git commit -m "<concise message>"`
- Push and open a PR

Using `git add -A` ensures new, modified and deleted files are included in the commit (handy for generated assets like `assets/demo.gif`).

### 2. Frontend (Port 5174)
```bash
cd client
npm run dev -- --port 5174
```
