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
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
python server/main.py
```

### 2. Frontend (Port 5174)
```bash
cd client
npm run dev -- --port 5174
```
