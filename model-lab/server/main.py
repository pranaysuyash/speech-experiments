from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

from server.api import runs, results, workbench, experiments, candidates

# Configure logging - reduce noise
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s [%(name)s] %(message)s'
)

# Silence verbose loggers
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # Hide HTTP access logs
logging.getLogger("uvicorn.error").setLevel(logging.INFO)
logging.getLogger("multipart").setLevel(logging.WARNING)  # Hide multipart parsing logs
logging.getLogger("watchfiles").setLevel(logging.WARNING)  # Hide file change detection
logging.getLogger("server.index").setLevel(logging.WARNING)  # Hide index refresh spam


logger = logging.getLogger("server")

app = FastAPI(title="Model Lab Analyst Console", version="0.1.0")

# CORS for Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(runs.router)
app.include_router(results.router)
app.include_router(workbench.router)
app.include_router(experiments.router)
app.include_router(candidates.router)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/health")
def api_health_check():
    """Health check accessible via /api/health for consistency with proxy"""
    return {"status": "ok"}

if __name__ == "__main__":
    # Local dev entry point
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=False  # Disable access logs
    )
