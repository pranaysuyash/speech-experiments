from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import logging.config

from server.api import runs, results, workbench, experiments, candidates

# Logging configuration - silence noisy loggers
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(levelname)s [%(name)s] %(message)s",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        # Silence noisy loggers
        "watchfiles": {"level": "WARNING"},
        "watchfiles.main": {"level": "WARNING"},
        "server.index": {"level": "WARNING"},
        "uvicorn.access": {"level": "WARNING"},
        "multipart": {"level": "WARNING"},
        # Keep errors visible
        "server": {"level": "INFO"},
        "harness": {"level": "INFO"},
    },
    "root": {
        "handlers": ["default"],
        "level": "INFO",
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
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
        log_config=LOGGING_CONFIG,  # Pass config to uvicorn
        access_log=False  # Disable access logs entirely
    )

