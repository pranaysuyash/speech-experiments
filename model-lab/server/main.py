from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

from server.api import runs, results

# Configure logging
logger = logging.getLogger("server")

app = FastAPI(title="Model Lab Analyst Console", version="0.1.0")

# CORS for Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(runs.router)
app.include_router(results.router)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/health")
def api_health_check():
    """Health check accessible via /api/health for consistency with proxy"""
    return {"status": "ok"}

if __name__ == "__main__":
    # Local dev entry point
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
