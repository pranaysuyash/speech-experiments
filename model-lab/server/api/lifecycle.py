from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from server.services.lifecycle import RunnerBusyError, kill_run, retry_run

router = APIRouter(prefix="/api/runs", tags=["lifecycle"])


class RetryRequest(BaseModel):
    from_step: str | None = None


@router.post("/{run_id}/kill")
def kill_run_endpoint(run_id: str):
    """
    Kill a running/stale run.
    Idempotent: returns success if run is stopped (even if already dead).
    """
    success, outcome = kill_run(run_id)
    if success:
        return {"status": "cancelled", "outcome": outcome}
    elif outcome == "not_found":
        raise HTTPException(status_code=404, detail="Run not found")
    else:
        raise HTTPException(status_code=500, detail=f"Failed to kill run: {outcome}")


@router.post("/{run_id}/retry")
def retry_run_endpoint(run_id: str, req: RetryRequest = Body(default=RetryRequest())):
    """
    Retry a failed/cancelled run.
    """
    try:
        result = retry_run(run_id, from_step=req.from_step)
        return result
    except RunnerBusyError:
        raise HTTPException(status_code=409, detail="System busy")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import logging

        logging.getLogger("server.api").exception(f"Retry failed for {run_id}")
        raise HTTPException(status_code=500, detail=f"Retry failed: {str(e)}")
