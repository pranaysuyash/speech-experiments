# Public Lab Mode

**Status**: Default (Active)
**Concept**: Anonymous, open-access builder environment.

## 1. Access Model
- **Authentication**: None.
- **Authorization**: Public Access.
- **Identity**: Anonymous (No user accounts).

This system is designed as a **Public Lab**. Anyone with the URL can:
1. Upload audio files.
2. Configure pipelines.
3. Inspect results (artifacts, transcripts, logs).

## 2. Shared Visibility
- **All Runs Are Public**: Every run created is visible to every other user.
- **No Private Workspaces**: There is one global "Workbench".
- **Real-time collaboration**: If two builders view the same run ID, they see the same state.

## 3. Privacy & Retention
- **NO Privacy Guarantee**: Do not upload sensitive, private, or confidential audio.
- **Volatile Storage**: Runs may be deleted at any time to reclaim disk space.
- **No Ownership**: You do not "own" a run. Anyone can view or link to it.

## 4. Future / Optional Deployment Modes
*(Feature not currently active in default deployment)*

**Gated Alpha Mode**:
- Uses `MODEL_LAB_ALPHA_KEY` environment variable.
- Requires `X-Alpha-Access-Token` header.
- Designed for private team instances.
