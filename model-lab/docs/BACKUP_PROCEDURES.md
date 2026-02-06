# Backup Procedures for Model Lab

## Overview

This document outlines backup procedures for the Model Lab file storage system.

## What to Backup

### Critical Data (Daily)
| Directory | Contents | Backup Priority |
|-----------|----------|-----------------|
| `runs/` | Experiment artifacts, manifests | HIGH - Contains irreplaceable results |
| `data/golden/` | Ground truth datasets | HIGH - Used for model evaluation |
| `models/*/claims.yaml` | Model evaluation claims | MEDIUM - Version controlled |

### Configuration (Weekly)
| File | Purpose |
|------|---------|
| `.env` files | Environment configuration |
| `pyproject.toml` | Dependencies |
| `docs/PROJECT_RULES.md` | Project conventions |

### Excluded from Backup
- `.venv/` - Recreated via `uv sync`
- `__pycache__/` - Regenerated automatically
- `.cache/` - Model weights (re-downloaded)

## Backup Commands

### Local Backup
```bash
# Create dated backup of runs directory
BACKUP_DIR="/backup/model-lab/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"
rsync -av --exclude='*.log' runs/ "$BACKUP_DIR/runs/"
rsync -av data/golden/ "$BACKUP_DIR/golden/"
```

### S3 Backup (if configured)
```bash
# Sync to S3 bucket
aws s3 sync runs/ s3://model-lab-backups/runs/ \
    --exclude "*.log" \
    --storage-class STANDARD_IA

aws s3 sync data/golden/ s3://model-lab-backups/golden/
```

### Docker Volume Backup
```bash
# If running in Docker, backup the volume
docker run --rm \
    -v model-lab-data:/source:ro \
    -v /backup:/backup \
    alpine tar czf /backup/model-lab-$(date +%Y%m%d).tar.gz -C /source .
```

## Restore Procedures

### From Local Backup
```bash
# Restore runs
rsync -av /backup/model-lab/YYYYMMDD/runs/ runs/

# Restore golden data
rsync -av /backup/model-lab/YYYYMMDD/golden/ data/golden/
```

### From S3
```bash
aws s3 sync s3://model-lab-backups/runs/ runs/
aws s3 sync s3://model-lab-backups/golden/ data/golden/
```

## Verification

After restore, verify integrity:
```bash
# Check manifest files are valid JSON
find runs/ -name "manifest.json" -exec python3 -c "import json; json.load(open('{}'))" \;

# Run basic tests
uv run pytest tests/unit/ -q
```

## Retention Policy

| Backup Type | Retention |
|-------------|-----------|
| Daily backups | 7 days |
| Weekly backups | 4 weeks |
| Monthly backups | 12 months |

## Automation (Cron Example)

```bash
# Add to crontab -e
# Daily at 2 AM
0 2 * * * /path/to/model-lab/scripts/backup.sh daily

# Weekly on Sunday at 3 AM
0 3 * * 0 /path/to/model-lab/scripts/backup.sh weekly
```
