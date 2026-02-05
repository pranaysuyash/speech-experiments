# Audio research CSV drop (2026-02-05)

**Source**: `~/Downloads/audio_*` files from 2026-02-05.

These files are treated as **research inputs** for Model-Lab:

- `audio_model_audit_expanded.csv`: canonical catalog schema (matches `model-lab/data/templates/model_catalog_schema.csv`)
- `audio_model_audit_expanded_plus.csv`: expanded catalog with additional scoring/ops columns
- `audio_model_audit_scored.csv`: scored catalog snapshot
- `audio_model_eval_harness.csv`: proposed evaluation suites (datasets + metrics)
- `audio_model_lab_integration_backlog.csv`: concrete lab integration tickets
- `audio_model_recommendations_top20_*.csv`: precomputed shortlists
- `audio_api_pricing_snapshot_template.csv`: pricing snapshot template
- `audio_model_priority_scoring_weights.txt`: scoring weights used for `priority_score`

## How Model-Lab uses these

- Default catalog file: `model-lab/data/model_catalog.csv` is currently a copy of `audio_model_audit_expanded.csv`.
- Reporter: `model-lab/scripts/report_model_catalog.py` prints a Markdown summary for quick review.

