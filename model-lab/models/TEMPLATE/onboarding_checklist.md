# Model Onboarding Checklist

Model: _______________
Date: _______________
Owner: _______________

## Required (blocks merge)

- [ ] `config.yaml` exists in `models/<model_id>/`
- [ ] Loader function registered in `harness/registry.py`
- [ ] Bundle Contract v1 satisfied (returns required keys)
- [ ] One smoke run executed successfully
- [ ] `runs/<model_id>/summary.json` saved
- [ ] Model card generated (`runs/<model_id>/model_card.json`)

## Validation

- [ ] Import test passes: `make test-imports`
- [ ] Model info shows correct metadata: `make model-info MODEL=<model_id>`
- [ ] ASR test runs without crash: `make asr MODEL=<model_id> DATASET=llm_primary`

## Documentation

- [ ] `README.md` in model directory
- [ ] Known issues documented
- [ ] Best use cases documented
- [ ] `docs/use_cases.md` updated if new capability

## Quality Gates

- [ ] WER within expected range (if ASR)
- [ ] No truncation/hallucination flags
- [ ] No TTS audio issues (if TTS)

## Sign-off

- [ ] Ready for CANDIDATE status
- [ ] CI passes with new model
