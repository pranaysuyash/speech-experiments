# Requests

**Status**: Open Signal Gathering
**Purpose**: Collect builder feedback on models, formats, and use cases.

## Disclaimers
1. **Signal Only**: Submitting a request creates a signal for maintainers. It does NOT guarantee implementation.
2. **No Timeline**: There are no SLAs or ETAs for requested features.
3. **Public Record**: Requests may be made public or discussed openly.

## How to Request

### 1. New Model Support
Create a GitHub Issue with the tag `[Request]`. Include:
- **Model Name**: (e.g. `whisper-v3-large`)
- **License**: (Must be permissible for public lab use)
- **HuggingFace Link**: URL to weights.
- **Why**: What specific benchmark or use case does this unlock?

### 2. New Output Format
- **Format**: (e.g. `Time-aligned JSON`, `SRT`, `TextGrid`)
- **Consumer**: What tool will ingest this output?

### 3. Use Case / Dataset
- **Description**: "I want to test X on Y type of audio."
- **Why**: Helps us prioritize the "Torture Test" corpus.

## Prioritization
Maintainers prioritize based on:
1. **Safety/Security**: Does it introduce risk?
2. **Generality**: Does it help many builders?
3. **Ease of Integration**: Is it a drop-in addition?
