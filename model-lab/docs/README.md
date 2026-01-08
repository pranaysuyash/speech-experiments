# Model Testing Lab

Clean, repeatable model testing harness for systematic AI model evaluation.

## Structure

```
model-lab/
├── env/                    # Environment specifications
├── harness/               # Shared testing instrumentation
├── notebooks/             # Experiment logs (one per model)
├── data/                  # Test data organized by modality
└── results/               # Experiment outputs (generated)
```

## Quick Start

1. **Activate environment**:
   ```bash
   cd model-lab
   source .venv/bin/activate
   ```

2. **Launch Jupyter**:
   ```bash
   jupyter lab
   ```

3. **Run experiments**: Open notebook for your target model

## Principles

- **Notebook = Experiment Log**: Each notebook is a timestamped experiment record
- **Harness = Instrumentation**: All measurement logic lives in reusable modules
- **No Mixed Concerns**: Never mix model logic with measurement logic
- **Version Everything**: All prompts, configs, and results are versioned

## Adding a New Model

1. Create notebook in appropriate modality folder
2. Import harness modules
3. Configure model-specific parameters
4. Run systematic tests
5. Export results for comparison

## Current Models

- **LFM-2.5-Audio**: Liquid AI audio model (see `notebooks/audio/lfm2_5_audio.ipynb`)