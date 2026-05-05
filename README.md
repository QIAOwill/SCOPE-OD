# SCOPE-OD: Semantic Consistency-Oriented Prior-Enhanced OD Flow Forecasting

SCOPE-OD is a semantic-prior-enhanced framework for origin-destination (OD) flow forecasting. The public repository contains executable model code and a small synthetic placeholder dataset. The placeholder files preserve the expected schema and directory layout, but they do not contain private or production data.

## Framework Overview

The proposed **Semantic Consistency-Oriented Prior-Enhanced Origin-Destination Flow Forecasting (SCOPE-OD)** framework is organized into four progressively connected modules:

1. **Role-Aware Semantic Prior Builder (RSPB)**
   - Builds role-specific semantic priors for origin cities, destination cities, and OD-pair relations.
   - Produces discrete semantic labels, soft probability distributions, confidence scores, entropy values, and dense tensor files used by downstream training.

2. **Dynamic Semantic Conditioning Module (DSCM)**
   - Converts semantic priors and time-varying regime signals into conditioning controls.
   - Supports semantic FiLM-style control, regime tokens, and uncertainty-aware semantic weighting.

3. **Adaptive Interaction Forecasting Backbone (AIFB)**
   - Forecasts OD flows through graph-aware interaction modeling.
   - Combines historical OD observations, graph kernels, pair-level static features, and adaptive pair gates.

4. **Prediction and Semantic Regularization Module (PSRM)**
   - Generates multi-step OD flow predictions.
   - Closes the semantic consistency loop by reconstructing static semantic roles and pair relations from latent OD states during training.

Together, these modules explicitly connect semantic prior construction, control generation, interaction restructuring, and semantic reconstruction from latent OD states.

## Repository Layout

```text
.
├── API_Config.json
├── Dataset/
│   ├── Data_Input/Sample_Group/
│   └── LLM_Outputs/Sample_Group/
├── LLM/
│   ├── main.py
│   └── LLM_Semantic/
├── SCOPE-OD/
│   ├── main.py
│   ├── configs/Sample_Group/01_quickstart.json
│   ├── model_part/
│   └── scripts/
├── requirements.txt
└── README.md
```

## Data Format

The public dataset is synthetic and contains five files under `Dataset/Data_Input/Sample_Group/`:

| File | Purpose |
| --- | --- |
| `Sample_Group_OD.txt` | Daily OD flow table with `date`, origin, destination, and `od_flow`. |
| `Sample_Group_city_static.txt` | City-level socioeconomic, transport, POI, and demographic placeholders. |
| `Sample_Group_city_dynamic.txt` | Daily city-level calendar and weather placeholders. |
| `Sample_Group_city_pair_static.txt` | OD-pair static distance, connectivity, and similarity placeholders. |
| `Sample_Group_pair_weather.txt` | Daily OD-pair weather-difference placeholders. |

Synthetic semantic prior tensors are provided under `Dataset/LLM_Outputs/Sample_Group/tensors/`. They are included so the quickstart can run without calling an external LLM API.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA environment if the default `pip` wheel is not appropriate for your system.

## Quickstart: Train on the Placeholder Dataset

```bash
cd SCOPE-OD
python main.py --config configs/Sample_Group/01_quickstart.json
```

By default, `main.py` uses one PyTorch CPU thread for stable quickstart behavior in constrained environments. Set `SCOPE_OD_NUM_THREADS` if you want to use more CPU threads, for example `SCOPE_OD_NUM_THREADS=8 python main.py --config configs/Sample_Group/01_quickstart.json`.

Outputs are written to:

```text
Results/SCOPE-OD/Sample_Group/
```

Each run saves the expanded config, training history, metrics, best checkpoint, and test predictions.

## Run Multiple Configs

```bash
cd SCOPE-OD
python run_config.py --configs Sample_Group/01_quickstart.json
```

## Regenerate Semantic Priors

The repository defaults to `mock_mode: true` in `API_Config.json`, so the RSPB pipeline can be tested without a real API key.

```bash
python LLM/main.py --groups Sample_Group --force
```

To use a real DeepSeek-compatible endpoint, set `mock_mode` to `false` and replace `YOUR_DEEPSEEK_API_KEY` in `API_Config.json`.

## Using Your Own Data

1. Create a new group directory under `Dataset/Data_Input/<Your_Group>/`.
2. Provide the five required input tables with the same schema as `Sample_Group`.
3. Generate semantic priors with:

```bash
python LLM/main.py --groups <Your_Group> --force
```

4. Create a config file under `SCOPE-OD/configs/<Your_Group>/`. Set `city_group` to `<Your_Group>`.
5. Train with:

```bash
cd SCOPE-OD
python main.py --config configs/<Your_Group>/<your_config>.json
```

## Notes for Public Release

- The included dataset and semantic tensors are synthetic placeholders.
- Local machine paths, Python bytecode, private caches, and raw private data have been removed.
- Large experiment outputs and checkpoints are ignored by `.gitignore`.
