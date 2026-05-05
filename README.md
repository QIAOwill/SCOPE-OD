# SCOPE-OD: Semantic Consistency-Oriented Prior-Enhanced OD Flow Forecasting

SCOPE-OD is a semantic-prior-enhanced framework for origin-destination (OD) flow forecasting. The public repository contains executable model code and a small synthetic placeholder dataset. The placeholder files preserve the expected schema and directory layout, but they do not contain private or production data.
<img width="6803" height="3779" alt="Image" src="https://github.com/user-attachments/assets/d6ccda51-4458-4ef6-8836-765d8346dea5" />

## LLM Prompt Configuration

SCOPE-OD uses the LLM only in the **Role-Aware Semantic Prior Builder (RSPB)** stage. The prompt is designed to convert city-level and OD-pair attributes into structured semantic priors. These priors are then saved as tables and tensors for downstream training. The LLM is not used to directly forecast OD flows.

The public repository uses synthetic placeholder data and enables `mock_mode: true` by default. Therefore, the quickstart can run without sending any request to an external LLM service. When `mock_mode` is disabled, the RSPB pipeline follows the prompt structure below.

### System Prompt

```text
You are an urban mobility semantics analyst.
Your task is to infer stable semantic mobility roles and OD-pair interaction patterns from structured city and OD-pair attributes.
Use only the provided attributes. Do not introduce external facts, private information, or unsupported assumptions.
Return valid JSON only. Do not include Markdown, explanations, or additional text outside the JSON object.
```

### City-Level Semantic Role Prompt Template

```text
Given the following city-level attributes for a city in the target study region, infer its semantic mobility role.

City identifier: {city_id}
City static attributes:
{city_static_attributes}

City dynamic/context attributes, if available:
{city_dynamic_attributes}

Assign the city to one origin-side role and one destination-side role.
Also provide a soft probability distribution over candidate roles, a confidence score, and a short evidence summary based only on the provided attributes.

Candidate origin-side roles:
{origin_role_candidates}

Candidate destination-side roles:
{destination_role_candidates}

Return the result using the following JSON schema:
{
  "city_id": "<string>",
  "origin_role": "<string>",
  "destination_role": "<string>",
  "origin_role_probabilities": {"<role>": <float>},
  "destination_role_probabilities": {"<role>": <float>},
  "confidence": <float>,
  "evidence": "<short attribute-grounded explanation>"
}
```

### OD-Pair Semantic Relation Prompt Template

```text
Given the following attributes for an origin-destination city pair, infer the semantic interaction relation of the OD pair.

Origin city identifier: {origin_city_id}
Destination city identifier: {destination_city_id}
Origin city semantic prior:
{origin_city_semantic_prior}
Destination city semantic prior:
{destination_city_semantic_prior}

OD-pair static attributes:
{pair_static_attributes}

OD-pair dynamic/context attributes, if available:
{pair_dynamic_attributes}

Assign the OD pair to one semantic interaction relation.
Also provide a soft probability distribution over candidate relations, a confidence score, and a short evidence summary based only on the provided attributes.

Candidate OD-pair relation labels:
{pair_relation_candidates}

Return the result using the following JSON schema:
{
  "origin_city_id": "<string>",
  "destination_city_id": "<string>",
  "pair_relation": "<string>",
  "pair_relation_probabilities": {"<relation>": <float>},
  "confidence": <float>,
  "evidence": "<short attribute-grounded explanation>"
}
```

### Prompt Output Usage

The parsed LLM outputs are converted into semantic-prior artifacts under `Dataset/LLM_Outputs/<Group>/`, including:

| Artifact Type | Purpose |
| --- | --- |
| Semantic label tables | Store discrete city-role and OD-pair relation assignments. |
| Probability tables | Store soft semantic distributions for uncertainty-aware conditioning. |
| Confidence and entropy values | Control the strength of semantic conditioning and regularization. |
| Tensor files | Provide model-ready semantic priors for DSCM, AIFB, and PSRM. |

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
