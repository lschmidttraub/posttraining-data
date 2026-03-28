# Post-Training Data Processing Pipeline for Self-Distillation Post-Training
This is fork of the `response-generation` branch of the `posttraining-data` repo, focused on expert response generation for self-distillation.
Make sure you have the `cuda-13-fix-final` branch of the `model-launch` repository in you `$SCRATCH` before running the script.

For clarity, I've moved all the folders that aren't (yet) directly relevant for us into the `legacy/` folder, as a useful reference. 

The data pipeline is split into 2 stages: preprocessing and response generation.
## Pre-processing

The preprocessing stage maps raw HuggingFace datasets into a standardized schema and saves the result as a single `train` split ready for response generation.

### Output schema

Every preprocessed dataset contains exactly these columns:

| Column | Description |
|---|---|
| `prompt` | Input prompt (string or JSON-serialized object) |
| `reference` | Reference / ground-truth response |
| `data_source` | Source dataset identifier |
| `data_source_id` | Per-row identifier within the source dataset |
| `meta_information` | Arbitrary JSON metadata (tools, constraints, …) |
| `turn` | Conversation turn index (0-indexed) |

### Registry

Mappers are organized by category in `preprocessing/registry.py`. Each category is a `dict[dataset_name → mapper_fn]`:


A **mapper function** has the signature `(batch: dict[str, list], indices: list[int]) -> dict[str, list]` and is applied with `Dataset.map(..., batched=True, with_indices=True)`. Batched mappers can emit more or fewer rows than they receive, which is useful for filtering and multi-turn expansion.

To register a new dataset, add its mapper function to the appropriate category dict. The dataset name must be the exact HuggingFace Hub identifier (or local path) used to load it.

### Pipeline

For each dataset in the selected category the runner:
1. Loads all splits from the Hub (or disk).
2. Applies the mapper to every split, dropping all original columns and selecting only the standard schema columns.
3. Concatenates all resulting splits across all datasets into a single flat dataset.
4. Deduplicates by `prompt`, keeping the first occurrence and discarding later duplicates.
5. Saves the result as a single `train` split to `OUTPUT_DIR`.

The choice of datasets to preprocess is controlled by 2 environment variables:
| Variable | Default | Description |
|---|---|---|
| `CATEGORY` | _(None)_ | Category key from `MAPPER_REGISTRY` (required) |
| `DATASETS` | _(empty)_ | Comma-separated subset of dataset names from the category. Defaults to all datasets in the category if empty. |

**Examples**
Process all `tool_calling` datasets:
```bash
CATEGORY=tool_calling preprocessing/run_preprocessing.sh
```

Process a specific subset and upload to the Hub:
```bash
CATEGORY=tool_calling \
DATASETS="Salesforce/xlam-function-calling-60k,MadeAgents/xlam-irrelevance-7.5k" \
UPLOAD_TO_HUB=1 \
HUB_DATASET_ID="lasgroup/xlam-combined" \
preprocessing/run_preprocessing.sh
```

## Response generation

The response generation stage queries a served model for every prompt in a preprocessed dataset and adds the model's responses as new columns.

### Output schema

The output dataset extends the preprocessing schema with four additional columns:

| Column | Description |
|---|---|
| `answer` | Model response (without thinking tokens) |
| `thinking` | Chain-of-thought / reasoning trace (empty string for non-reasoning models) |
| `generation_model` | Full model identifier used for generation |
| `generation_meta` | JSON string with generation hyperparameters (`temperature`, `max_length`) |

### Pipeline

`run_generation.sh` submits a single-node SLURM client job per model listed in `JOBS`. Each job runs `run_generation.py`, which orchestrates the following steps:

1. **(Optional) Preprocessing** — if `PREPROCESS=1`, `run_generation.py` calls `preprocessing/run.py` inline before generation to avoid storing an intermediate dataset.
2. **Server launch** — submits a separate SLURM job via `$SCRATCH/model-launch` that starts a vLLM or SGLang server and waits for it to register its URL in the job log.
3. **Health check** — polls the server's `/health` endpoint, then sends a small probe inference request to confirm the full pipeline (including any router) is operational.
4. **Async generation** — `generate.py` streams requests to the server's `/v1/chat/completions` endpoint with up to `--concurrent` (default 2000) requests in flight simultaneously. Responses are written to `responses.jsonl` immediately as they arrive, so the run is resumable if interrupted.
5. **Reconstruction** — after all requests complete, the JSONL file is merged back into a HuggingFace `Dataset` with the four output columns and saved to `<BASE_OUTPUT_DIR>/<model-name>/`.
6. **Server teardown** — the server SLURM job is cancelled automatically.

### Thinking token handling

For reasoning models, `generate.py` reads the `reasoning_content` field that SGLang/vLLM expose for thinking models (e.g. Qwen3, QwQ). As a fallback it parses inline `<think>…</think>` tags from the content, which also handles GLM-style responses that omit the opening tag.

### Resuming interrupted runs

If a `responses.jsonl` checkpoint exists in the output directory, already-completed indices are skipped automatically. Pass `--retry-existing` (set by default in `run_generation.sh`) to also retry any indices whose saved answer is an empty string. 

### Running

Edit the `run_generation.sh` script's environment variables and `JOBS`, then submit:

Key environment variables:

| Variable |  Description |
|---|---|
| `INPUT_DATASET` | HuggingFace dataset to generate responses for |
| `INPUT_DATASETS` | Comma-separated list of datasets (requires `PREPROCESS=1`) |
| `PREPROCESS` | Set to `1` preprocess as well (not recommended) |
| `PREPROCESS_MAPPER`  | Mapper name for inline preprocessing |
| `PREPROCESSED_DATASET_DIR`  | Where to save/read the preprocessed dataset |
| `BASE_OUTPUT_DIR` | Root output directory; model name is appended as a subdirectory |

### Known limitations

CUDA graphs are disabled (`--enforce-eager`) due to compatibility issues with the current multi-node vLLM setup.



# Old README
`posttraining-data` is a turn-key 8-stage pipeline for processing HuggingFace datasets into training-ready format. It was used to prepare Apertus' post-training data and notably its [SFT mixture](https://huggingface.co/datasets/swiss-ai/apertus-sft-mixture). More information can be found in the [Apertus tech report](https://github.com/swiss-ai/apertus-tech-report).

## Pipeline Stages

The pipeline consists of the following self-contained stages:
1. **01-hf-download**: Downloads HuggingFace datasets with metadata tracking → produces HF DatasetDict
2. **02-standardisation**: Converts datasets to unified chat format → produces HF DatasetDict  
3. **03-license-based-filtering**: Removes samples with licensing restrictions → produces HF DatasetDict
4. **04-decontamination**: Removes contaminated samples from evaluation sets → produces HF DatasetDict
5. **05-annotations**: Adds LLM-based classifications and language detection → produces HF DatasetDict
6. **06-field-based-filtering**: General field analysis and filtering → produces HF DatasetDict
7. **07-dataset-aggregation**: Combines multiple datasets into training mixtures → produces HF Dataset ready for training
8. **08-judge-evaluation**: Evaluates datasets with LLM judges.

A few additional running scripts and miscellaneous commands are also provided in `examples`. 

## Setup

Create virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
