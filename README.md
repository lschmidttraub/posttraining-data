# Post-Training Data Processing Pipeline for Self-Distillation Post-Training
This is fork of the `response-generation` branch of the `posttraining-data` repo, focused on expert response generation for self-distillation.
Make sure you have the `cuda-13-fix-final` branch of the `model-launch` repository in you `$SCRATCH` before running the script.

For clarity, I've moved all the folders that aren't (yet) directly relevant for us into the `legacy/` folder, as a useful reference. 

The data pipeline is split into 2 stages: preprocessing and response generation.
## Pre-processing
In the pre-processing stage, transform our datasets into a standardized format containing the following columns:
`prompt`, `reference`, `data_source`, `data_source_id`, `meta_information` and `turn`.

We implement a batched mapping function for each individual dataset.

## Response-generation
The response generation stage then uses the `prompt` and `reference` columns to create `answer`, `thinking`, `generation_model` and `generation_meta` columns.

The generation works as follows:
- The `response-generation/run_generation.sh` file submits a single-node job than runs `response-generation/run_generation.py`
- The python script first submits a job to launch the server through the `model-launch` repo
- It then waits for the server to start, and once it starts, it locally runs `response-generation/generate.py`, which sends requests to the server.

The generation script also optionally takes care of preprocessing, to avoid any unnecessary intermediates.


## Current difficulties with generation
Multi-worker generation with vLLM currently doesn't work.
We use a single worker with CUDA graphs disactivated (through the `--enforce-eager` flag) to generate.


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
