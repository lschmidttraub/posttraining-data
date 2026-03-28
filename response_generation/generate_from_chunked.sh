#!/bin/bash

CHUNKED_DATASET_DIR="${CHUNKED_DATASET_DIR:-$SCRATCH/datasets/chunked/science}"

for DATASET_DIR in "${CHUNKED_DATASET_DIR}"/*/; do
  INPUT_DATASET="${DATASET_DIR%/}" sbatch response_generation/run_generation.sh
done
