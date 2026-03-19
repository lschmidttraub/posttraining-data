This directory contains code for evaluating judging approaches on benchmarks from the literature, including:

- IF-RewardBench (Wen et al., 2026)
- CALM (Ye et al., 2025)
- JudgeBench (Tan et al., 2024)
- RMBench (Liu et al., 2024)

Add new judging approaches under `judges/`. 

Add new benchmarks under `benchmarks`. Add new code under `scripts` to run the judges on these benchmarks wherever necessary, keeping the logs of these runs under `logs/`. Report the outcomes of these evaluations under `results`.