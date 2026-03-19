
## IF-RewardBench

OverallAssessment mode is inappropriate because it compares response sets rather than evaluate single responses. Do ConstraintAsssessment instead.

Running `python constraint_assessment_inference_vllm.py --model_name Qwen/Qwen3-8B --model_path Qwen/Qwen3-8B` is kind of slow (~15 min for 6K samples)

Not appropriate for evaluating the usefulness of the custom judges we have because this benchmark measures how well the judge fills out a checklist of instructions that the response should follow, whereas our judges differ mainly in the prompting and there is no suitable way to inject our custom prompts

What's still possible is to compare different ways of filling out the checklist (e.g., taking output literally vs computing expected logprob)

Conclusion: Skip this benchmark for now.

