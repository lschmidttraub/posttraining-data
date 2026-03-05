import subprocess
import os
from pathlib import Path
import time
from datasets import load_dataset

agieval_datasets = [
    {
        "name_or_path": dataset_name,
        "config_name": None,
        "split_name": "test",
        "prompt_col_name": "query",
    }
    for dataset_name in [
        "hails/agieval-aqua-rat",
        "hails/agieval-gaokao-biology",
        "hails/agieval-gaokao-chemistry",
        "hails/agieval-gaokao-chinese",
        "hails/agieval-gaokao-english",
        "hails/agieval-gaokao-geography",
        "hails/agieval-gaokao-history",
        "hails/agieval-gaokao-mathqa",
        "hails/agieval-gaokao-physics",
        "hails/agieval-logiqa-en",
        "hails/agieval-logiqa-zh",
        "hails/agieval-sat-math",
        "hails/agieval-lsat-ar",
        "hails/agieval-lsat-lr",
        "hails/agieval-lsat-rc",
        "hails/agieval-sat-en",
        "hails/agieval-sat-en-without-passage",
        "hails/agieval-math",
        "hails/agieval-gaokao-mathcloze",
        "hails/agieval-jec-qa-kd",
        "hails/agieval-jec-qa-ca",
    ]
]

BENCHMARK_DATASETS = [
    {
        "name_or_path": "cais/mmlu",
        "config_name": "all",
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "TIGER-Lab/MMLU-Pro",
        "config_name": None,
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "CohereLabs/Global-MMLU",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "li-lab/MMLU-ProX",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "truthfulqa/truthful_qa",
        "config_name": "iterate",
        "split_name": "validation",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "alexandrainst/m_truthfulqa",
        "config_name": "iterate",
        "split_name": "val",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "CohereLabs/include-base-44",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "include-base_v2",
        "config_name": "iterate",
        "split_name": None,
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "lukaemon/bbh",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "input",
    },
    {
        "name_or_path": "EleutherAI/drop",
        "config_name": None,
        "split_name": "validation",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "ibm-research/acp_bench",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "allenai/ai2_arc",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "LumiOpen/arc_challenge_mt",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "alexandrainst/m_arc",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "instruction",
    },
    {
        "name_or_path": "Idavidrein/gpqa",
        "config_name": "gpqa_main",
        "split_name": "train",
        "prompt_col_name": "Pre-Revision Question",
    },
    {
        "name_or_path": "Qwen/P-MMEval",
        "config_name": ["mlogiqa"],
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "juletxara/mgsm",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "DigitalLearningGmbH/MATH-lighteval",
        "config_name": "default",
        "split_name": "test",
        "prompt_col_name": "problem",
    },
    {
        "name_or_path": "openai/gsm8k",
        "config_name": "main",
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "madrylab/gsm8k-platinum",
        "config_name": None,
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "allenai/math_qa",
        "config_name": None,
        "split_name": "test",
        "prompt_col_name": "Problem",
    },
    {
        "name_or_path": "EleutherAI/hendrycks_math",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "problem",
    },
    {
        "name_or_path": "AI-MO/aimo-validation-aime",
        "config_name": None,
        "split_name": "train",
        "prompt_col_name": "problem",
    },
    {
        "name_or_path": "Qwen/PolyMath",
        "config_name": "iterate",
        "split_name": ["top", "high", "medium", "low"],
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "openai/openai_humaneval",
        "config_name": None,
        "split_name": "test",
        "prompt_col_name": "prompt",
    },
    {
        "name_or_path": "THUDM/LongBench",
        "config_name": ["hotpotqa"],
        "split_name": "test",
        "prompt_col_name": "input",
    },
    {
        "name_or_path": "google-research-datasets/mbpp",
        "config_name": "full",
        "split_name": "test",
        "prompt_col_name": "text",
    },
    {
        "name_or_path": "bigcode/bigcodebench",
        "config_name": None,
        "split_name": "v0.1.0_hf",
        "prompt_col_name": "instruct_prompt",
    },
    {
        "name_or_path": "google/IFEval",
        "config_name": None,
        "split_name": "train",
        "prompt_col_name": "prompt",
    },
    {
        "name_or_path": "Rowan/hellaswag",
        "config_name": None,
        "split_name": "test",
        "prompt_col_name": "ctx",
    },
    {
        "name_or_path": "alexandrainst/m_hellaswag",
        "config_name": "iterate",
        "split_name": "val",
        "prompt_col_name": "ctx",
    },
    {
        "name_or_path": "facebook/Multi-IF",
        "config_name": None,
        "split_name": "train",
        "prompt_col_name": "turn_1_prompt",
    },
    {
        "name_or_path": "tatsu-lab/alpaca_eval",
        "config_name": None,
        "split_name": "eval",
        "prompt_col_name": "instruction",
    },
    {
        "name_or_path": "CohereLabs/m-ArenaHard-v2.0",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "prompt",
    },
    {
        "name_or_path": "lmarena-ai/arena-hard-auto",
        "config_name": None,
        "split_name": "test",
        "prompt_col_name": "prompt",
    },
    {
        "name_or_path": "toxigen/toxigen-data",
        "config_name": "prompts",
        "split_name": [
            "hate_trans_1k",
            "neutral_black_1k",
            "hate_native_american_1k",
            "neutral_immigrant_1k",
            "hate_middle_east_1k",
            "neutral_lgbtq_1k",
            "neutral_women_1k",
            "neutral_chinese_1k",
            "hate_latino_1k",
            "hate_bisexual_1k",
            "hate_mexican_1k",
            "hate_asian_1k",
            "neutral_mental_disability_1k",
            "neutral_mexican_1k",
            "hate_mental_disability_1k",
            "neutral_bisexual_1k",
            "neutral_latino_1k",
            "hate_chinese_1k",
            "neutral_jewish_1k",
            "hate_muslim_1k",
            "neutral_asian_1k",
            "hate_physical_disability_1k",
            "hate_jewish_1k",
            "neutral_muslim_1k",
            "hate_immigrant_1k",
            "hate_black_1k",
            "hate_lgbtq_1k",
            "hate_women_1k",
            "neutral_middle_east_1k",
            "neutral_native_american_1k",
            "neutral_physical_disability_1k",
        ],
        "prompt_col_name": "text",
    },
    {
        "name_or_path": "allenai/real-toxicity-prompts",
        "config_name": None,
        "split_name": "train",
        "prompt_col_name": "prompt",
    },
    {
        "name_or_path": "oskarvanderwal/bbq",
        "config_name": "All",
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "CohereLabs/aya_redteaming",
        "config_name": None,
        "split_name": [
            "arabic",
            "english",
            "filipino",
            "french",
            "hindi",
            "russian",
            "serbian",
            "spanish",
        ],
        "prompt_col_name": "prompt",
    },
    {
        "name_or_path": "ToxicityPrompts/PolygloToxicityPrompts",
        "config_name": "iterate",
        "split_name": "full",
        "prompt_col_name": "prompt",
    },
    {
        "name_or_path": "nayeon212/BLEnD",
        "config_name": ["multiple-choice-questions"],
        "split_name": "test",
        "prompt_col_name": "prompt",
    },
    {
        "name_or_path": "shanearora/CaLMQA",
        "config_name": None,
        "split_name": "train",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "kellycyy/CulturalBench",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "prompt_question",
    },
    {
        "name_or_path": "swiss-ai/harmbench",
        "config_name": ["DirectRequest", "HumanJailbreaks"],
        "split_name": "test",
        "prompt_col_name": "Behavior",
    },
    {
        "name_or_path": "DAMO-NLP-SG/MultiJail",
        "config_name": None,
        "split_name": "train",
        "prompt_col_name": ["en", "zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"],
    },
    {
        "name_or_path": "tau/commonsense_qa",
        "config_name": None,
        "split_name": ["train", "test", "validation"],
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "HuggingFaceH4/MATH-500",
        "config_name": None,
        "split_name": "test",
        "prompt_col_name": "problem",
    },
    {
        "name_or_path": "swiss-ai/hallulens",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "prompt",
    },
    {
        "name_or_path": "EleutherAI/lambada_openai",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "text",
    },
    {
        "name_or_path": "baber/piqa",
        "config_name": None,
        "split_name": ["train", "test", "validation"],
        "prompt_col_name": "goal",
    },
    {
        "name_or_path": "HiTZ/truthfulqa-multi",
        "config_name": "iterate",
        "split_name": "validation",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "allenai/IFBench_test",
        "config_name": None,
        "split_name": "train",
        "prompt_col_name": "prompt",
    },
    {
        "name_or_path": "cais/wmdp",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "cambridgeltl/xcopa",
        "config_name": "iterate",
        "split_name": ["validation", "test"],
        "prompt_col_name": "premise",
    },
    {
        "name_or_path": "facebook/xnli",
        "config_name": [
            "ar",
            "bg",
            "de",
            "el",
            "en",
            "es",
            "fr",
            "hi",
            "ru",
            "sw",
            "th",
            "tr",
            "ur",
            "vi",
            "zh",
        ],
        "split_name": "train",
        "prompt_col_name": "premise",
    },
    {
        "name_or_path": "swiss-ai/polyglotoxicityprompts",
        "config_name": "iterate",
        "split_name": "full",
        "prompt_col_name": "text",
    },
    {
        "name_or_path": "swiss-ai/math_qa",
        "config_name": None,
        "split_name": ["train", "test", "validation"],
        "prompt_col_name": "Problem",
    },
    {
        "name_or_path": "swiss-ai/blend-sample",
        "config_name": None,
        "split_name": "test",
        "prompt_col_name": "prompt",
    },
    {
        "name_or_path": "swiss-ai/mlogiqa",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "swiss-ai/realtoxicityprompts",
        "config_name": "realtoxicityprompts_full",
        "split_name": "train",
        "prompt_col_name": "prompt",
    },
] + agieval_datasets

# Create logs directory
logs_dir = Path("./logs")
logs_dir.mkdir(exist_ok=True)

for dataset in BENCHMARK_DATASETS:
    dataset_name = dataset["name_or_path"]

    if os.path.exists(
        f"/iopsstor/scratch/cscs/smarian/datasets/apertus/decontamination_cache/{dataset_name}"
    ):
        print(f"Dataset {dataset_name} already exists. Skipping download.")
        continue

    print(f"Downloading dataset: {dataset_name}")
    log_file = logs_dir / f"{dataset_name.replace('/', '_')}.log"

    with open(log_file, "w") as f:
        env = os.environ.copy()
        env["HF_TOKEN"] = os.getenv("HF_TOKEN")

        subprocess.run(
            [
                "huggingface-cli",
                "download",
                dataset_name,
                "--repo-type",
                "dataset",
                "--local-dir",
                f"/iopsstor/scratch/cscs/smarian/datasets/apertus/decontamination_cache/{dataset_name}",
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )

    time.sleep(15)  # Sleep for a short time to avoid overwhelming the Hugging Face API
