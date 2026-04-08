from typing import Any, Callable

from datasets import DatasetDict, load_dataset

from preprocessing.mappers.math_and_coding.big_math_rl import load_big_math_rl, map_big_math_rl
from preprocessing.mappers.math_and_coding.dapo_math import load_dapo_math, map_dapo_math
from preprocessing.mappers.math_and_coding.deepmath import map_deepmath
from preprocessing.mappers.math_and_coding.nemotron_cascade_code import load_nemotron_cascade_code, map_nemotron_cascade_code
from preprocessing.mappers.math_and_coding.nemotron_cp import load_nemotron_cp, map_nemotron_cp
from preprocessing.mappers.math_and_coding.nemotron_math import load_nemotron_math, map_nemotron_math
from preprocessing.mappers.math_and_coding.numina_math import map_numina_math
from preprocessing.mappers.math_and_coding.ocr2 import load_ocr2, map_ocr2
from preprocessing.mappers.science.medical_o1 import map_medical_o1
from preprocessing.mappers.science.multi_subject_rlvr import map_multi_subject_rlvr
from preprocessing.mappers.science.natural_reasoning import map_natural_reasoning
from preprocessing.mappers.science.nemotron_science import map_nemotron_science
from preprocessing.mappers.science.textbook_reasoning import map_textbook_reasoning
from preprocessing.mappers.tool_calling.nemotron_rl_agentic_conversational_tool_use_pivot import map_nemotron_rl_agentic_conversational_tool_use_pivot
from preprocessing.mappers.tool_calling.toolace import map_toolace
from preprocessing.mappers.tool_calling.when2call import map_when2call_train_sft
from preprocessing.mappers.tool_calling.xlam_function_calling import map_xlam_function_calling

def load_when2call() -> DatasetDict:
    return DatasetDict({"train": load_dataset("nvidia/When2Call", "train_sft", split="train")})

def load_nemotron_agentic() -> DatasetDict:
    return load_dataset("nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1", trust_remote_code=True)


MapperFn = Callable[[dict[str, list[Any]], list[int]], dict[str, list[Any]]]


TOOL_CALLING_MAPPERS: dict[str, MapperFn] = {
    "MadeAgents/xlam-irrelevance-7.5k": map_xlam_function_calling,
    "Salesforce/xlam-function-calling-60k": map_xlam_function_calling,
    "Team-ACE/ToolACE": map_toolace,
    "nvidia/When2Call": map_when2call_train_sft,
    "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1": map_nemotron_rl_agentic_conversational_tool_use_pivot,
}

MATH_MAPPERS: dict[str, MapperFn] = {
    "zwhe99/DeepMath-103K": map_deepmath,
    "nlile/NuminaMath-1.5-RL-Verifiable": map_numina_math,
    "open-r1/DAPO-Math-17k-Processed": map_dapo_math,
    "open-r1/Big-Math-RL-Verified-Processed": map_big_math_rl,
    "nvidia/Nemotron-Math-v2": map_nemotron_math,
}

CODING_MAPPERS: dict[str, MapperFn] = {
    "nvidia/OpenCodeReasoning-2": map_ocr2,
    "nvidia/Nemotron-Cascade-SFT-Stage-1": map_nemotron_cascade_code,
    "nvidia/Nemotron-Competitive-Programming-v1": map_nemotron_cp,
}

SCIENCE_MAPPERS: dict[str, MapperFn] = {
    "MegaScience/TextbookReasoning": map_textbook_reasoning,
    "facebook/natural_reasoning": map_natural_reasoning,
    "nvidia/Nemotron-Science-v1": map_nemotron_science,
    "FreedomIntelligence/medical-o1-verifiable-problem": map_medical_o1,
    "virtuoussy/Multi-subject-RLVR": map_multi_subject_rlvr,
}

INSTRUCTION_FOLLOWING_MAPPERS: dict[str, MapperFn] = {
}


MAPPER_REGISTRY: dict[str, dict[str, MapperFn]] = {
    "tool_calling": TOOL_CALLING_MAPPERS,
    "math": MATH_MAPPERS,
    "coding": CODING_MAPPERS,
    "science": SCIENCE_MAPPERS,
    "instruction_following": INSTRUCTION_FOLLOWING_MAPPERS,
}

MAPPERS: dict[str, MapperFn] = {k: v for d in MAPPER_REGISTRY.values() for k, v in d.items()}


DATASET_LOADERS: dict[str, Callable[[], DatasetDict]] = {
    "open-r1/DAPO-Math-17k-Processed": load_dapo_math,
    "open-r1/Big-Math-RL-Verified-Processed": load_big_math_rl,
    "nvidia/Nemotron-Math-v2": load_nemotron_math,
    "nvidia/OpenCodeReasoning-2": load_ocr2,
    "nvidia/Nemotron-Cascade-SFT-Stage-1": load_nemotron_cascade_code,
    "nvidia/Nemotron-Competitive-Programming-v1": load_nemotron_cp,
    "nvidia/When2Call": load_when2call,
    "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1": load_nemotron_agentic,
}