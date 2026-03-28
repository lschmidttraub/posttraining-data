from typing import Any, Callable

from preprocessing.mappers.science.medical_o1 import map_medical_o1
from preprocessing.mappers.science.multi_subject_rlvr import map_multi_subject_rlvr
from preprocessing.mappers.science.natural_reasoning import map_natural_reasoning
from preprocessing.mappers.science.nemotron_science import map_nemotron_science
from preprocessing.mappers.science.textbook_reasoning import map_textbook_reasoning
from preprocessing.mappers.tool_calling.xlam_function_calling import map_salesforce_xlam_function_calling_60k


MapperFn = Callable[[dict[str, list[Any]], list[int]], dict[str, list[Any]]]


TOOL_CALLING_MAPPERS: dict[str, MapperFn] = {
    "MadeAgents/xlam-irrelevance-7.5k": map_salesforce_xlam_function_calling_60k,
    "Salesforce/xlam-function-calling-60k": map_salesforce_xlam_function_calling_60k,
}

MATH_AND_CODING_MAPPERS: dict[str, MapperFn] = {
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
    "math_and_coding": MATH_AND_CODING_MAPPERS,
    "science": SCIENCE_MAPPERS,
    "instruction_following": INSTRUCTION_FOLLOWING_MAPPERS,
}

MAPPERS: dict[str, MapperFn] = {k: v for d in MAPPER_REGISTRY.values() for k, v in d.items()}