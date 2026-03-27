from typing import Any, Callable

from preprocessing.mappers.xlam_function_calling import map_salesforce_xlam_function_calling_60k


MapperFn = Callable[[dict[str, list[Any]], list[int]], dict[str, list[Any]]]


MAPPERS: dict[str, MapperFn] = {
    "Salesforce/xlam-function-calling-60k": map_salesforce_xlam_function_calling_60k,
}
