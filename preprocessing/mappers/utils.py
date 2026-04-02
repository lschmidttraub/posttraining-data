import json
from typing import Any, Callable, Optional

from preprocessing.schema import STANDARD_COLUMNS


def row_mapper_to_batched(
    fn: Callable[[dict[str, Any], int], Optional[dict[str, Any]]],
) -> Callable[[dict[str, list[Any]], list[int]], dict[str, list[Any]]]:
    """Wrap a per-row mapper into the batched signature expected by the pipeline.

    *fn(row, idx)* receives a single row dict and its dataset index.
    It must return a dict with keys from STANDARD_COLUMNS, or ``None`` to
    skip (filter) the row.
    """

    def batched(
        batch: dict[str, list[Any]], indices: list[int]
    ) -> dict[str, list[Any]]:
        keys = list(batch.keys())
        n = len(indices)
        out: dict[str, list[Any]] = {col: [] for col in STANDARD_COLUMNS}
        for i in range(n):
            row = {k: batch[k][i] for k in keys}
            result = fn(row, indices[i])
            if result is not None:
                for col in STANDARD_COLUMNS:
                    out[col].append(result[col])
        return out

    return batched


def safe_json_loads(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value if value is not None else default


def normalize_argument_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, dict)):
        return value
    return str(value)


def inject_system_prompt(messages: list[dict[str, Any]], system: str) -> list[dict[str, str]]:
    normalized_messages = [
        {
            "role": str(message["role"]),
            "content": str(message["content"]),
        }
        for message in messages
    ]

    for idx, message in enumerate(normalized_messages):
        if message["role"] != "system":
            continue

        content = message["content"]
        normalized_messages[idx] = {
            "role": "system",
            "content": f"{content}\n{system}" if content else system,
        }
        return normalized_messages

    return [{"role": "system", "content": system}, *normalized_messages]
