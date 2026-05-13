import re
from typing import List


COMPARISON_PATTERN = re.compile(r"^(t\d+) = (.+?)\s(<=|>=|<>|!=|=|<|>)\s(.+)$")


def optimize(instructions: List[str]) -> List[str]:
    optimized: List[str] = []

    for instruction in instructions:
        folded = _constant_fold(instruction)
        if folded == "NOP":
            continue

        if optimized and optimized[-1] == folded:
            continue

        if _is_redundant_self_assignment(folded):
            continue

        optimized.append(folded)

    return optimized


def _constant_fold(instruction: str) -> str:
    match = COMPARISON_PATTERN.match(instruction)
    if not match:
        return instruction

    temp, left, operator, right = match.groups()
    left_value = _parse_literal(left)
    right_value = _parse_literal(right)

    if left_value is None or right_value is None:
        return instruction

    result = _eval_comparison(left_value, right_value, operator)
    return f"{temp} = {result}"


def _parse_literal(value: str):
    value = value.strip()

    if value in {"True", "False"}:
        return value == "True"

    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return None


def _eval_comparison(left, right, operator: str) -> bool:
    if operator == "=":
        return left == right
    if operator in {"!=", "<>"}:
        return left != right
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    if operator == ">=":
        return left >= right
    raise ValueError(f"Unsupported operator: {operator}")


def _is_redundant_self_assignment(instruction: str) -> bool:
    return bool(re.match(r"^([A-Za-z_]\w*)\s*=\s*\1$", instruction))
