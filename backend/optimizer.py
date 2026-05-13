import re
from typing import List

OPERATORS = ("<=", ">=", "<>", "!=", "=", "<", ">")


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
    if " = " not in instruction:
        return instruction

    temp, expression = instruction.split(" = ", 1)
    comparison = _split_comparison(expression)
    if comparison is None:
        return instruction

    left, operator, right = comparison
    left_value = _parse_literal(left)
    right_value = _parse_literal(right)

    if left_value is None or right_value is None:
        return instruction

    result = _eval_comparison(left_value, right_value, operator)
    return f"{temp} = {result}"


def _split_comparison(expression: str):
    expression = expression.strip()
    for operator in OPERATORS:
        marker = f" {operator} "
        if marker not in expression:
            continue
        left, right = expression.split(marker, 1)
        return left.strip(), operator, right.strip()
    return None


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
    match = re.match(r"^\s*([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*$", instruction)
    return bool(match and match.group(1) == match.group(2))
