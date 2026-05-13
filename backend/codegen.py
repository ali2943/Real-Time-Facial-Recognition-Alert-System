from typing import List


def generate_code(instructions: List[str]) -> List[str]:
    return [f"Step {index}: {instruction}" for index, instruction in enumerate(instructions, start=1)]
