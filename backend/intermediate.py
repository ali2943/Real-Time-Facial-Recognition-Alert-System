from typing import Dict, Any, List, Tuple


def generate_intermediate(ast: Dict[str, Any]) -> List[str]:
    stmt_type = ast["type"]
    instructions: List[str] = []

    if stmt_type == "CREATE_TABLE":
        instructions.append(f"CREATE_TABLE {ast['table']} ({', '.join(ast['columns'])})")
        return instructions

    if stmt_type == "INSERT":
        values = ", ".join(_value_repr(v) for v in ast["values"])
        instructions.append(f"INSERT_ROW {ast['table']} VALUES ({values})")
        return instructions

    if stmt_type == "SELECT":
        instructions.append(f"LOAD_TABLE {ast['table']}")
        temp_counter = 1
        if ast.get("where"):
            cond_temp, temp_counter = _emit_expression(ast["where"], instructions, temp_counter)
            instructions.append(f"FILTER {cond_temp}")
        instructions.append(f"PROJECT {', '.join(ast['columns'])}")
        instructions.append("EMIT_RESULT")
        return instructions

    if stmt_type == "UPDATE":
        instructions.append(f"LOAD_TABLE {ast['table']}")
        temp_counter = 1
        if ast.get("where"):
            cond_temp, temp_counter = _emit_expression(ast["where"], instructions, temp_counter)
            instructions.append(f"FILTER {cond_temp}")
        for assignment in ast["assignments"]:
            instructions.append(f"SET {assignment['column']} = {_value_repr(assignment['value'])}")
        instructions.append("WRITE_BACK")
        return instructions

    if stmt_type == "DELETE":
        instructions.append(f"LOAD_TABLE {ast['table']}")
        temp_counter = 1
        if ast.get("where"):
            cond_temp, temp_counter = _emit_expression(ast["where"], instructions, temp_counter)
            instructions.append(f"FILTER {cond_temp}")
        instructions.append("DELETE_ROWS")
        return instructions

    return ["NOP"]


def _emit_expression(expr: Dict[str, Any], instructions: List[str], temp_counter: int) -> Tuple[str, int]:
    if expr["type"] == "logical":
        left_temp, temp_counter = _emit_expression(expr["left"], instructions, temp_counter)
        right_temp, temp_counter = _emit_expression(expr["right"], instructions, temp_counter)
        current_temp = f"t{temp_counter}"
        instructions.append(f"{current_temp} = {left_temp} {expr['operator']} {right_temp}")
        return current_temp, temp_counter + 1

    if expr["type"] == "comparison":
        left = _value_repr(expr["left"])
        right = _value_repr(expr["right"])
        current_temp = f"t{temp_counter}"
        instructions.append(f"{current_temp} = {left} {expr['operator']} {right}")
        return current_temp, temp_counter + 1

    raise ValueError("Invalid expression type for intermediate code generation")


def _value_repr(node: Dict[str, Any]) -> str:
    if node["type"] == "literal":
        value = node["value"]
        return repr(value) if isinstance(value, str) else str(value)
    return str(node["value"])
