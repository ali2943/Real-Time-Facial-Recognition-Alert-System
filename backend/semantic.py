from typing import Dict, Any, List


class SemanticAnalyzer:
    def analyze(self, ast: Dict[str, Any], symbol_table: Dict[str, Any]) -> str:
        tables = symbol_table.get("tables", {})
        stmt_type = ast["type"]

        if stmt_type == "CREATE_TABLE":
            table = ast["table"]
            if table in tables:
                raise ValueError(f"Semantic error: table '{table}' already exists")
            if len(set(ast["columns"])) != len(ast["columns"]):
                raise ValueError("Semantic error: duplicate column names in CREATE TABLE")
            return f"Semantic analysis successful: CREATE TABLE '{table}' is valid"

        table_name = ast.get("table")
        if table_name not in tables:
            raise ValueError(f"Semantic error: table '{table_name}' does not exist")

        table_columns = tables[table_name]["columns"]

        if stmt_type == "INSERT":
            if len(ast["values"]) != len(table_columns):
                raise ValueError(
                    f"Semantic error: INSERT value count ({len(ast['values'])}) does not match table columns ({len(table_columns)})"
                )
            return f"Semantic analysis successful: INSERT into '{table_name}' is valid"

        if stmt_type == "SELECT":
            self._validate_select_columns(ast["columns"], table_columns)
            self._validate_where(ast.get("where"), table_columns)
            return f"Semantic analysis successful: SELECT from '{table_name}' is valid"

        if stmt_type == "UPDATE":
            for assignment in ast["assignments"]:
                if assignment["column"] not in table_columns:
                    raise ValueError(
                        f"Semantic error: column '{assignment['column']}' does not exist in table '{table_name}'"
                    )
                self._validate_value_node(assignment["value"], table_columns)
            self._validate_where(ast.get("where"), table_columns)
            return f"Semantic analysis successful: UPDATE on '{table_name}' is valid"

        if stmt_type == "DELETE":
            self._validate_where(ast.get("where"), table_columns)
            return f"Semantic analysis successful: DELETE from '{table_name}' is valid"

        raise ValueError(f"Semantic error: unsupported statement type '{stmt_type}'")

    def _validate_select_columns(self, columns: List[str], table_columns: List[str]) -> None:
        if columns == ["*"]:
            return
        for column in columns:
            if column not in table_columns:
                raise ValueError(f"Semantic error: column '{column}' not found")

    def _validate_where(self, where_expr: Dict[str, Any] | None, table_columns: List[str]) -> None:
        if not where_expr:
            return
        self._validate_expression(where_expr, table_columns)

    def _validate_expression(self, expr: Dict[str, Any], table_columns: List[str]) -> None:
        expr_type = expr["type"]
        if expr_type == "logical":
            self._validate_expression(expr["left"], table_columns)
            self._validate_expression(expr["right"], table_columns)
            return
        if expr_type == "comparison":
            self._validate_value_node(expr["left"], table_columns)
            self._validate_value_node(expr["right"], table_columns)
            return
        raise ValueError("Semantic error: invalid WHERE expression")

    def _validate_value_node(self, node: Dict[str, Any], table_columns: List[str]) -> None:
        if node["type"] == "identifier" and node["value"] not in table_columns:
            raise ValueError(f"Semantic error: column '{node['value']}' not found")
