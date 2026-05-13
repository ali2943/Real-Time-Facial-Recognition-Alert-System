from copy import deepcopy
from typing import Dict, Any, List


class InMemoryDatabase:
    def __init__(self):
        self.tables: Dict[str, Dict[str, Any]] = {}

    def export_symbol_table(self) -> Dict[str, Any]:
        tables_snapshot = {}
        for table_name, table_data in self.tables.items():
            tables_snapshot[table_name] = {
                "columns": list(table_data["columns"]),
                "values": deepcopy(table_data["rows"]),
            }
        return {"tables": tables_snapshot}

    def execute(self, ast: Dict[str, Any]) -> Any:
        stmt_type = ast["type"]

        if stmt_type == "CREATE_TABLE":
            table = ast["table"]
            self.tables[table] = {"columns": list(ast["columns"]), "rows": []}
            return {"message": f"Table '{table}' created"}

        if stmt_type == "INSERT":
            table = self.tables[ast["table"]]
            row = {}
            for column_name, value_node in zip(table["columns"], ast["values"]):
                row[column_name] = value_node["value"]
            table["rows"].append(row)
            return {"message": f"1 row inserted into '{ast['table']}'"}

        if stmt_type == "SELECT":
            table = self.tables[ast["table"]]
            rows = self._filter_rows(table["rows"], ast.get("where"))
            if ast["columns"] == ["*"]:
                return deepcopy(rows)
            return [{column: row.get(column) for column in ast["columns"]} for row in rows]

        if stmt_type == "UPDATE":
            table = self.tables[ast["table"]]
            rows = self._filter_rows(table["rows"], ast.get("where"))
            for row in rows:
                for assignment in ast["assignments"]:
                    row[assignment["column"]] = self._resolve_value(assignment["value"], row)
            return {"message": f"{len(rows)} row(s) updated in '{ast['table']}'"}

        if stmt_type == "DELETE":
            table = self.tables[ast["table"]]
            if not ast.get("where"):
                count = len(table["rows"])
                table["rows"] = []
                return {"message": f"{count} row(s) deleted from '{ast['table']}'"}

            kept_rows: List[Dict[str, Any]] = []
            removed_count = 0
            for row in table["rows"]:
                if self._evaluate_expression(ast["where"], row):
                    removed_count += 1
                else:
                    kept_rows.append(row)
            table["rows"] = kept_rows
            return {"message": f"{removed_count} row(s) deleted from '{ast['table']}'"}

        raise ValueError(f"Unsupported statement type: {stmt_type}")

    def _filter_rows(self, rows: List[Dict[str, Any]], where_expr: Dict[str, Any] | None) -> List[Dict[str, Any]]:
        if where_expr is None:
            return rows
        return [row for row in rows if self._evaluate_expression(where_expr, row)]

    def _evaluate_expression(self, expr: Dict[str, Any], row: Dict[str, Any]) -> bool:
        expr_type = expr["type"]

        if expr_type == "logical":
            left = self._evaluate_expression(expr["left"], row)
            right = self._evaluate_expression(expr["right"], row)
            if expr["operator"] == "AND":
                return left and right
            return left or right

        if expr_type == "comparison":
            left = self._resolve_value(expr["left"], row)
            right = self._resolve_value(expr["right"], row)
            return self._compare(left, right, expr["operator"])

        raise ValueError("Unsupported expression type")

    def _resolve_value(self, node: Dict[str, Any], row: Dict[str, Any]):
        if node["type"] == "identifier":
            return row.get(node["value"])
        return node["value"]

    def _compare(self, left, right, operator: str) -> bool:
        if operator == "=":
            return left == right
        if operator in {"!=", "<>"}:
            return left != right
        if operator == ">":
            return left > right
        if operator == "<":
            return left < right
        if operator == ">=":
            return left >= right
        if operator == "<=":
            return left <= right
        raise ValueError(f"Unsupported operator '{operator}'")
