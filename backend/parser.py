from typing import List, Dict, Any


class Parser:
    def __init__(self, tokens: List[Dict[str, Any]]):
        self.tokens = tokens
        self.index = 0

    def parse(self) -> Dict[str, Any]:
        statement = self.parse_statement()
        self._consume_optional_punct(";")
        self._expect("EOF")
        return statement

    def parse_statement(self) -> Dict[str, Any]:
        keyword = self._peek_value().upper()
        if keyword == "CREATE":
            return self.parse_create()
        if keyword == "INSERT":
            return self.parse_insert()
        if keyword == "SELECT":
            return self.parse_select()
        if keyword == "UPDATE":
            return self.parse_update()
        if keyword == "DELETE":
            return self.parse_delete()
        raise ValueError(f"Unsupported statement starting with '{self._peek_value()}'")

    def parse_create(self) -> Dict[str, Any]:
        self._expect_keyword("CREATE")
        self._expect_keyword("TABLE")
        table = self._expect("IDENTIFIER")["value"]
        self._expect_punct("(")

        columns = [self._expect("IDENTIFIER")["value"]]
        while self._consume_optional_punct(","):
            columns.append(self._expect("IDENTIFIER")["value"])

        self._expect_punct(")")
        return {"type": "CREATE_TABLE", "table": table, "columns": columns}

    def parse_insert(self) -> Dict[str, Any]:
        self._expect_keyword("INSERT")
        self._expect_keyword("INTO")
        table = self._expect("IDENTIFIER")["value"]
        self._expect_keyword("VALUES")
        self._expect_punct("(")

        values = [self.parse_literal()]
        while self._consume_optional_punct(","):
            values.append(self.parse_literal())

        self._expect_punct(")")
        return {"type": "INSERT", "table": table, "values": values}

    def parse_select(self) -> Dict[str, Any]:
        self._expect_keyword("SELECT")

        if self._peek_type() == "PUNCT" and self._peek_value() == "*":
            self._advance()
            columns = ["*"]
        else:
            columns = [self._expect("IDENTIFIER")["value"]]
            while self._consume_optional_punct(","):
                columns.append(self._expect("IDENTIFIER")["value"])

        self._expect_keyword("FROM")
        table = self._expect("IDENTIFIER")["value"]

        where = None
        if self._peek_type() == "KEYWORD" and self._peek_value().upper() == "WHERE":
            self._advance()
            where = self.parse_or_expression()

        return {"type": "SELECT", "table": table, "columns": columns, "where": where}

    def parse_update(self) -> Dict[str, Any]:
        self._expect_keyword("UPDATE")
        table = self._expect("IDENTIFIER")["value"]
        self._expect_keyword("SET")

        assignments = []
        assignments.append(self.parse_assignment())
        while self._consume_optional_punct(","):
            assignments.append(self.parse_assignment())

        where = None
        if self._peek_type() == "KEYWORD" and self._peek_value().upper() == "WHERE":
            self._advance()
            where = self.parse_or_expression()

        return {"type": "UPDATE", "table": table, "assignments": assignments, "where": where}

    def parse_delete(self) -> Dict[str, Any]:
        self._expect_keyword("DELETE")
        self._expect_keyword("FROM")
        table = self._expect("IDENTIFIER")["value"]

        where = None
        if self._peek_type() == "KEYWORD" and self._peek_value().upper() == "WHERE":
            self._advance()
            where = self.parse_or_expression()

        return {"type": "DELETE", "table": table, "where": where}

    def parse_assignment(self) -> Dict[str, Any]:
        column = self._expect("IDENTIFIER")["value"]
        self._expect_operator("=")
        value = self.parse_literal_or_identifier()
        return {"column": column, "value": value}

    def parse_or_expression(self) -> Dict[str, Any]:
        expr = self.parse_and_expression()
        while self._peek_type() == "KEYWORD" and self._peek_value().upper() == "OR":
            self._advance()
            right = self.parse_and_expression()
            expr = {"type": "logical", "operator": "OR", "left": expr, "right": right}
        return expr

    def parse_and_expression(self) -> Dict[str, Any]:
        expr = self.parse_comparison_expression()
        while self._peek_type() == "KEYWORD" and self._peek_value().upper() == "AND":
            self._advance()
            right = self.parse_comparison_expression()
            expr = {"type": "logical", "operator": "AND", "left": expr, "right": right}
        return expr

    def parse_comparison_expression(self) -> Dict[str, Any]:
        left = self.parse_literal_or_identifier()
        operator = self._expect("OPERATOR")["value"]
        right = self.parse_literal_or_identifier()
        return {"type": "comparison", "left": left, "operator": operator, "right": right}

    def parse_literal_or_identifier(self) -> Dict[str, Any]:
        token_type = self._peek_type()
        if token_type == "IDENTIFIER":
            return {"type": "identifier", "value": self._advance()["value"]}
        return self.parse_literal()

    def parse_literal(self) -> Dict[str, Any]:
        token_type = self._peek_type()
        if token_type not in {"NUMBER", "STRING"}:
            raise ValueError(f"Expected literal, got {self._peek_value()}")
        token = self._advance()
        return {"type": "literal", "value": token["value"]}

    def _peek(self) -> Dict[str, Any]:
        if self.index >= len(self.tokens):
            return {"type": "EOF", "value": "EOF"}
        return self.tokens[self.index]

    def _peek_type(self) -> str:
        return self._peek()["type"]

    def _peek_value(self) -> Any:
        return self._peek()["value"]

    def _advance(self) -> Dict[str, Any]:
        if self.index >= len(self.tokens):
            raise ValueError("Unexpected end of input")
        token = self.tokens[self.index]
        self.index += 1
        return token

    def _expect(self, expected_type: str) -> Dict[str, Any]:
        token = self._peek()
        if token["type"] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {token['type']} ({token['value']})")
        return self._advance()

    def _expect_keyword(self, keyword: str) -> Dict[str, Any]:
        token = self._expect("KEYWORD")
        if token["value"].upper() != keyword.upper():
            raise ValueError(f"Expected keyword {keyword}, got {token['value']}")
        return token

    def _expect_operator(self, op: str) -> Dict[str, Any]:
        token = self._expect("OPERATOR")
        if token["value"] != op:
            raise ValueError(f"Expected operator {op}, got {token['value']}")
        return token

    def _expect_punct(self, punct: str) -> Dict[str, Any]:
        token = self._expect("PUNCT")
        if token["value"] != punct:
            raise ValueError(f"Expected '{punct}', got '{token['value']}'")
        return token

    def _consume_optional_punct(self, punct: str) -> bool:
        if self._peek_type() == "PUNCT" and self._peek_value() == punct:
            self._advance()
            return True
        return False
