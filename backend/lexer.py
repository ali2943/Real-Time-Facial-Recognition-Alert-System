import re
from typing import List, Dict, Any

KEYWORDS = {
    "SELECT",
    "FROM",
    "WHERE",
    "CREATE",
    "TABLE",
    "INSERT",
    "INTO",
    "VALUES",
    "UPDATE",
    "SET",
    "DELETE",
    "AND",
    "OR",
}

TOKEN_PATTERN = re.compile(
    r"\s*(?:"
    r"(?P<NUMBER>\d+(?:\.\d+)?)|"
    r"(?P<STRING>'[^']*'|\"[^\"]*\")|"
    r"(?P<OPERATOR><=|>=|<>|!=|=|<|>)|"
    r"(?P<PUNCT>[(),;*])|"
    r"(?P<IDENTIFIER>[A-Za-z_][A-Za-z0-9_]*)|"
    r"(?P<INVALID>.)"
    r")"
)


def tokenize(query: str) -> List[Dict[str, Any]]:
    tokens: List[Dict[str, Any]] = []
    pos = 0

    while pos < len(query):
        match = TOKEN_PATTERN.match(query, pos)
        if not match:
            raise ValueError(f"Unexpected token near position {pos}")

        token_type = match.lastgroup
        token_value = match.group(token_type)
        pos = match.end()

        if token_type == "INVALID":
            raise ValueError(f"Invalid character: {token_value}")

        if token_type == "IDENTIFIER":
            upper_value = token_value.upper()
            if upper_value in KEYWORDS:
                tokens.append({"type": "KEYWORD", "value": upper_value, "category": "keyword"})
            else:
                tokens.append({"type": "IDENTIFIER", "value": token_value, "category": "identifier"})
            continue

        if token_type == "NUMBER":
            number_value = float(token_value) if "." in token_value else int(token_value)
            tokens.append({"type": "NUMBER", "value": number_value, "category": "literal"})
            continue

        if token_type == "STRING":
            tokens.append(
                {
                    "type": "STRING",
                    "value": token_value[1:-1],
                    "raw": token_value,
                    "category": "literal",
                }
            )
            continue

        if token_type == "OPERATOR":
            tokens.append({"type": "OPERATOR", "value": token_value, "category": "operator"})
            continue

        if token_type == "PUNCT":
            tokens.append({"type": "PUNCT", "value": token_value, "category": "punctuation"})
            continue

    tokens.append({"type": "EOF", "value": "EOF", "category": "meta"})
    return tokens
