import os
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory

from lexer import tokenize
from parser import Parser
from semantic import SemanticAnalyzer
from intermediate import generate_intermediate
from optimizer import optimize
from codegen import generate_code
from executor import InMemoryDatabase


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
db = InMemoryDatabase()
semantic_analyzer = SemanticAnalyzer()


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.post("/run")
def run_query():
    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        tokens = tokenize(query)
        parser = Parser(tokens)
        ast = parser.parse()

        semantic_message = semantic_analyzer.analyze(ast, db.export_symbol_table())

        intermediate_code = generate_intermediate(ast)
        optimized_code = optimize(intermediate_code)
        code_generation = generate_code(optimized_code)
        output = db.execute(ast)

        return jsonify(
            {
                "tokens": [token for token in tokens if token["type"] != "EOF"],
                "symbol_table": db.export_symbol_table(),
                "semantic": semantic_message,
                "intermediate_code": intermediate_code,
                "optimized_code": optimized_code,
                "code_generation": code_generation,
                "final_code": "\n".join(code_generation),
                "output": output,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Unhandled query execution error")
        return jsonify({"error": "Internal compiler error"}), 500


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "0") == "1")
