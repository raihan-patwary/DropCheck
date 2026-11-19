"""Checks that the notebooks in kernels/ run exactly the code that ran on Kaggle.

The comments and text cells were tidied after the Kaggle runs. This script downloads each notebook from Kaggle and
compares every code cell by its syntax tree, which ignores comments and docstrings but catches any change to the code.

    python scripts/check_code_matches_kaggle.py

Needs the Kaggle command line tool, logged in.
"""
import ast
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = {
    "poultry-feces-01-data-audit": "01_data_audit/01_data_audit.ipynb",
    "poultry-feces-02-paper-replication": "02_paper_replication/02_paper_replication.ipynb",
    "poultry-feces-03-leak-free": "03_leak_free/03_leak_free.ipynb",
    "poultry-feces-04-stability-and-recipe": "04_stability_and_recipe/04_stability_and_recipe.ipynb",
    "poultry-feces-05-shortcuts-and-labels": "05_shortcuts_and_labels/05_shortcuts_and_labels.ipynb",
}


def drop_docstrings(tree):
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def code_cells(path):
    cells = json.load(open(path, encoding="utf-8"))["cells"]
    sources = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
               for c in cells if c["cell_type"] == "code"]
    return [ast.dump(drop_docstrings(ast.parse(s))) for s in sources]


if __name__ == "__main__":
    all_same = True
    with tempfile.TemporaryDirectory() as tmp:
        for slug, local in NOTEBOOKS.items():
            out = os.path.join(tmp, slug)
            subprocess.run(["kaggle", "kernels", "pull", f"mdraihanpatwary02/{slug}", "-p", out],
                           check=True, capture_output=True)
            remote = os.path.join(out, next(f for f in os.listdir(out) if f.endswith(".ipynb")))
            same = code_cells(remote) == code_cells(os.path.join(ROOT, "kernels", local))
            all_same &= same
            print(f"{'same' if same else 'DIFFERENT':9s} {slug}")
    print("all code identical to the Kaggle runs" if all_same else "some code differs")
    sys.exit(0 if all_same else 1)
