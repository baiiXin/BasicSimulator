from __future__ import annotations

import ast
from pathlib import Path


def test_production_package_does_not_import_newton():
    package = Path(__file__).resolve().parents[1] / "standalone_contact"
    offenders = []

    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "newton" or alias.name.startswith("newton."):
                        offenders.append((path.name, node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "newton" or module.startswith("newton."):
                    offenders.append((path.name, node.lineno, module))

    assert not offenders, f"Production package imports Newton: {offenders}"
