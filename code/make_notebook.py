import json
from pathlib import Path

root = Path('.')
files = {
    "model": root / "model.py",
    "plots": root / "plots.py",
    "solvers.__init__": root / "solvers" / "__init__.py",
    "solvers.common": root / "solvers" / "common.py",
    "solvers.qp_box_eq": root / "solvers" / "qp_box_eq.py",
    "solvers.centralized": root / "solvers" / "centralized.py",
    "solvers.admm": root / "solvers" / "admm.py",
    "solvers.dual": root / "solvers" / "dual.py",
    "solvers.primal": root / "solvers" / "primal.py",
    "solvers.primal_dual": root / "solvers" / "primal_dual.py",
    "solvers.decomposition": root / "solvers" / "decomposition.py",
    "main": root / "main.py",
}

def clean_text(text: str) -> str:
    if text.startswith('\\\n'):
        return text[2:]
    return text

def md_cell(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")[:-1]] + (["\n"] if text.endswith("\n") else []) if "\n" in text else [text]
    }

def code_cell(text: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")[:-1]] + (["\n"] if text.endswith("\n") else []) if "\n" in text else [text]
    }

cells = []
intro = "# Distributed Optimal Load management (Notebook version)\n\nThis notebook bundles the entire project code (models, solvers, plotting, and driver) into a single file.\nRun the cells in order to recreate the original Python modules in-memory and execute the demo run."
cells.append(md_cell(intro))

setup_code = """import os, sys, types, json, numpy as np

# Limit BLAS threads for deterministic lightweight runs
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

# Create an in-memory solvers package placeholder so relative imports work
solvers_pkg = types.ModuleType('solvers')
solvers_pkg.__path__ = []
sys.modules['solvers'] = solvers_pkg
"""
cells.append(code_cell(setup_code))

for name, path in files.items():
    text = clean_text(path.read_text())
    title = name.replace('__init__','__init__ (package)')
    cells.append(md_cell(f"## Module: `{title}`"))
    var_name = name.replace('.', '_') + "_code"
    pkg = name.rsplit('.', 1)[0] if '.' in name else ''
    attr = name.split('.')[-1]
    lines = []
    lines.append("import types, sys, os")
    lines.append(f"{var_name} = r'''{text}'''\n")
    lines.append(f"module = types.ModuleType('{name}')")
    lines.append(f"module.__dict__['__package__'] = '{pkg}'")
    lines.append("module.__dict__['__file__'] = os.path.join(os.getcwd(), '" + attr + ".py')")
    lines.append(f"exec({var_name}, module.__dict__)")
    lines.append(f"sys.modules['{name}'] = module")
    if name.startswith('solvers.'):
        lines.append("solvers_pkg = sys.modules['solvers']")
        lines.append(f"setattr(solvers_pkg, '{attr}', module)")
    cells.append(code_cell('\n'.join(lines)))

cells.append(md_cell("## Example run"))
cells.append(code_cell("from main import run\nrun()\n"))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = root / "project.ipynb"
out_path.write_text(json.dumps(nb, indent=2))
print(f"Wrote notebook to {out_path}")
