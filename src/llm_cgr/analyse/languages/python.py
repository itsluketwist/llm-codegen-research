"""Utility functions for Python code analysis."""

import ast
import sys

from llm_cgr.analyse.languages.code_data import CodeData


PYTHON_STDLIB = getattr(
    sys, "stdlib_module_names", []
)  # use this below to determine packages


class PythonAnalyser(ast.NodeVisitor):
    def __init__(self):
        self.defined_funcs: set[str] = set()
        self.called_funcs: set[str] = set()
        self.imports: set[str] = set()
        self.packages: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # save defined functions
        self.defined_funcs.add(node.name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func = node.func

        # save `foo()` function calls
        if isinstance(func, ast.Name):
            self.called_funcs.add(func.id)

        # save `lib.method()` function calls
        elif isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name):
                self.called_funcs.add(f"{func.value.id}.{func.attr}")
            else:
                self.called_funcs.add(func.attr)

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        # save `import module` imports
        for alias in node.names:
            # save all imports
            self.imports.add(alias.name)

            # save external packages
            top_level = alias.name.split(".")[0]
            if top_level not in PYTHON_STDLIB:
                self.packages.add(top_level)

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        # save `from module import thing` imports
        module = node.module or ""

        # save external packages
        # node.level is 0 for absolute imports, 1+ for relative imports
        if module and module not in PYTHON_STDLIB and node.level == 0:
            self.packages.add(module)

        # save all imports
        for alias in node.names:
            full = f"{module}.{alias.name}" if module else alias.name
            self.imports.add(full)

        self.generic_visit(node)


def analyse_python_code(code: str) -> CodeData:
    """
    Analyse Python code to extract functions and imports.
    """
    tree = ast.parse(code)
    analyser = PythonAnalyser()
    analyser.visit(tree)
    return CodeData(
        defined_funcs=analyser.defined_funcs,
        called_funcs=analyser.called_funcs,
        packages=analyser.packages,
        imports=analyser.imports,
    )


# Example usage
if __name__ == "__main__":
    sample = """
import os, sys
from collections import deque
from .src import API
from . import thing

def foo(x):
    return bar(x)

API.something()

foo(10)
"""
    print(analyse_python_code(sample))
    # -> {'defined': ['foo'], 'called': ['bar', 'foo'], 'imports': ['collections.deque', 'os', 'sys']}


# # regex to extract imports from python code: `from module import thing`
# PYTHON_FROM_MODULE_IMPORT_REGEX = re.compile(
#     pattern=r"^\s*from\s+(\w[\w.]+)\s+import\s*\(?\s*(.*)$",
# )

# # regex to extract imports from python code: `import module`
# PYTHON_IMPORT_MODULE_REGEX = re.compile(
#     pattern=r"^\s*import\s+(.*)$",
# )

# def extract_python_imports(code: str) -> list[str]:
#     """Extracts library names from Python code."""

#     imports = set()

#     for line in code.splitlines():
#         line = line.split("#")[0]  # ignore comments

#         # find `from module import thing` imports
#         if match := PYTHON_FROM_MODULE_IMPORT_REGEX.search(string=line):
#             _import = match.group(1)

#             if "." in _import:
#                 _import, _, _ = _import.partition(".")

#             imports.add(_import)

#         # find `import module` imports
#         elif match := PYTHON_IMPORT_MODULE_REGEX.search(string=line):
#             matches = [m.strip() for m in match.group(1).split(",")]

#             for _import in matches:
#                 if _import.startswith("."):
#                     continue  # skip local imports

#                 if " as " in _import:
#                     _import, _, _ = _import.partition(" as ")

#                 if "." in _import:
#                     _import, _, _ = _import.partition(".")

#                 imports.add(_import)

#     return sorted(imports)


# def extract_python_functions(code: str) -> list[str]:
#     """Extracts function names from Python code."""
