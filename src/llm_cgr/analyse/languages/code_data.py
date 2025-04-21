"""Define the CodeData class, to store code analysis data."""

from dataclasses import dataclass
from typing import Iterable


@dataclass
class CodeData:
    """
    A class to hold code analysis data.
    """

    valid: bool | None
    defined_funcs: list[str]
    called_funcs: list[str]
    packages: list[str]
    imports: list[str]

    def __init__(
        self,
        valid: bool | None = None,
        defined_funcs: Iterable | None = None,
        called_funcs: Iterable | None = None,
        packages: Iterable | None = None,
        imports: Iterable | None = None,
    ):
        self.valid = valid
        self.defined_funcs = sorted(defined_funcs) if defined_funcs else []
        self.called_funcs = sorted(called_funcs) if called_funcs else []
        self.packages = sorted(packages) if packages else []
        self.imports = sorted(imports) if imports else []
