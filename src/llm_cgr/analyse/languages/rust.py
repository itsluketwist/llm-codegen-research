"""Utility functions for Rust code analysis."""

import re

from llm_cgr.analyse.languages.code_data import CodeData


# rust's three standard library crates - everything else comes from crates.io
RUST_STDLIB = frozenset({"std", "core", "alloc"})

# matches extern crate declarations: extern crate serde;
_EXTERN_CRATE_RE = re.compile(r"^\s*extern\s+crate\s+(\w+)", re.MULTILINE)


def _strip_comments(code: str) -> str:
    """Remove // line comments and /* */ block comments from code."""
    # remove block comments first (they can span multiple lines)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    # remove line comments
    code = re.sub(r"//[^\n]*", "", code)
    return code


def _extract_use_statements(code: str) -> list[str]:
    """
    Extract raw use paths from all use declarations in the code.

    Uses a bracket-aware scanner rather than plain regex, because braced imports
    like `use std::{io, fmt};` span multiple tokens and can be multi-line.
    Returns a list of raw path strings, e.g. ["std::collections::HashMap", "tokio"].
    """
    paths = []
    i = 0
    while i < len(code):
        # find the next 'use' keyword (word boundary ensures we skip 'reuse', etc.)
        match = re.search(r"\buse\s+", code[i:])
        if match is None:
            break

        start = i + match.end()
        depth = 0
        j = start

        # scan forward until we hit a ';' at brace-depth 0
        while j < len(code):
            c = code[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            elif c == ";" and depth == 0:
                paths.append(code[start:j].strip())
                break
            j += 1

        i = start  # advance past the 'use' keyword we just processed

    return paths


def _split_top_level(s: str) -> list[str]:
    """
    Split a comma-separated string into items, respecting brace nesting.

    Used to split the contents of {...} groups in use paths.
    """
    items = []
    depth = 0
    current: list[str] = []

    for c in s:
        if c == "{":
            depth += 1
            current.append(c)
        elif c == "}":
            depth -= 1
            current.append(c)
        elif c == "," and depth == 0:
            items.append("".join(current).strip())
            current = []
        else:
            current.append(c)

    if current:
        items.append("".join(current).strip())

    return items


def _expand_use_path(path: str, prefix: str = "") -> list[str]:
    """
    Recursively expand a Rust use path into fully-qualified dotted import strings.

    Handles simple paths, braced groups, wildcards, and 'as' aliases.
    For example:
      "std::collections::HashMap" -> ["std.collections.HashMap"]
      "std::{io::Read, fmt}"      -> ["std.io.Read", "std.fmt"]
      "std::io::{self, Write}"    -> ["std.io", "std.io.Write"]
    Returns a list of dotted import paths.
    """
    # strip any trailing 'as alias' at this level of the path
    path = re.sub(r"\s+as\s+\w+\s*$", "", path).strip()

    if not path:
        return []

    # handle wildcard: use std::collections::*;
    if path.endswith("::*"):
        prefix_part = path[:-3].replace("::", ".")
        full = f"{prefix}.{prefix_part}" if prefix else prefix_part
        return [f"{full}.*"]

    brace_idx = path.find("{")

    if brace_idx == -1:
        # simple path, no braces: std::collections::HashMap
        converted = path.replace("::", ".")
        full = f"{prefix}.{converted}" if prefix else converted
        return [full]

    # braced path: std::{io::Read, fmt} or std::io::{self, Write}
    # everything before the brace is the common prefix
    before_brace = path[:brace_idx].rstrip(":").replace("::", ".")
    new_prefix = (
        f"{prefix}.{before_brace}"
        if (prefix and before_brace)
        else (prefix or before_brace)
    )

    # extract the content inside the outermost braces
    close_idx = path.rfind("}")
    inner = path[brace_idx + 1 : close_idx]

    results = []
    for item in _split_top_level(inner):
        item = item.strip()
        if not item:
            continue

        if item == "self":
            # 'self' refers to the module itself (the prefix path)
            results.append(new_prefix)
        else:
            results.extend(_expand_use_path(item, prefix=new_prefix))

    return results


def analyse_rust_code(code: str) -> CodeData:
    """
    Analyse Rust code to extract imported crates and their paths.

    Only extracts use and extern crate declarations; no usage tracking or syntax
    validation is performed (valid is always True).
    Returns a CodeData object with import information.
    """
    std_libs: set[str] = set()
    ext_libs: set[str] = set()
    imports: set[str] = set()

    # strip comments so 'use' inside comments isn't matched
    clean_code = _strip_comments(code)

    # process each use declaration
    for raw_path in _extract_use_statements(clean_code):
        for dotted_path in _expand_use_path(raw_path):
            imports.add(dotted_path)
            # the top-level segment is the crate name
            top_level = dotted_path.split(".")[0]
            if top_level in RUST_STDLIB:
                std_libs.add(top_level)
            else:
                ext_libs.add(top_level)

    # handle extern crate declarations (older rust style, still used occasionally)
    for match in _EXTERN_CRATE_RE.finditer(clean_code):
        crate_name = match.group(1)
        # 'self' and 'std' as extern crate are special cases, skip them
        if crate_name in ("self", "super"):
            continue
        imports.add(crate_name)
        if crate_name in RUST_STDLIB:
            std_libs.add(crate_name)
        else:
            ext_libs.add(crate_name)

    return CodeData(
        valid=True,
        std_libs=std_libs,
        ext_libs=ext_libs,
        imports=imports,
        lib_usage={},
    )
