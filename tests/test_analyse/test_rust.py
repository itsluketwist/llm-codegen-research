"""Test the Rust analyse module."""

from llm_cgr import Markdown
from llm_cgr.analyse.languages.rust import analyse_rust_code


TEST_LLM_RESPONSE = """
Here's a Rust solution to process some data.

```rust
use std::collections::HashMap;
use std::io::{self, Read};
use tokio::runtime::Runtime;
use serde::{Serialize, Deserialize};

fn main() {
    let mut map: HashMap<String, i32> = HashMap::new();
    let rt = Runtime::new().unwrap();
    println!("{:?}", map);
}
```

Some more code:

```rs
use std::fmt;
use anyhow::Result;

fn format_it(x: i32) -> Result<String> {
    Ok(fmt::format(format_args!("{}", x)))
}
```

Run some code:

```bash
cargo run
```

Some very bad unknown code:

```
for import xxx)[
```
"""


def test_markdown():
    """
    Test the Markdown class extracting and analysing multiple Rust code blocks.
    """
    analysed = Markdown(text=TEST_LLM_RESPONSE)

    assert analysed.text == TEST_LLM_RESPONSE
    assert f"{analysed}" == TEST_LLM_RESPONSE
    assert len(analysed.code_blocks) == 4
    assert [cb.__repr__() for cb in analysed.code_blocks] == [
        "CodeBlock(language=rust, lines=10)",
        "CodeBlock(language=rust, lines=6)",
        "CodeBlock(language=bash, lines=1)",
        "CodeBlock(language=unspecified, lines=1)",
    ]
    assert analysed.code_errors == []
    assert analysed.languages == ["bash", "rust"]

    # first rust block: std and external crates, braced imports
    rust_code_one = analysed.code_blocks[0]
    assert rust_code_one.language == "rust"
    assert rust_code_one.valid is True
    assert rust_code_one.error is None
    assert rust_code_one.ext_libs == ["serde", "tokio"]
    assert rust_code_one.std_libs == ["std"]
    assert rust_code_one.lib_imports == [
        "serde.Deserialize",
        "serde.Serialize",
        "std.collections.HashMap",
        "std.io",
        "std.io.Read",
        "tokio.runtime.Runtime",
    ]
    assert rust_code_one.lib_usage == {}

    # second rust block: 'rs' alias should normalise to 'rust'
    rust_code_two = analysed.code_blocks[1]
    assert rust_code_two.language == "rust"
    assert rust_code_two.valid is True
    assert rust_code_two.error is None
    assert rust_code_two.ext_libs == ["anyhow"]
    assert rust_code_two.std_libs == ["std"]
    assert rust_code_two.lib_imports == ["anyhow.Result", "std.fmt"]
    assert rust_code_two.lib_usage == {}

    # bash block: no analysis
    bash_code = analysed.code_blocks[2]
    assert bash_code.language == "bash"
    assert bash_code.valid is None
    assert bash_code.error is None
    assert bash_code.ext_libs == []
    assert bash_code.std_libs == []
    assert bash_code.lib_imports == []
    assert bash_code.lib_usage == {}

    # unlabelled block: defaults to python, invalid python -> valid=None
    unknown_code = analysed.code_blocks[3]
    assert unknown_code.language is None
    assert unknown_code.valid is None
    assert unknown_code.error is None

    # check navigation helpers
    assert analysed.first_code_block("rust") == rust_code_one
    assert analysed.first_code_block("bash") == bash_code
    assert analysed.first_code_block("python") is None


def test_braced_imports():
    """Test nested brace expansion for complex use declarations."""
    code = """
use std::{
    collections::{HashMap, BTreeMap},
    io::Write,
};
"""
    result = analyse_rust_code(code)
    assert result.valid is True
    assert result.std_libs == ["std"]
    assert result.ext_libs == []
    assert result.lib_imports == [
        "std.collections.BTreeMap",
        "std.collections.HashMap",
        "std.io.Write",
    ]


def test_extern_crate():
    """Test that extern crate declarations are extracted as ext_libs."""
    code = """
extern crate regex;
extern crate serde;

fn main() {}
"""
    result = analyse_rust_code(code)
    assert result.valid is True
    assert result.std_libs == []
    assert result.ext_libs == ["regex", "serde"]
    assert "regex" in result.lib_imports
    assert "serde" in result.lib_imports


def test_wildcard_import():
    """Test that wildcard imports are tracked with a '.*' suffix."""
    code = "use std::prelude::*;"
    result = analyse_rust_code(code)
    assert result.valid is True
    assert result.std_libs == ["std"]
    assert result.lib_imports == ["std.prelude.*"]


def test_self_import():
    """Test that 'self' in a braced import refers to the module itself."""
    code = "use std::io::{self, Read};"
    result = analyse_rust_code(code)
    assert result.valid is True
    assert result.std_libs == ["std"]
    assert result.lib_imports == ["std.io", "std.io.Read"]


def test_alias_import():
    """Test that 'as' aliases are stripped and the original path is kept."""
    code = "use std::collections::HashMap as Map;"
    result = analyse_rust_code(code)
    assert result.valid is True
    assert result.lib_imports == ["std.collections.HashMap"]


def test_comment_stripping():
    """Test that use statements inside comments are not extracted."""
    code = """
// use fake::crate;
/* use another::fake; */
use real::crate_name;
"""
    result = analyse_rust_code(code)
    assert result.valid is True
    assert result.lib_imports == ["real.crate_name"]
    assert result.ext_libs == ["real"]
