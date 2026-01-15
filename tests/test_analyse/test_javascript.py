"""Test the JavaScript analyse module."""

from llm_cgr import Markdown
from llm_cgr.analyse.languages.javascript import analyse_javascript_code


TEST_LLM_RESPONSE = """
Here's a JavaScript solution to process some data and return an answer.

```javascript
import lodash from 'lodash';
import { get } from 'axios';
import fs from 'fs';
import { createHash } from 'crypto';
import * as path from 'path';

async function processData(data) {
    const response = await get("https://api.example.com/data");
    const arr = lodash.map([1, 2, 3, 4, 5], x => x * 2);
    const normalized = lodash.chain(data).filter().value();
    const hash = createHash('sha256');
    return lodash.sortBy(arr, response, lodash.random);
}

processData("data");
```

Some more code:

```js
import dayjs from 'dayjs';
import * as url from 'url';

const formatted = dayjs().format('YYYY-MM-DD');
const parsed = url.parse("https://example.com");
```

Run some code:

```bash
node script.js
```

Some very bad javascript code:

```javascript
import problem from 'problem';

problem.bad_brackets((()
```

Some very bad unknown code:

```
for import xxx)[
```
"""


def test_markdown():
    """
    Test the MarkdownResponse class, extracting and analysing multiple JavaScript code blocks.
    """
    analysed = Markdown(text=TEST_LLM_RESPONSE)

    assert analysed.text == TEST_LLM_RESPONSE
    assert f"{analysed}" == TEST_LLM_RESPONSE
    assert len(analysed.code_blocks) == 5
    assert [cb.__repr__() for cb in analysed.code_blocks] == [
        "CodeBlock(language=javascript, lines=15)",
        "CodeBlock(language=javascript, lines=5)",
        "CodeBlock(language=bash, lines=1)",
        "CodeBlock(language=javascript, lines=3)",
        "CodeBlock(language=unspecified, lines=1)",
    ]
    assert len(analysed.code_errors) == 1
    assert "3:" in analysed.code_errors[0]
    assert analysed.languages == ["bash", "javascript"]

    js_code_one = analysed.code_blocks[0]
    assert js_code_one.language == "javascript"
    assert js_code_one.valid is True
    assert js_code_one.error is None
    assert js_code_one.ext_libs == [
        "axios",
        "lodash",
    ]
    assert js_code_one.std_libs == [
        "crypto",
        "fs",
        "path",
    ]
    assert js_code_one.lib_imports == [
        "axios.get",
        "crypto.createHash",
        "fs",
        "lodash",
        "path",
    ]
    assert js_code_one.lib_usage == {
        "axios": [
            {
                "type": "call",
                "member": "get",
                "args": ['"https://api.example.com/data"'],
                "kwargs": {},
            }
        ],
        "lodash": [
            {
                "type": "call",
                "member": "map",
                "args": ["[1, 2, 3, 4, 5]", "<arrow function>"],
                "kwargs": {},
            },
            {
                "type": "call",
                "member": "chain",
                "args": ["data"],
                "kwargs": {},
            },
            {
                "type": "call",
                "member": "sortBy",
                "args": ["arr", "response", "lodash.random"],
                "kwargs": {},
            },
            {
                "type": "access",
                "member": "random",
            },
        ],
        "crypto": [
            {
                "type": "call",
                "member": "createHash",
                "args": ['"sha256"'],
                "kwargs": {},
            }
        ],
    }

    js_code_two = analysed.code_blocks[1]
    assert js_code_two.language == "javascript"
    assert js_code_two.valid is True
    assert js_code_two.error is None
    assert js_code_two.ext_libs == ["dayjs"]
    assert js_code_two.std_libs == ["url"]
    assert js_code_two.lib_imports == ["dayjs", "url"]
    assert js_code_two.lib_usage == {
        "dayjs": [
            {
                "type": "call",
                "member": "",
                "args": [],
                "kwargs": {},
            }
        ],
        "url": [
            {
                "type": "call",
                "member": "parse",
                "args": ['"https://example.com"'],
                "kwargs": {},
            }
        ],
    }

    bash_code = analysed.code_blocks[2]
    assert bash_code.language == "bash"
    assert bash_code.valid is None
    assert bash_code.error is None
    assert bash_code.ext_libs == []
    assert bash_code.std_libs == []
    assert bash_code.lib_imports == []
    assert bash_code.lib_usage == {}

    bad_code = analysed.code_blocks[3]
    assert bad_code.language == "javascript"
    assert bad_code.valid is False
    assert bad_code.error is not None
    assert "Unexpected" in bad_code.error or "Error" in bad_code.error
    assert bad_code.ext_libs == []
    assert bad_code.std_libs == []
    assert bad_code.lib_imports == []
    assert bad_code.lib_usage == {}

    unknown_code = analysed.code_blocks[4]
    assert unknown_code.text == "for import xxx)["
    assert unknown_code.language is None
    assert unknown_code.valid is None
    assert unknown_code.error is None
    assert unknown_code.ext_libs == []
    assert unknown_code.std_libs == []
    assert unknown_code.lib_imports == []
    assert unknown_code.lib_usage == {}

    assert unknown_code.markdown == "```\nfor import xxx)[\n```"
    assert f"{unknown_code}" == "for import xxx)["

    assert analysed.first_code_block("javascript") == js_code_one
    assert analysed.first_code_block("bash") == bash_code
    assert analysed.first_code_block("python") is None


def test_relative_imports():
    """Test that relative imports are handled but not categorized as std/ext libs."""
    code = """
import { helper } from './utils';
import config from '../config';

helper();
"""
    result = analyse_javascript_code(code)
    assert result.valid is True
    assert result.std_libs == []
    assert result.ext_libs == []


def test_scoped_npm_packages():
    """Test handling of scoped npm packages like @org/package."""
    code = """
import { something } from '@babel/core';
import preset from '@babel/preset-env';

something();
preset();
"""
    result = analyse_javascript_code(code)
    assert result.valid is True
    assert result.std_libs == []
    assert "@babel/core" in result.ext_libs
    assert "@babel/preset-env" in result.ext_libs


def test_node_protocol_imports():
    """Test handling of node: protocol imports."""
    code = """
import fs from 'node:fs';
import path from 'node:path';

fs.readFile('test.txt');
"""
    result = analyse_javascript_code(code)
    assert result.valid is True
    assert "fs" in result.std_libs
    assert "path" in result.std_libs
    assert result.ext_libs == []


def test_computed_member_expressions():
    """Test that computed member expressions are handled correctly."""
    code = """
import lodash from 'lodash';

const key = 'map';
const result = lodash[key]([1, 2, 3]);
const value = lodash.data[0];
"""
    result = analyse_javascript_code(code)
    assert result.valid is True
    assert "lodash" in result.ext_libs


def test_complex_argument_unparsing():
    """Test unparsing of various argument types."""
    code = """
import lib from 'testlib';

lib.withObject({ name: "test", value: 42 });
lib.withCall(console.log("hello"));
lib.withComputed(obj[key]);
lib.withFunction(function() { return 1; });
lib.withTemplate(`hello ${name}`);
lib.withSpread(...args);
lib.withUnknown(class {});
"""
    result = analyse_javascript_code(code)
    assert result.valid is True

    lib_usage = result.lib_usage.get("testlib", [])
    args_by_member = {u["member"]: u["args"] for u in lib_usage if u["type"] == "call"}

    assert args_by_member["withObject"] == ['{name: "test", value: 42}']
    assert args_by_member["withCall"] == ['console.log("hello")']
    assert args_by_member["withComputed"] == ["obj[key]"]
    assert args_by_member["withFunction"] == ["<function>"]
    assert args_by_member["withTemplate"] == ["<template literal>"]
    assert args_by_member["withSpread"] == ["...args"]
    assert args_by_member["withUnknown"] == ["<ClassExpression>"]


def test_access_filtering_with_nested_members():
    """Test that access records are filtered when a nested member is also used."""
    code = """
import lib from 'testlib';

const x = lib.utils;
const y = lib.utils.helper();
"""
    result = analyse_javascript_code(code)
    assert result.valid is True

    lib_usage = result.lib_usage.get("testlib", [])
    access_members = [u["member"] for u in lib_usage if u["type"] == "access"]
    assert "utils" not in access_members


def test_iife_and_complex_callees():
    """Test immediately invoked function expressions and other complex callees."""
    code = """
import lib from 'testlib';

(function() { return lib; })();
(() => lib)();
(lib.getFunc())();
"""
    result = analyse_javascript_code(code)
    assert result.valid is True


def test_scoped_package_without_subpath():
    """Test scoped package imported directly without subpath."""
    code = """
import pkg from '@scope';

pkg.method();
"""
    result = analyse_javascript_code(code)
    assert result.valid is True
    assert "@scope" in result.ext_libs
