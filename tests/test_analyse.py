from llm_cgr import Markdown


TEST_LLM_RESPONSE = """
Here's a Python solution to process some data and return an answer.

```python
import numpy as np
from requests import get
import json
from collections import defaultdict

def process_data(data):
    response = get("https://api.example.com/data")
    data = np.array([1, 2, 3, 4, 5])
    return np.process(data, response)
```

Some more code:

```
import pandas as pd

csv = pd.read_csv("data.csv")
```

Run some code:

```bash
python script.py
```
"""


def test_markdown():
    """
    Test the MarkdownResponse class, extracting and analysing multiple code blocks.
    """
    # parse the response
    analysed = Markdown(text=TEST_LLM_RESPONSE)

    # check initial properties
    assert analysed.text == TEST_LLM_RESPONSE
    assert len(analysed.code_blocks) == 3
    assert analysed.languages == ["bash", "python"]

    # expected python code block
    python_code_one = analysed.code_blocks[0]
    assert python_code_one.language == "python"
    assert python_code_one.defined_funcs == ["process_data"]
    assert python_code_one.called_funcs == ["get", "np.array", "np.process"]
    assert python_code_one.packages == ["numpy", "requests"]
    assert python_code_one.imports == [
        "collections.defaultdict",
        "json",
        "numpy",
        "requests.get",
    ]

    # unspecified code block defaults to python
    python_code_two = analysed.code_blocks[1]
    assert python_code_two.language == "python"
    assert python_code_two.defined_funcs == []
    assert python_code_two.called_funcs == ["pd.read_csv"]
    assert python_code_two.packages == ["pandas"]
    assert python_code_two.imports == ["pandas"]

    # bash code block with no analysis
    bash_code = analysed.code_blocks[2]
    assert bash_code.language == "bash"
    assert bash_code.defined_funcs == []
    assert bash_code.called_funcs == []
    assert bash_code.packages == []
    assert bash_code.imports == []
