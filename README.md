# **llm-codegen-research**


![lint code workflow](https://github.com/itsluketwist/llm-codegen-research/actions/workflows/lint.yaml/badge.svg)
![test code workflow](https://github.com/itsluketwist/llm-codegen-research/actions/workflows/test.yaml/badge.svg)
![release workflow](https://github.com/itsluketwist/llm-codegen-research/actions/workflows/release.yaml/badge.svg)


<div>
    <!-- badges from : https://shields.io/ -->
    <!-- logos available : https://simpleicons.org/ -->
    <a href="https://opensource.org/licenses/MIT">
        <img alt="MIT License" src="https://img.shields.io/badge/Licence-MIT-yellow?style=for-the-badge&logo=docs&logoColor=white" />
    </a>
    <a href="https://www.python.org/">
        <img alt="Python 3" src="https://img.shields.io/badge/Python_3-blue?style=for-the-badge&logo=python&logoColor=white" />
    </a>
    <a href="https://openai.com/blog/openai-api/">
        <img alt="OpenAI API" src="https://img.shields.io/badge/OpenAI_API-412991?style=for-the-badge&logo=openai&logoColor=white" />
    </a>
    <a href="https://www.anthropic.com/api/">
        <img alt="Anthropic API" src="https://img.shields.io/badge/Claude_API-D97757?style=for-the-badge&logo=claude&logoColor=white" />
    </a>
    <a href="https://api.together.ai/">
        <img alt="together.ai API" src="https://img.shields.io/badge/together.ai_API-B5B5B5?style=for-the-badge&logoColor=white" />
    </a>
</div>


## *usage*

A collection of methods and classes I repeatedly use when conducting research on LLM code-generation.
Covers both prompting various LLMs, and analysing the markdown responses.

```python
from llm_cgr import api, md

response = api.quick_complete("Write python code to generate the nth fibonacci number.")

markdown = md.Markdown(response)
```

## *installation*

Install directly from GitHub, using pip:

```shell
pip install git+https://github.com/itsluketwist/llm-codegen-research
```

## *development*

Clone the repository code:

```shell
git clone https://github.com/itsluketwist/llm-codegen-research.git
```

Once cloned, install the package locally in a virtual environment:

```shell
python -m venv .venv

. .venv/bin/activate

pip install -e ".[dev]"
```

Install and use pre-commit to ensure code is in a good state (uses [ruff](https://astral.sh/ruff)):

```shell
pre-commit install

pre-commit autoupdate

pre-commit run --all-files
```

Dependencies are managed with [`uv`](https://astral.sh/blog/uv), add new packages to `requirements.txt`, then install `uv` and compile:

```shell
uv pip compile requirements.txt -o requirements.lock
```

## *testing*

*todos*

Run the test suite using:

```shell
pytest .
```
