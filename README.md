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
from llm_cgr import generate, Markdown

response = generate("Write python code to generate the nth fibonacci number.")

markdown = Markdown(text=response)
```

## *installation*

Install directly from PyPI, using pip:

```shell
pip install llm-codegen-research
```

## *development*

Clone the repository code:

```shell
git clone https://github.com/itsluketwist/llm-codegen-research.git
```

We use [`uv`](https://astral.sh/blog/uv) for project management.
Once cloned, create a virtual environment and install uv and the project:

```shell
python -m venv .venv

. .venv/bin/activate

pip install uv

uv sync
```

Use `make` commands to lint and test:

```shell
make lint

make test
```

Use `uv` to add new dependencies into the project and `uv.lock`:

```shell
uv add openai
```
