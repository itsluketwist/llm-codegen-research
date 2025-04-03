# **python-template**


![lint code workflow](https://github.com/itsluketwist/python-template/actions/workflows/lint.yaml/badge.svg)
![test code workflow](https://github.com/itsluketwist/python-template/actions/workflows/test.yaml/badge.svg)
![release workflow](https://github.com/itsluketwist/python-template/actions/workflows/release.yaml/badge.svg)


<div>
    <!-- badges from : https://shields.io/ -->
    <!-- logos available : https://simpleicons.org/ -->
    <a href="https://opensource.org/licenses/MIT">
        <img alt="MIT License" src="https://img.shields.io/badge/Licence-MIT-yellow?style=for-the-badge&logo=docs&logoColor=white" />
    </a>
    <a href="https://www.python.org/">
        <img alt="Python 3" src="https://img.shields.io/badge/Python_3-blue?style=for-the-badge&logo=python&logoColor=white" />
    </a>
</div>


## *usage*

Once cloned, find and replace all instances of `python-template` with the new repository name.
Remove below `README.md` sections where appropriate (whether this is a project or library),
similarly determine whether the `pyproject.toml` or `requirements.txt` files are necessary.

## *installation*

Install directly from GitHub, using pip:

```shell
pip install git+https://github.com/itsluketwist/python-template
```

## *development*

Clone the repository code:

```shell
git clone https://github.com/itsluketwist/python-template.git
```

_(for projects...)_ Once cloned, install the requirements locally in a virtual environment:

```shell
python -m venv .venv

. .venv/bin/activate

pip install -r requirements.txt -r requirements-dev.txt
```

_(for libraries...)_ Once cloned, install the package locally in a virtual environment:

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

Dependencies are managed with [uv](https://astral.sh/blog/uv), add new packages to `requirements.in`, then compile:

```shell
uv pip compile requirements.in -o requirements.txt
```

## *todos*

- Add docs template / support.


## *testing*

Run the test suite using:

```shell
pytest .
```


## *inspiration*

This is currently how I like to make python projects/libraries, it ain't that deep.
