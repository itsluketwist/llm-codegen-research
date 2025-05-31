# simple make file for common commands

new: newvenv newlint

check: lint test

newvenv:
	python -m venv .venv
	. venv/bin/activate

newlint:
	pre-commit install
	pre-commit autoupdate

compile:
	uv pip compile requirements.txt -o requirements.lock

lint:
	pre-commit run --all-files

test:
	uv run pytest

testlocal:
	uv run pytest --ignore=tests/test_llm_api.py

testapi:
	uv run pytest tests/test_llm_api.py

coverage:
	uv run pytest --cov=src/ --cov-report=term-missing

badge:
	uv run pytest --cov=src/ --cov-report=xml ; genbadge coverage -i coverage.xml -o coverage.svg ;

