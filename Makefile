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

coverage:
	uv run pytest --cov=src/
