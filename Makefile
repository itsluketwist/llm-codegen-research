# simple make file for common commands

new: new-venv new-lint

check: lint test

new-venv:
	python -m venv .venv
	. venv/bin/activate

new-lint:
	pre-commit install
	pre-commit autoupdate

update-lint:
	pre-commit autoupdate

compile:
	uv pip compile requirements.txt -o requirements.lock

lint:
	pre-commit run --all-files

test:
	uv run pytest

test-local:
	uv run pytest -m "not api"

test-api:
	uv run pytest -m "api"

test-coverage:
	uv run pytest --cov=src/ --cov-report=term-missing

test-badge:
	uv run pytest --cov=src/ --cov-report=xml ; genbadge coverage -i coverage.xml -o coverage.svg ;

