.PHONY: install test lint check run

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check .

check: lint test

run:
	python -m newslens
