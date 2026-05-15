.PHONY: test lint format typecheck build publish

test:
	uv run pytest

lint:
	uvx ruff check src/ tests/
	uvx ruff format --check src/ tests/

format:
	uvx ruff format src/ tests/

typecheck:
	uvx ty check src/

build:
	uv build

publish: build
	uv publish
