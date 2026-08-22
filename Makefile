.PHONY: setup lint format test train dashboard clean

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

test:
	uv run pytest -v

train:
	uv run python -m f1_skill

dashboard:
	uv run streamlit run app/streamlit_app.py

clean:
	rm -rf .venv .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
