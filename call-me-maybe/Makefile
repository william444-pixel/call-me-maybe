PYTHON = python
MODULE = src
SRC_DIR = src
MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

all:run

install:
	uv sync

run:
	uv run $(PYTHON) -m $(MODULE)

debug:
	uv run $(PYTHON) -m pdb -m $(MODULE)

clean:
	find . -type d \( -name "__pycache__" -o -name ".mypy_cache" \) -exec rm -rf {} +

lint:
	flake8 $(SRC_DIR)
	uv run mypy src $(MYPY_FLAGS)
