# Makefile

define HELP_MESSAGE
keval

# Installing

1. Create a new Conda environment: `conda create --name keval python=3.12`
2. Activate the environment: `conda activate keval`
3. Install the package: `make install-dev`

# Running Tests

1. Run autoformatting: `make format`
2. Run static checks: `make static-checks`
3. Run unit tests: `make test`

endef
export HELP_MESSAGE

all:
	@echo "$$HELP_MESSAGE"
.PHONY: all

# ------------------------ #
#        PyPI Build        #
# ------------------------ #

install-dev:
	@pip install --verbose -e '.[dev]'

install:
	@kol run onshape link TODO --robot_path keval/resources/gpr/
	@pip install --verbose -e .

build-for-pypi:
	@pip install --verbose build wheel twine
	@python -m build --sdist --wheel --outdir dist/ .
	@twine upload dist/*
.PHONY: build-for-pypi

push-to-pypi: build-for-pypi
	@twine upload dist/*
.PHONY: push-to-pypi

# ------------------------ #
#       Static Checks      #
# ------------------------ #

py-files := $(shell find . -name '*.py')

format:
	@isort --profile black .
	@ruff format .
.PHONY: format

static-checks:
	@isort --profile black --check --diff  .
	@ruff check .
	@mypy --install-types --non-interactive .
.PHONY: lint

# ------------------------ #
#        Unit tests        #
# ------------------------ #

test:
	python -m pytest
.PHONY: test
