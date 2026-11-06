PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python; fi)
PIP ?= $(PYTHON) -m pip
RUFF ?= ruff

.PHONY: help install install-dev install-cnn lint test verify run benchmark-skab clean

help:
	@echo "Targets:"
	@echo "  install       Install runtime dependencies"
	@echo "  install-dev   Install runtime + lint dependencies"
	@echo "  install-cnn   Install TensorFlow for the optional CNN-LSTM path"
	@echo "  lint          Run ruff"
	@echo "  test          Run the unittest suite"
	@echo "  verify        Run lint then tests"
	@echo "  run           Run the default FD001 pipeline"
	@echo "  benchmark-skab Run the standalone SKAB benchmark"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements-dev.txt

install-cnn:
	$(PIP) install -r requirements-cnn.txt

lint:
	$(RUFF) check .

test:
	$(PYTHON) -m unittest discover -s tests -q

verify: lint test

run:
	$(PYTHON) main.py

benchmark-skab:
	$(PYTHON) main.py --benchmark skab

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
