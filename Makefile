# Sextant. `make install && make test`

PY      ?= .venv/bin/python
VENV_PY ?= python3.12          # NOT python3: that is 3.6 on some machines

.PHONY: help install test bench bench-scaling bench-public demo clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'

install: ## create .venv and install numpy plus pytest
	$(VENV_PY) -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements-dev.txt

test: ## the whole suite
	$(PY) -m pytest -q

bench: ## the recall and speed curve at one corpus size
	$(PY) -m bench.curve --n 20000 --dim 128 --queries 300

bench-scaling: ## graph against exact search as the corpus grows (slow)
	$(PY) -m bench.scaling --sizes 5000,20000,40000 --dim 128 --target 0.90

bench-public: ## recall on real GloVe embeddings (downloads 121 MB once, needs h5py)
	$(PY) -m pip install -q h5py
	@test -f bench/.data/glove-25-angular.hdf5 || (mkdir -p bench/.data && curl -L -o bench/.data/glove-25-angular.hdf5 http://ann-benchmarks.com/glove-25-angular.hdf5)
	$(PY) -m bench.public --n 20000 --queries 500

demo: ## build a small index and print the curve
	$(PY) -m sextant.cli demo --n 5000 --dim 64

clean:
	rm -rf .venv .pytest_cache **/__pycache__ *.npz
