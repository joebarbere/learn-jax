# learn-jax — everything runs in the container (containers/run.sh); nothing is installed on the host.
.DEFAULT_GOAL := help
# Join the llama.cpp network automatically once it exists (make llama-up creates it).
NETWORK ?= $(shell podman network exists learn-jax-net 2>/dev/null && echo learn-jax-net)
RUN  := NETWORK=$(NETWORK) containers/run.sh
PORT ?= 2718
N    ?= 00
MOD   = $(shell ls -d modules/$(N)-* 2>/dev/null | head -1)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build: ## Build the course image (JAX + ROCm + marimo)
	podman build -t learn-jax -f containers/Containerfile containers

build-torch: ## Build the PyTorch side-by-side image
	podman build -t learn-jax-torch -f containers/Containerfile.torch containers

up: ## Start marimo on http://127.0.0.1:$(PORT) in the background (then: make url)
	-@podman rm -f learn-jax >/dev/null 2>&1
	DETACH=1 NAME=learn-jax PORT=$(PORT) $(RUN) marimo edit modules --headless --host 0.0.0.0 --port $(PORT)
	@sleep 3; $(MAKE) --no-print-directory url

url: ## Print the marimo URL (with access token)
	@podman logs learn-jax 2>&1 | grep -oE 'http://[^ ]*' | tail -1 | sed -E 's|//[^:/]+:|//127.0.0.1:|'

down: ## Stop marimo
	-podman rm -f learn-jax

lesson: ## Open one lesson in marimo: make lesson N=03
	-@podman rm -f learn-jax >/dev/null 2>&1
	DETACH=1 NAME=learn-jax PORT=$(PORT) $(RUN) marimo edit $(MOD)/lesson.py --headless --host 0.0.0.0 --port $(PORT)
	@sleep 3; $(MAKE) --no-print-directory url

torch-lesson: ## Open a PyTorch side-by-side notebook: make torch-lesson N=10
	-@podman rm -f learn-jax >/dev/null 2>&1
	IMAGE=localhost/learn-jax-torch DETACH=1 NAME=learn-jax PORT=$(PORT) $(RUN) marimo edit $(MOD)/torch_compare.py --headless --host 0.0.0.0 --port $(PORT)
	@sleep 3; $(MAKE) --no-print-directory url

test: ## Grade your exercises for one module: make test N=03
	$(RUN) python -m pytest $(MOD)

progress: ## Show the syllabus with your progress
	@$(RUN) python -m learnjax.progress

check: ## Course self-check: every lesson runs headless and every test passes on the reference solutions
	LEARNJAX_SOLUTIONS=1 $(RUN) python -m pytest
	@for f in modules/*/lesson.py; do echo "== $$f"; LEARNJAX_SOLUTIONS=1 $(RUN) python $$f >/dev/null || exit 1; done
	@for f in $$(ls modules/*/torch_compare.py 2>/dev/null); do echo "== $$f (torch)"; IMAGE=localhost/learn-jax-torch $(RUN) python $$f >/dev/null || exit 1; done
	@echo "all lessons ran"

shell: ## Shell in the container
	$(RUN) bash

BACKEND ?= vulkan
llama-download: ## Download the module 19 GGUF models (~12 GB)
	containers/llamacpp.sh download

llama-up: ## Start the llama.cpp server: make llama-up [BACKEND=vulkan|rocm] [MODEL=...]
	containers/llamacpp.sh start $(BACKEND) $(MODEL)

llama-down: ## Stop the llama.cpp server
	containers/llamacpp.sh stop

llama-bench: ## Benchmark models x backends (module 19), writes runs/19-local-serving/bench.json
	python3 modules/19-local-serving/bench.py

papers: ## Check every paper link and regenerate docs/PAPERS.md
	$(RUN) python tools/check_papers.py
	$(RUN) python tools/gen_papers.py

lint: ## Ruff
	$(RUN) ruff check learnjax modules

.PHONY: help build build-torch up url down lesson torch-lesson test progress check shell papers lint llama-download llama-up llama-down llama-bench
