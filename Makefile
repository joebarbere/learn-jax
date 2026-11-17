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

lock: ## Re-resolve the images' pinned dependencies (containers/requirements*.lock) with uv, inside the base images
	containers/lock.sh

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

read: ## Read one lesson as a read-only app (code hidden, sliders live): make read N=16
	-@podman rm -f learn-jax >/dev/null 2>&1
	DETACH=1 NAME=learn-jax PORT=$(PORT) $(RUN) marimo run $(MOD)/lesson.py --headless --host 0.0.0.0 --port $(PORT)
	@sleep 3; $(MAKE) --no-print-directory url

BOOK := podman run --rm --security-opt label=disable -v $(CURDIR):/course -w /course -e PYTHONPATH=/course localhost/learn-jax-book
book-image: ## Build the book image (Chromium + Paged.js + KaTeX + fonts; no GPU)
	podman build -t learn-jax-book -f containers/Containerfile.book containers

render: ## Render every lesson (light + dark plots) for the site and book; GPU, ~1–2 h, skips up-to-date lessons
	tools/book/render.sh light
	tools/book/render.sh dark

site: ## Static HTML of every lesson (light + dark) into site/ (after make render); open site/light/index.html
	$(BOOK) python tools/book/build.py

pdf: ## The book, "Arrays to Agents": site/arrays-to-agents.pdf and -dark.pdf (after make render)
	$(BOOK) python tools/book/build.py --pdf

glossary: ## Regenerate docs/GLOSSARY.md from learnjax/glossary.py (on the host, with uv)
	uv run --no-dev python tools/gen_glossary.py

rosetta: ## Regenerate docs/ROSETTA.md (math ↔ Python ↔ NumPy ↔ JAX ↔ PyTorch) from learnjax/rosetta/
	$(RUN) python tools/gen_rosetta.py

rosetta-browse: ## Browse the math ↔ code tabs for every operation in marimo
	-@podman rm -f learn-jax >/dev/null 2>&1
	DETACH=1 NAME=learn-jax PORT=$(PORT) $(RUN) marimo run tools/rosetta_browse.py --headless --host 0.0.0.0 --port $(PORT)
	@sleep 3; $(MAKE) --no-print-directory url

md: ## Export every lesson to Markdown (prose + code, no outputs) into markdown/
	@mkdir -p markdown
	@for d in modules/*/; do n=$$(basename $$d); [ -f $$d/lesson.py ] || continue; \
	  $(RUN) marimo export md $$d/lesson.py -o /course/markdown/$$n.md >/dev/null 2>&1 && echo "markdown/$$n.md" || echo "FAILED $$n"; done

test: ## Grade your exercises for one module: make test N=03
	$(RUN) python -m pytest $(MOD)

progress: ## Show the syllabus with your progress
	@$(RUN) python -m learnjax.progress

check: ## Course self-check: every lesson runs headless and every test passes on the reference solutions
	LEARNJAX_SOLUTIONS=1 $(RUN) python -m pytest
	@for f in modules/*/lesson.py; do echo "== $$f"; LEARNJAX_SOLUTIONS=1 $(RUN) python $$f >/dev/null || exit 1; done
	@if podman image exists localhost/learn-jax-torch; then \
	  for f in $$(ls modules/*/torch_compare.py 2>/dev/null); do echo "== $$f (torch)"; IMAGE=localhost/learn-jax-torch $(RUN) python $$f >/dev/null || exit 1; done; \
	else echo "== skipping the PyTorch side-by-side notebooks: no learn-jax-torch image (make build-torch to include them)"; fi
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

lms-build: ## Build the LM Studio (llmster) headless image (module 27)
	containers/lmstudio.sh build

lms-up: ## Start LM Studio headless on http://127.0.0.1:1234 [BACKEND=vulkan|rocm]
	containers/lmstudio.sh start $(BACKEND)

lms-down: ## Stop LM Studio
	containers/lmstudio.sh stop

export-models: ## Export your models (16, 20, 26) + SmolLM2 to HF and GGUF, verify, import into LM Studio
	LEARNJAX_SOLUTIONS=1 $(RUN) python modules/27-lmstudio/export.py
	IMAGE=localhost/learn-jax-torch $(RUN) python modules/27-lmstudio/verify_hf.py
	modules/27-lmstudio/build_gguf.sh
	modules/27-lmstudio/import_models.sh

lms-bench: ## Same model four ways: JAX vs llama.cpp vs LM Studio Vulkan/ROCm (module 27)
	modules/27-lmstudio/bench.sh

PROFILE ?= gpu1
aws-plan: ## Plan an AWS session (no cost): make aws-plan PROFILE=gpu1|gpu1-l40s|multihost|gpu4|trn
	terraform -chdir=infra/aws init -input=false >/dev/null
	terraform -chdir=infra/aws plan -input=false -var profile=$(PROFILE) -out=$(PROFILE).tfplan

aws-howto: ## Print the owner-run commands for a paid AWS session (apply / run / destroy)
	@echo "Review the plan from 'make aws-plan PROFILE=$(PROFILE)', then (this costs money):"
	@echo "  terraform -chdir=infra/aws apply $(PROFILE).tfplan"
	@echo "  infra/aws/run_remote.sh"
	@echo "  terraform -chdir=infra/aws destroy -var profile=$(PROFILE)"
	@echo "Instances also terminate themselves after max_minutes (default 120). See infra/aws/README.md."

bench: ## Single-device benchmark on this machine (module 25 local column), writes runs/25-aws/
	$(RUN) python modules/25-aws/bench_single.py

# an aws-ai checkout next to this repo, if any; otherwise rag-data fetches the corpus from arXiv
AWS_AI_CORPUS ?= $(abspath ../aws-ai/local-kb/.state/corpus)
rag-data: ## Module 38 corpus: copy aws-ai's radio abstracts into the cache, else fetch them from arXiv (~15 min)
	@if [ -f "$(AWS_AI_CORPUS)/radio-abstracts.jsonl" ]; then \
	  echo "copying from $(AWS_AI_CORPUS)"; \
	  podman run --rm --security-opt label=disable -v learn-jax-cache:/cache -v "$(AWS_AI_CORPUS)":/src:ro \
	    localhost/learn-jax sh -c 'mkdir -p /cache/learnjax/radio && cp /src/radio-abstracts.jsonl /cache/learnjax/radio/ && wc -l /cache/learnjax/radio/radio-abstracts.jsonl'; \
	else \
	  $(RUN) python tools/fetch_radio.py; \
	fi

papers: ## Check every paper link and regenerate docs/PAPERS.md
	$(RUN) python tools/check_papers.py
	$(RUN) python tools/gen_papers.py

lineage: ## Check the model family tree (registry rules, arXiv titles, HF configs), regenerate docs/LINEAGE.md
	$(RUN) python tools/check_lineage.py
	$(RUN) python tools/gen_lineage.py

links: ## Link the first JAX API mentions to the docs, then check every external link
	$(RUN) python tools/link_jax_docs.py modules/*/lesson.py
	$(RUN) python tools/check_links.py

lint: ## Ruff
	$(RUN) ruff check learnjax modules tools tests

.PHONY: rosetta rosetta-browse help build lock build-torch up url down lesson torch-lesson read md test progress check shell papers lineage links lint aws-plan aws-howto bench lms-build lms-up lms-down export-models lms-bench llama-download llama-up llama-down llama-bench rag-data book-image render site pdf glossary
