PYTHON ?= .venv/bin/python
DESCRIPTION ?=
ATTRIBUTES ?= {}
RANKER ?= semantic
# deepseek-extraction runs DeepSeek extraction + recommendations on an SCU Wave GPU over ssh.
WAVE ?= gkakar@login01.wave.scu.edu
WAVE_DIR ?= ardook-deepseek
TOP_K ?= 5

.PHONY: help tracker synthetic recommend recommend-synthetic deepseek-extraction

help:
	@echo "make tracker             Evaluate tracker rankings (requires DESCRIPTION)"
	@echo "make synthetic           Evaluate rankings on generated data"
	@echo "make recommend           Show real-tracker recommendations (requires DESCRIPTION)"
	@echo "make recommend-synthetic Show recommendations for a sample generated profile"
	@echo "Use RANKER=lora on recommend targets to use the optional CUDA LoRA fit judge"
	@echo "make deepseek-extraction Show real-tracker recommendations with DeepSeek extraction on a Wave GPU (requires DESCRIPTION)"

tracker:
	$(PYTHON) -m src.evaluate_relevance --dataset tracker --description "$(DESCRIPTION)" --attributes '$(ATTRIBUTES)'

synthetic:
	$(PYTHON) -m src.evaluate_relevance --dataset synthetic

recommend:
	$(PYTHON) -m src.recommend --dataset real --description "$(DESCRIPTION)" --attributes '$(ATTRIBUTES)' --ranker $(RANKER)

recommend-synthetic:
	$(PYTHON) -m src.recommend --dataset synthetic --profile-id 1 --ranker $(RANKER)

deepseek-extraction: export DESCRIPTION := $(DESCRIPTION)
deepseek-extraction: export ATTRIBUTES := $(ATTRIBUTES)
deepseek-extraction:
	@test -n "$$DESCRIPTION" || { echo "DESCRIPTION is required"; exit 2; }
	@run=$$(printf 'module load PyTorch/2.11.0-Python-3.12; export HF_HOME=$$HOME/.cache/huggingface HF_HUB_OFFLINE=1; .venv/bin/python -m src.recommend --dataset real --description %q --attributes %q --deepseek-extract --top-k %s' "$$DESCRIPTION" "$$ATTRIBUTES" "$(TOP_K)"); \
	ssh $(WAVE) "cd $(WAVE_DIR) && J=\$$(squeue -u \$$USER -p hub -t R -h -o %i | head -n1); if [ -n \"\$$J\" ]; then A=\"--jobid=\$$J --overlap\"; else A=\"--partition=hub --gres=gpu:rtx6000:1 --mem=32G --time=00:30:00\"; fi; srun \$$A bash -lc $$(printf %q "$$run")"
