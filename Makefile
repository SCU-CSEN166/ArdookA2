PYTHON ?= .venv/bin/python
DESCRIPTION ?=
ATTRIBUTES ?= {}

.PHONY: help tracker synthetic recommend recommend-synthetic

help:
	@echo "make tracker             Evaluate tracker rankings (requires DESCRIPTION)"
	@echo "make synthetic           Evaluate rankings on generated data"
	@echo "make recommend           Show real-tracker recommendations (requires DESCRIPTION)"
	@echo "make recommend-synthetic Show recommendations for a sample generated profile"

tracker:
	$(PYTHON) -m src.evaluate_relevance --dataset tracker --description "$(DESCRIPTION)" --attributes '$(ATTRIBUTES)'

synthetic:
	$(PYTHON) -m src.evaluate_relevance --dataset synthetic

recommend:
	$(PYTHON) -m src.recommend --dataset real --description "$(DESCRIPTION)" --attributes '$(ATTRIBUTES)'

recommend-synthetic:
	$(PYTHON) -m src.recommend --dataset synthetic --profile-id 1
