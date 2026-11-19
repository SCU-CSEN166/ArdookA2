# CSEN 166 — Term Team Project for Social Impact (TTP4SI)

**Team:** A2  
**Challenge:** Ardook
**Team members:** Seerat Kakar, Jason Wu, Michael Tiburcio
**Repository:** https://github.com/SCU-CSEN166/ArdookA2
**Last updated:** Nov 5, 2026

> **How to use this README:** Maintain this single file throughout the project. Complete the Week 5 project plan first, then update the appropriate weekly section before each milestone. Be specific and honest: describe what you attempted, what worked, what did not work, and what evidence supports your conclusions. For every week, document each member's actual contribution. If a task was not completed, explain the blocker and next step. Do not include private or sensitive data in this public README.

## Week 5 — Scope: Project Plan

**Deliverable during Week 6 Lab:** A project plan (write in this README file) describing your problem formulation, proposed approach, each team member's responsibilities, proposed baseline models with Hugging Face links, proposed evaluation metrics, proposed data with links, and goals for each week.

### 1. Problem formulation and social impact

- **Problem:** Nonprofits spend a lot of time figuring out which grants they can actually apply for. Finding grant listings isn’t the hard part. The hard part is reading each one closely enough to know whether they’re eligible, whether it’s worth their time, and when it’s due. We’re building a system that takes a nonprofit’s profile and a list of grants, determines eligibility and fit, and produces recommendations grounded in evidence from the original grant sources.
- **Intended users:** The person at a small nonprofit who handles grants.
- **Inputs and outputs:** The system takes in a nonprofit's profile and a list of grants, and it returns a ranked list where each grant is marked eligible, not eligible, or unsure, along with a short reason and quotes from the source that back it up.
- **Success criteria:** A grant writer should be able to trust our shortlist without re-reading every grant page. That means the system rarely marks a grant eligible when it isn't, and it says "unsure" when the information is missing instead of guessing.
- **Constraints and risks:** The main risk is the AI making things up, like eligibility rules or deadlines that aren't actually in the grant. The grant data is also messy, and some links may be outdated. Telling an org they're eligible when they're not is worse than missing a grant, since they could waste days on an application.

### 2. Proposed technical approach

- **Pipeline:** This pipeline takes a nonprofit’s profile and a spreadsheet of grant opportunities, checks which grants the nonprofit is eligible for, filters out those that don’t meet the requirements, compares the remaining grants to the nonprofit’s needs, and ranks them based on relevance. The ranking system looks at both the full grant description and specific sections to make sure important details aren’t missed. Each recommendation includes a relevance score and supporting evidence. We also use an LLM to give a short explanation of why each grant might be a good fit for the nonprofit.
- **Baseline(s):** To evaluate our pipeline, we'll compare it to a simple keyword-overlap matcher with deterministic eligibility checks. Grants that don't pass requirements like having expired deadlines or incompatible organization types will be excluded when providing recommendations. In addition, if there's missing information in any grant, that'll be marked as unverified. For grants that pass these deterministic checks, they will be ranked by the proportion of nonprofit mission/description keywords which can be found in the grant's own description and requirements. Thus, this would be a good baseline for measuring whether semantic similarity does improve recommendation relevance.
- **Proposed models/tools:**
  - [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — Open source model that converts nonprofit descriptions and grant requirements into embeddings for ranking opportunities using cosine similarity. Selected as a lightweight model suited to semantic search.
  - [FLAN-T5-small](https://huggingface.co/google/flan-t5-small) — Open source model that generates optional explanations of potential fit using retrieved grant evidence. Its explanations will require evaluation and it does not determine eligibility or scores.
- **Alternative approach:** Given that our current pipeline fails, we will try to train our own ML model in order to generate accurate recommendations for any given nonprofit. Currently, we're leveraging similarity search and RAG to see if this implementation can suffice for making sure that nonprofits have the most precise and accurate recommendations for their specific needs.

### 3. Proposed data

| Dataset/source     | Link or access instructions                                                                                 | Purpose    | Size / languages                   | Privacy or usage considerations                           |
| ------------------ | ----------------------------------------------------------------------------------------------------------- | ---------- | ---------------------------------- | --------------------------------------------------------- |
| [Ardook Data]      | [URL](https://docs.google.com/spreadsheets/d/1o_9ft5_eBtUpgBe9rj8TeUIXbJkzKGT7QQRXFxfDqdA/edit?gid=0#gid=0) | [Training] | [reference labels for evaluation]  | [Original data that was given]                            |
| [Ardook Mock Data] | [found in our data folder]                                                                                  | [Training] | [mock data for training + testing] | [Created mock data using the original data we were given] |
| [Ardook Messy Mock Data] | [found in our data folder]                                                                            | [Training] | [more mock data, but messy (blanks, typos, duplicates, vague requirements)] | [Created mock data using the original data we were given] |

- **Data preparation:** [Cleaneed + deduplication]
- **Data limitations:** [We had limited data, so we decided to create a mock version of the original data we were given to increase the number of test cases we can train our solution with]

### 4. Proposed evaluation

| Metric / test                    | What it measures    | How it will be calculated | Target / success criterion |
| -------------------------------- | ------------------- | ------------------------- | -------------------------- |
| Good picks | Are the grants we show actually good ones? | Of the 5 grants we show, count the real matches, divide by 5, and average over all test nonprofits | At least 60%; currently 60% Precision@5 on tracker data |
| Wrong eligibility | Do we ever show a grant the nonprofit can't apply for? | Shown grants that break a known rule, divided by all shown grants | 0% (we're at 0% now) |
| Quotes are real | Are our quotes and deadlines really in the grant source? | Check that each quote appears word for word in the spreadsheet row | 100% |
| Missing info is flagged | Do we say "unknown" when information is missing? | For nonprofits with a missing detail, count the grants marked "unknown" instead of guessed | 100% |

- **Evaluation examples:** We test on fake data made to look like the Ardook spreadsheet as well as on the real data
- **Human verification:** We check a sample of recommendations against the official grant pages. We also read each generated explanation to make sure it only says what's in the grant.
- **Failure cases to watch:** The generated explanation says something that isn't in the grant, and the fake data is cleaner than real data, so our scores on it will look better than they should.

### 5. Team responsibilities

| Member           | Primary responsibility                       | Specific Weekly contribution                                                                                                                                                      | Planned next task                                                                                                  |
| ---------------- | -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Jason Wu         | Build the initial RAG pipeline prototype     | Completed the initial prototype, including grant data ingestion and cleaning, deterministic eligibility checks, semantic similarity scoring, and optional generated explanations. | Begin productionizing the system by scaling the pipeline to process requests concurrently for multiple nonprofits. |
| Michael Tiburcio | Add a vector database and fake training data | Saved the grant embeddings in a vector database so they aren't rebuilt every run. Made a fake dataset modeled on the Ardook spreadsheet.                                          | Measure how good the recommendations are on the fake data and improve them.                                        |
| Seerat Kakar     | Evaluate recommendation rankings             | Added a comparison of semantic and keyword rankings using the tracker Fit ratings and synthetic labels, plus a recommendation command.                                            | Verify tracker ratings and improve semantic model performance.                                                     |

---

## Week 6 — Build: Working Baseline

**Deliverable during Week 7 Lab:** A runnable baseline, setup/run instructions, and 1–2 example inputs and outputs. Update this section with what worked, what failed, and individual contributions.

### Implementation status

- **What runs end-to-end now?** The system takes an organization's description and attributes, removes grants that fail a known eligibility rule, ranks the rest with semantic search and returns the top 5 grants. Each result has a score, its eligibility checks, quotes from the grant row and a source link. Eligibility is always shown as "needs verification", because the system never claims an organization definitely qualifies. Grant embeddings are saved in a Chroma vector store, so they are not rebuilt on every run.
- **How to run it:**
```
cd ArdookA2
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make synthetic
make tracker DESCRIPTION="Ardook is an early-stage, woman-led U.S. startup building AI, physical-AI and data products with a social-impact mission."
make recommend DESCRIPTION="Ardook is an early-stage, woman-led U.S. startup building AI, physical-AI and data products with a social-impact mission."
python -m src.evaluate_safety
```
- Optional LoRA fit judge (CUDA GPU required): train the adapter with `notebooks/deepseek_lora_evaluation.ipynb`, then run `make recommend-synthetic RANKER=lora` or add `--ranker lora` to `python -m src.recommend`.
- **Model(s) and data actually used:**
  - Models: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) turns grants and organization descriptions into embeddings, which are stored in ChromaDB. [FLAN-T5-small](https://huggingface.co/google/flan-t5-small) writes the short explanation for each recommendation and does not decide eligibility.
  - Data: the real Ardook grant tracker, a fake dataset of 150 grants and 200 organizations, and a messy copy of the fake data with blanks, typos and duplicates.
- **Example 1:** Input: "Ardook is an early-stage, woman-led U.S. startup building AI, physical-AI and data products with a social-impact mission." → Output: NVIDIA Inception, Oasis Women in AI Pitch Competition, AI for Good Innovation Factory, Humanity AI, Google for Startups Accelerator (3 of 5 rated Strong/Excellent)
- **Example 2:** Input: "Our nonprofit works on rapid rehousing and eviction prevention for families and is looking for early funding to grow." → Output: Marigold Foundry Fund, Ember Signal Grant, Atlas Impact Initiative, Orchard Bridge Initiative, Cedar Community Program (all marked "needs verification")

### Attempts and outcomes

| Attempt / change | Worked? (Yes / Partly / No) | Evidence (output, metric, link) | What we learned / next step |
| ---------------- | --------------------------- | ------------------------------- | --------------------------- |
| Semantic search vs keyword matching on fake data | Yes | Precision@5: 47.0% vs 36.9% | Semantic search did better, but it's still under our 60% goal |
| Semantic search on the real tracker | Partly | 3 of 5 good picks (60%), while keyword matching got 4 of 5 (80%) | Keywords caught words like "AI" and "women" that semantic search missed. Next, try using both together |
| Eligibility rules from the workbook | Yes | 0 ineligible grants shown | The rules keep wrong grants out, so ranking can just focus on fit |
| Testing on the messy data | Partly | All 20 safety checks passed, but 65 organizations got the same grant twice in their top 5 | Duplicate listings take up spots. Next, remove duplicates |

### Individual contributions during that week

| Member           | What I personally implemented/tested                                                                                                                                                                                                                                    | Result (worked / partly / failed) | Evidence (commit, PR, experiment)        | Blocker / next step                                                                                     |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Seerat Kakar     | I updated the ranking to compare an organization’s profile with the whole grant and its individual requirements. It now reuses saved embeddings and lets us set a minimum similarity score. I also added commands to run recommendations and compare them with the keyword baseline. | Worked | `bd643cae2a584f40487374cfdb4179c8013260ac` | Improve ranking performance by refining how grant relevance scores are calculated. |
| Jason Wu | I updated the real data evaluator to use an explicit applicant profile and integrated workbook based eligibility rules for 6 programs into evaluations and recommendations. I also changed how conditional fit-labels were handled and CLI commands worked as intended. | Finished                          | 8ea4a922d27308a331ad788780288004b504f6cc | Continue to iterate on the pipeline and make sure that it's as accurate as possible for recommendations |
| Michael Tiburcio  |  I updated our data and focused on making more inconsistent data to test whether or not our current pipelines and evaluations can flag when it is unsure whether the information is correct or not. I am also working on creating a reasoning feature, so it tells us why it is unsure about a recommendation (not enough information or too much guessing | Finished |  aad000250db026deb2ad6c7354e2027ece0916f2 | Continue to iterate on the reasoning part and get to a place where our model can be more consistent regardless of whether our data is clean or not |
---

## Week 7 — Evaluate: Results Checkpoint

**Deliverable during Week 8 Lab:** Preliminary evaluation results, metrics, development-set details, and 2–3 representative successes or failures. Include what you will improve next.

### Evaluation setup

- **Development data:** We tested on three sets. The fake dataset has 150 grants and 200 organizations, labeled by the same rules our eligibility checker uses. The messy dataset is a copy of it with blanks, typos, vague descriptions and duplicate listings. The third set is the Ardook grant data we were given, which already rates each grant's fit, and we count Strong or Excellent as a good pick. We also prepared 160 organization–grant pairs for human review; labeling is still pending.
- **Metrics and rationale:** We use Precision@5 (how many of the 5 grants we show are good matches) and Recall@5 (how many of all the good grants made it into our top 5). A user only sees 5 recommendations, so those 5 are what matter. We also run 20 safety checks, like never showing a grant the organization can't apply for and never showing the same grant twice.
- **Comparison:** The baseline is our Week 6 system, which used semantic search only. We added one change at a time and measured each on the same data.

| System / version | Precision@5 | Recall@5 | Notes |
| ---------------- | ----------: | -------: | ----- |
| Keyword matching only | 36.9% | 29.1% | Starting point from Week 5 |
| Week 6 baseline (semantic search) | 47.0% | 37.7% | Better than keywords, but under our 60% goal |
| + mix keyword matching into the ranking | 53.9% | 41.0% | Biggest gain of the week |
| + remove duplicate listings | 53.9% | 41.0% | Same score on clean data, but on messy data duplicates in the top 5 went from 65 organizations to 0 |
| Tried 5 other embedding models and a reranker | about the same | | No real gain, so we kept all-MiniLM-L6-v2 |
| Week 6 baseline, given Ardook data | 60.0% | 37.5% | 3 of 5 good picks |
| Current, given Ardook data | 80.0% | 50.0% | 4 of 5 good picks |

### Successes, failures, and interpretation

| Example / category | Expected behavior | Actual behavior | Success or failure? | Likely reason |
| ------------------ | ----------------- | --------------- | ------------------- | ------------- |
| Ardook: "early-stage, woman-led U.S. startup building AI, physical-AI and data products with a social-impact mission" | Mostly Strong/Excellent grants in the top 5 | Went from 3 to 4 good picks. Two Moderate grants dropped out and two Excellent ones came in | Success | Keyword matching catches exact words like "AI" and "woman-led" that semantic search missed |
| "free clinics and care navigation for uninsured patients" (messy data) | Each grant shows up once | Before, "Quill Signal Grant" and "quill signal grant" took 2 of the 5 spots. Now it shows once | Success | We now group listings that are almost the same and only keep one |
| "Our mission is to make things better." (messy data) | Say there isn't enough information, or return weak matches | Still returns 5 grants, and only 1 is a good match | Failure | There's nothing in the description to match on, and the system doesn't flag vague input yet |
| "tutoring and digital skills training for underserved students" (messy data) | Education grants in the top 5 | 0 of 5 are good matches | Failure | The messy grant text is blank or full of typos, so there isn't enough clean text to match the organization to the right grants |

- **What improved?** Precision@5 went from 47.0% to 53.9% on the fake data and from 60% to 80% on the given Ardook data. Duplicate listings in the top 5 dropped from 65 organizations to 0, and all 20 safety checks still pass.
- **What did not improve?** The messy data is still only about 15% Precision@5. Vague descriptions like "make things better" still get 5 confident-looking results. Switching embedding models didn't help either.
### Synthetic classifier comparison

- **Development data:** The existing [clean synthetic database](data/ardook_extra_synthetic_data.db) contains 200 fictional organizations, 150 fictional grants, and 30,000 labeled pairs. Labels are generated by [the data generator](src/ardook_extra_synthetic_data.py): `unrelated` means no shared focus; `ineligible` means shared focus with a known rule violation; `match` means shared focus without a known rule violation. Missing applicant facts are tolerated and deadlines are ignored by these labels, so `match` does not establish eligibility. This experiment does not use the still-unlabeled human review workbook.
- **Split:** Seed 42; organization inputs and grant inputs are each split approximately 60/20/20, grouping exact duplicate text together. Only pairs whose organization and grant belong to the same split are retained: 10,710 training pairs (119 organizations / 90 grants), 1,230 validation pairs (41 / 30), and 1,200 test pairs (40 / 30). The other 16,860 cross-split pairs are excluded. No organization/grant IDs or exact input texts cross splits, but generated phrasing and templates can recur. Test class counts are 1,003 unrelated, 143 ineligible, and 54 match.
- **Metrics and rationale:** Accuracy, macro-precision, macro-recall, macro-F1, per-class precision/recall/F1, and confusion matrices. Macro-F1 is the model-selection metric because most pairs are unrelated. Match recall measures whether useful candidates are missed. These are classification metrics, not recommendation Precision@5.
- **Comparison:** A majority baseline, TF-IDF pair features with logistic regression, three frozen sentence encoders with logistic regression, and MiniLM-L6 pair features with a PyTorch classifier. Every model receives the same organization descriptions/facts and grant program/type/requirements/amount text. Generated focus labels, existing Fit ratings, rule-check outputs and reviewer notes are excluded from inputs. TF-IDF vocabulary and embedding-feature scaling are fitted on training data only. Logistic regression uses balanced class weights and C=1. The PyTorch model uses a 64-unit hidden layer, dropout 0.2, balanced cross-entropy, and Adam; its epoch is selected from 20 epochs by validation macro-F1. No sentence encoder is fine-tuned.

Results from the November 12, 2026 run on the same 1,200 test pairs:

| Classifier experiment | Validation macro-F1 | Test accuracy | Test macro-precision | Test macro-recall | Test macro-F1 | Test match recall |
| --------------------- | ------------------: | ------------: | -------------------: | ----------------: | ------------: | ----------------: |
| Majority: always unrelated | 0.306 | 83.6% | 0.279 | 0.333 | 0.304 | 0.0% |
| TF-IDF + logistic regression | 0.488 | 75.4% | 0.517 | 0.649 | 0.549 | 44.4% |
| MiniLM-L6 + logistic regression | 0.430 | 75.3% | 0.481 | 0.553 | 0.503 | 38.9% |
| MiniLM-L6 + PyTorch classifier | **0.531** | **85.7%** | **0.618** | 0.548 | **0.574** | 24.1% |
| MiniLM-L12 + logistic regression | 0.416 | 71.5% | 0.449 | 0.491 | 0.454 | 20.4% |
| MPNet + logistic regression | 0.513 | 82.1% | 0.538 | 0.581 | 0.552 | **48.1%** |

MiniLM-L6 + PyTorch was selected using validation macro-F1 (epoch 19), before using test metrics for comparison. Full metrics, settings, dataset/script hashes, package versions, split IDs, and epoch history are in [report.json](results/model_comparison_v1/report.json). Every test prediction and its input text is in [test_predictions.json](results/model_comparison_v1/test_predictions.json). This is one synthetic split, with shared records across pairs; the figures do not establish real-world accuracy or statistical significance.

Reproduce with the locally cached `sentence-transformers/all-MiniLM-L6-v2`, `all-MiniLM-L12-v2`, and `all-mpnet-base-v2` models:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  .venv/bin/python -m src.compare_models --output results/model_comparison_rerun
```

The script uses CPU, performs no model downloads, and refuses to overwrite an existing output directory. This experiment saves reports and predictions, not deployable checkpoints, and leaves the recommendation pipeline unchanged.

#### Classifier successes, failures, and interpretation

| Example / category | Expected behavior | Actual behavior | Success or failure? | Likely reason |
| ------------------ | ----------------- | --------------- | ------------------- | ------------- |
| Profile 35 / grant row 147: mental-health nonprofit and Harbor Impact Initiative | `match`: mental-health focus and no known generated rule violation | Selected PyTorch classifier predicted `match` | Success against the synthetic label | The grant explicitly supports mental-health services; the supplied revenue is below the cap. |
| Profile 57 / grant row 103: nonprofit and Marigold Community Grant | `ineligible`: the grant requires a for-profit company | Selected PyTorch classifier predicted `match`; TF-IDF classifier predicted `ineligible` | Failure | Shared women's-entrepreneurship language may outweigh the entity restriction. A learned score must not replace explicit eligibility checks. |
| Profile 5 / grant row 104: UK arts nonprofit and Tidewater Horizon Initiative | `match` under the generated rules | Selected PyTorch classifier predicted `unrelated` | Failure | The grant combines arts with women's empowerment; mixed focus wording may obscure the arts match. This is a hypothesis, not a verified cause. |

- **What improved?** The selected MiniLM-L6 + PyTorch classifier reached test macro-F1 0.574 versus 0.503 for MiniLM-L6 + logistic regression and 0.304 for the majority baseline. Merely increasing the encoder size did not consistently help: MiniLM-L12 performed worse than L6, while MPNet improved macro-F1 and match recall over L6 logistic regression. These are classifier comparisons, not improvements measured in the live recommendation pipeline.
- **What did not improve?** The selected classifier found only 13 of 54 true matches, versus 26 for MPNet. Of its 31 predicted matches, seven were labeled ineligible and eleven unrelated. Its higher overall macro-F1 therefore does not make it an acceptable replacement for eligibility checks. The majority baseline's 83.6% accuracy also shows why accuracy alone is inadequate. Synthetic labels sometimes conflict with information implied in descriptions because they use only structured attributes for eligibility; independent human labels are still needed.
- **Next experiment:** Confirm the intended Ardook applicant profile and complete independent human labeling. Compare the MiniLM-L6 classifier and MPNet approach on held-out reviewed pairs, measuring macro-F1, match recall, and false matches. Before integration, compare recommendation Precision@5 on the same candidate sets while retaining deterministic eligibility checks. Use validation data for model selection and reserve test data for the final comparison. The synthetic comparison is complete; human-reviewed evaluation and evidence of improved real recommendations are still pending.

### Individual contributions during that week

| Member | Work completed  | Result (worked / partly / failed) | Evidence | Next step   |
| ------ | --------------- | --------------------------------- | -------- | ----------- |
| Jason Wu | I created the 160-pair human-labeling workflow and logistic regression/PyTorch training scripts. I then implemented and ran six classifier configurations on the existing synthetic data, holding out both organizations and grants. I recorded accuracy, macro-F1, per-class metrics, confusion matrices, and representative failures. | Worked for the synthetic experiment: the validation-selected MiniLM-L6 + PyTorch classifier achieved 85.7% test accuracy and 0.574 macro-F1, but only 24.1% match recall. Human reviews and real-world improvement remain unverified. | [Review workbook](outputs/ranking-review/ranking_review.xlsx), [model comparison script](src/compare_models.py), [full results](results/model_comparison_v1/report.json), [test predictions](results/model_comparison_v1/test_predictions.json) | Complete human-reviewed evaluation, investigate missed matches and entity-rule errors, and compare ranking performance before integration. |
| Michael Tiburcio | I improved the vector store so now it only re-embeds grants that changed and also made ranking based on semantic search with keyword matching and also now ignoring duplicate listings. | Worked: the precision went from 47.0% to 53.9% on our fake data and from 60% to 80% on the real tracker. | PR #6 (feature/vector-store-embeddings) | Fine-tune the embedding model on the human labels once they're done. |
| Seerat Kakar | I added a DeepSeek feature that extracts grant details from the spreadsheet. It checks extracted details against the original text and can use them to help rank grants. | Partly: no visible improvement in matching results yet. | `src/deepseek_extractor.py`, `src/recommend.py` | Evaluate whether DeepSeek-extracted details improve matching, then add a LoRA adapter to score candidate fit. |

---

## Week 8 — Stress-Test: Reliability on Hidden Data

**Deliverable during Week 9 Lab:** Hidden-test results (as provided by the TA), analysis of at least three important failure cases, and a prioritized improvement plan.

- **Submitted system version / commit:** `babbfcd` on main. The website (frontend) uses the match classifier. The pairing-table PyTorch model is saved in `models/pytorch_pairing_v1`.
- **Hidden-test metrics and leaderboard result:** We tested on the 10 hidden grants and checked the answers against our own labels. There's no leaderboard result yet. The pairing-table PyTorch model got an F1 of 0.83 (precision 0.71, recall 1.00). The match classifier got 0.75 (precision 1.00, recall 0.60). The pipeline with rules got 0.91, and the older PyTorch model got 0.57.
- **Difference from development results:** On our fake test data the pairing model got 97.6% right, but on the hidden grants it only got F1 0.83. That makes sense, since the fake labels follow the same rules the model learned. The hidden grants are real and there are only 10 of them, so getting one grant wrong moves the score a lot.

| Failure case / category | What went wrong | Possible cause | Severity / user impact | Proposed fix |
| ----------------------- | --------------- | -------------- | ---------------------- | ------------ |
| Tory Burch Fellows (#3) | The pairing model said Relevant, but the grant needs at least USD 75K in yearly revenue and Ardook is just starting out. | We only catch revenue limits written like "under $50k", so "at least USD 75K" slipped by. | High: we'd tell them to apply for a grant they can't get. | Catch minimum-revenue wording too, using the DeepSeek extractor. |
| D-Prize (#8) | The pairing model said Relevant, but the grant is for groups handing out proven poverty fixes, not AI or data products. | The pairing table only checks organization type, region and woman-led. So it leaves the topic to the text, which got it wrong. | Medium: they'd waste time on a grant that doesn't fit what they build. | Add a requirement for sector or type of project, and add more fake examples of it. |
| AWS Activate (#4) and Women Who Tech (#10) | The match classifier said Not relevant for both, but both are Relevant. It missed 2 of the 5 good grants. | Its cutoff (0.25) is strict, and it gave them low scores (0.03 and 0.18). | Medium: a good grant never shows up. | Lower the cutoff a little, or use it together with the pairing model, which catches everything. |

### Changes attempted after stress-testing

| Change   | Worked? (Yes / Partly / No) | Evidence | Keep, revise, or revert? |
| -------- | --------------------------- | -------- | ------------------------ |
| [Change] | [Status]                    | [Result] | [Decision]               |
| Frontend (`streamlit run app.py`) now scores with the saved match classifier (`results/ardook_match_v1/model.joblib`, threshold 0.25, offline encoder) | Yes | Same scores as `python -m src.predict_match --ardook` on both workbooks; `tests/test_app_predict.py` | Keep |

### Individual contributions during that week

| Member | Work completed  | Result (worked / partly / failed) | Evidence | Next step   |
| ------ | --------------- | --------------------------------- | -------- | ----------- |
| Michael Tiburcio | Built a simple frontend to make testing easier, so you can upload a CSV/Excel file of grants or paste one grant and get Relevant / Not relevant back. Also wrote a parser that handles different column names, blank rows and messy files. To run it, use "pip install -r requirements.txt" then "streamlit run app.py" | Worked |  00163026e64da2453ba5d52a8a364e6363ae6b54 | Add a few more features to the frontend, help improve the PyTorch pipeline, use less strict thresholds so it doesn't reject almost everything, and retrain |
| Jason Wu | I trained a grant-matching classifier on clean synthetic data and real Ardook grants. I compared models and tested it on synthetic, real, and hidden test cases. | Worked: it found 4 of 5 real held-out matches with 2 false positives, and got 8 of 10 hidden test cases correct. | [Training script](src/train_match_classifier.py), [saved report](results/ardook_match_v1/report.json), [hidden-test metrics](results/ardook_match_v1/hidden_test/metrics.csv) | Conduct more evaluation and possible hyperparameter tuning. |
| Seerat Kakar | Added a pairing table feature to the PyTorch model, which crosses each applicant fact (entity type, region, woman-led) with each grant requirement, so the model can learn which combinations make an applicant ineligible. I also generated extra training pairs worded like the real tracker, added Armenia as a region, saved the trained model, and added a LoRA stage to the PyTorch model, which improved validation macro-F1 slightly, from 0.872 to 0.880. | Worked. Result: hidden test F1 0.83 (precision 0.71, recall 1.00) vs the final labels, up from 0.57 for the earlier PyTorch model. | `src/train_pytorch.py`, `src/augment_pairs.py`, `src/train_pairing.py`, `src/pytorch_classifier.py`, `app.py`; commit `f548bc4` | Try replacing the regex matching of grant requirements with the DeepSeek extractor so the pairing table works on new wording, and maybe add more requirement types (startup stage, sector). |

**Running the frontend** (`streamlit run app.py`): the app works offline, so the BGE encoder it uses has to be on your computer first. pip doesn't download it. Run this once while you're online (about 1.2 GB), then start the app:
  ```
  python -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-large-en-v1.5')"
  ```
  If you skip this, the app shows an error saying it can't connect to huggingface.co.

# How to run the app
`cd ~/ArdookA2`
`git fetch origin`
`git worktree add --detach /tmp/ardook-frontend origin/main`
`cd /tmp/ardook-frontend`

`source ~/ArdookA2/.venv/bin/activate`
`pip install -r requirements.txt`
 
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m streamlit run app.py`

---

## Week 9 — Responsible AI, Peer Testing, and Final Delivery

**Deliverable during Week 10 Lab:** Document peer-testing feedback, Responsible AI concerns, changes made in response, and the final runnable system. Submit the final project report as instructed.

### Peer testing

- **Team evaluated:** Ardook3
- **One thing that worked well:** The CPU-only test worked after we installed the 2 Hugging Face models and we were able to get the expected results..
- **One meaningful failure:** We tried testing it on our own data but because it was different structure then what their code expects, it was not able to run.
- **One actionable recommendation:** Making it more reproducible and able to handle unique inputs would make their model stronger.
- **Feedback received from another team:** We need better documentation like how to run and install everything in one section like "How to run" at the end. Otherwise, it's good and model performance is strong.

### Responsible AI assessment

- **Potential harms or affected users:** [Who could be affected and how?]
- **Privacy and data handling:** [How sensitive information is protected]
- **Reliability and unsupported claims:** [How outputs are verified or uncertainty is communicated]
- **Fairness, accessibility, and usability:** [Relevant concerns and tests]
- **Human oversight:** [Where a human should review or correct outputs]
- **Remaining limitations:** [What should not be trusted or deployed yet]

### Final improvements and outcomes

| Feedback / issue | Change made | Worked? (Yes / Partly / No) | Evidence         | Remaining limitation |
| ---------------- | ----------- | --------------------------- | ---------------- | -------------------- |
| [Issue]          | [Change]    | [Status]                    | [Link or metric] | [Limitation]         |

### Individual contributions during that week

| Member | Final implementation / evaluation / documentation contribution | Result (worked / partly / failed) | Evidence | Reflection / remaining work |
| ------ | -------------------------------------------------------------- | --------------------------------- | -------- | --------------------------- |
| [Name] | [Specific work]                                                | [Status]                          | [Link]   | [Reflection]                |

### Final submission checklist

- [ ] Code and final system/prototype are in the repository
- [ ] Setup and run instructions are complete and tested
- [ ] Data sources, model versions, and dependencies are documented
- [ ] Evaluation results are reported and compared with the baseline
- [ ] Important failures and limitations are explained
- [ ] Hidden-test and peer-feedback findings are addressed
- [ ] Responsible AI considerations and human oversight are discussed
- [ ] Every team member's contributions are documented for every week

**Final commit / release:** [Commit hash or tag]
