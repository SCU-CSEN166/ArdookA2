# CSEN 166 — Term Team Project for Social Impact (TTP4SI)

**Team:** A2  
**Challenge:** Ardook
**Team members:** Seerat Kakar, Jason Wu, Michael Tiburcio
**Repository:** https://github.com/SCU-CSEN166/ArdookA2
**Last updated:** Nov 5, 2026

> **How to use this README:** Maintain this single file throughout the project. Complete the Week 5 project plan first, then update the appropriate weekly section before each milestone. Be specific and honest: describe what you attempted, what worked, what did not work, and what evidence supports your conclusions. For every week, document each member's actual contribution. If a task was not completed, explain the blocker and next step. Do not include private or sensitive data in this public README.

## Remember - Best Practices

### Human-reviewed relevance classifier pilot

The review workbook is `outputs/ranking-review/ranking_review.xlsx`. Its **Guide**
tab defines the labels; fill the amber columns in **Reviews**. The 160 pairs use
eight explicitly fictional applicants and the real tracker snapshot dated
November 12, 2026. Labels are blank until a human reviews them. Grant websites
have not been verified for this dataset.

Judge relevance as `strong_fit`, `possible_fit`, `poor_fit`, or
`insufficient_information`. Record eligibility separately as `ineligible`,
`unknown`, or `no_known_conflict`; the last value does not confirm eligibility.
An expired grant can still have strong mission fit. Include a reviewer, review
date, rationale, and an exact supplied grant quote when marking ineligible.
Missing facts are unknown, not a mismatch. First have two reviewers independently
label 20 pairs in separate copies, discuss differences, and record agreed labels
in one canonical workbook. Do not manufacture labels to meet training thresholds.

The frozen source dataset is `data/ranking_review/pairs.jsonl`, with workbook
hash, source rows, source URLs and reference date. Each organization has ten
keyword-selected and ten randomly selected candidates, shuffled for review.
Existing tracker Fit ratings and applicant-specific advice are withheld.
Profiles are assigned to train (100 pairs), validation (20), or test (40)
before labeling. Organizations never cross splits; grants can appear in multiple
splits. Keep the test labels out of model tuning. This small sample cannot measure
generalization to new grants or full-catalog recall.

```bash
# Use the existing project environment; install requirements if needed.
.venv/bin/python -m pip install -r requirements.txt

# Check review progress and label validity without training.
.venv/bin/python -m src.train_ranking_classifier \
  --reviews outputs/ranking-review/ranking_review.xlsx --validate-only

# After human review, train and evaluate the offline baseline.
.venv/bin/python -m src.train_ranking_classifier \
  --reviews outputs/ranking-review/ranking_review.xlsx

```

Training requires at least 30 reviewed training pairs with five examples of each
of the three relevance classes, plus ten reviewed pairs in each held-out split.
These are starter checks, not evidence that the dataset is large enough. Review
all 160 where possible; insufficient-information rows are excluded from training.
If a class is scarce, collect more examples rather than changing truthful labels.

The model uses TF-IDF features for organization and grant text, shared terms and
text differences, with balanced logistic regression. Vocabulary is fitted only
on training data. Reviewer notes, human labels, split IDs, existing Fit ratings
and eligibility decisions are excluded from features. Outputs are
`models/relevance_v1/model.joblib` and `report.json`, including held-out macro-F1,
per-class metrics, confusion matrices and a majority-class baseline. Use a new
`--output` directory for each experiment; existing runs are never overwritten.
Only load model files you created or trust.

This is an offline relevance experiment; it does not change recommendations or
replace deterministic eligibility checks. Compare its ranking performance with
the existing pipeline before integrating it. No trained model or quality claim
is available until actual human reviews are supplied.

A minimal PyTorch alternative uses the same review data and TF-IDF pair features
with a 64-unit hidden layer and three relevance outputs. It runs on CPU, selects
the best epoch by validation macro-F1, and evaluates the test set afterward:

```bash
.venv/bin/python -m src.train_pytorch \
  --reviews outputs/ranking-review/ranking_review.xlsx --epochs 20
```

It saves `model.pt`, `features.joblib`, and `report.json` in `models/pytorch_v1/`.
The checkpoint includes architecture dimensions and label order. Reload it with
`torch.load(path, weights_only=True)`, construct `build_model(input_dim, hidden_dim)`
from `src.train_pytorch`, load its `state_dict`, and call `model.eval()`. Transform
new `pair_texts` with the saved features before passing float32 tensors to the model.
This is starter code for the small review dataset, not a demonstrated improvement
over logistic regression. Human reviews are required before either model can train.

To reproduce the source sample in a new file (never overwrite reviewed data):
`python -m src.prepare_ranking_review --as-of 2026-11-12 --output /tmp/pairs.jsonl`.
The same workbook and profiles produce the same sample. The workbook joins labels
to frozen records by `pair_id`; editing displayed source text does not change the
training inputs. Source corrections require a new dataset version and review.

### Data access and privacy

[Explain where authorized data can be accessed. Do not commit personal information, private WhatsApp exports, API keys, credentials, or restricted grant/application records. Use synthetic or anonymized examples in the README.]

### Known limitations

[Keep a current list of what the system cannot reliably do.]

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

- **What runs end-to-end now?** [Describe exactly.]
- **How to run it:** [Commands, dependencies, configuration; refer to Setup below.]
- **Model(s) and data actually used:** [Names, versions, links, and any changes from the plan.]
- **Example 1:** Input: [non-sensitive example] → Output: [result]
- **Example 2:** Input: [non-sensitive example] → Output: [result]

### Attempts and outcomes

| Attempt / change | Worked? (Yes / Partly / No) | Evidence (output, metric, link) | What we learned / next step |
| ---------------- | --------------------------- | ------------------------------- | --------------------------- |
| [Attempt]        | [Status]                    | [Evidence]                      | [Learning]                  |

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

- **Development data:** [Source, size, selection, annotations, link or access instructions]
- **Metrics and rationale:** [Why these metrics reflect the intended use]
- **Comparison:** [Baseline versus current approach, if different]

| System / version | Metric 1 | Metric 2 | Notes   |
| ---------------- | -------: | -------: | ------- |
| Baseline         |  [Value] |  [Value] | [Notes] |
| Current          |  [Value] |  [Value] | [Notes] |

### Successes, failures, and interpretation

| Example / category | Expected behavior | Actual behavior | Success or failure? | Likely reason |
| ------------------ | ----------------- | --------------- | ------------------- | ------------- |
| [Case]             | [Expected]        | [Observed]      | [Status]            | [Hypothesis]  |

- **What improved?** [Evidence]
- **What did not improve?** [Evidence]
- **Next experiment (Jason):** Complete and validate human relevance and eligibility labels for the 160 prepared organization–grant pairs, then train the logistic regression baseline and PyTorch classifier using the fixed organization-level splits. Compare classification performance using held-out macro-F1 and per-class precision/recall. Before integrating either classifier, compare its ranking with the existing recommendation system using Precision@5 on the same reviewed candidate sets, keeping eligibility checks separate. Use validation data for model selection and reserve test data for the final comparison. The labeling workflow and training scripts are implemented; human labeling, classifier evaluation, and evidence of improved recommendations are still pending.

### Individual contributions during that week

| Member | Work completed  | Result (worked / partly / failed) | Evidence | Next step   |
| ------ | --------------- | --------------------------------- | -------- | ----------- |
| Jason Wu | I created a human-labeling workflow with 160 organization–grant pairs using eight fictional applicant profiles and real tracker records. I added separate relevance and eligibility labels, review validation, and organization-level training/validation/test splits. I also implemented a logistic regression baseline and a PyTorch classifier training script with validation-based checkpoint selection and held-out evaluation. | Partly: the review dataset and training workflows are implemented, but the examples still need human labels. No improvement in recommendation quality has been established yet. | [Review workbook](outputs/ranking-review/ranking_review.xlsx), [dataset preparation](src/prepare_ranking_review.py), [review validation and baseline](src/ranking_classifier.py), [PyTorch training](src/train_pytorch.py) | Complete and validate human reviews, train the classifiers, and compare held-out relevance results before integrating a model into recommendations. |
| Michael Tiburcio | I improved the vector store so now it only re-embeds grants that changed and also made ranking based on semantic search with keyword matching and also now ignoring duplicate listings. | Worked: the precision went from 47.0% to 53.9% on our fake data and from 60% to 80% on the real tracker. | PR #6 (feature/vector-store-embeddings) | Fine-tune the embedding model on Jason's human labels once they're done. |

---

## Week 8 — Stress-Test: Reliability on Hidden Data

**Deliverable during Week 9 Lab:** Hidden-test results (as provided by the TA), analysis of at least three important failure cases, and a prioritized improvement plan.

- **Submitted system version / commit:** [Commit hash]
- **Hidden-test metrics and leaderboard result:** [Results, when available]
- **Difference from development results:** [Compare; do not assume the hidden set is identical]

| Failure case / category | What went wrong | Possible cause | Severity / user impact | Proposed fix |
| ----------------------- | --------------- | -------------- | ---------------------- | ------------ |
| [Case 1]                | [Observation]   | [Hypothesis]   | [Impact]               | [Fix]        |
| [Case 2]                | [Observation]   | [Hypothesis]   | [Impact]               | [Fix]        |
| [Case 3]                | [Observation]   | [Hypothesis]   | [Impact]               | [Fix]        |

### Changes attempted after stress-testing

| Change   | Worked? (Yes / Partly / No) | Evidence | Keep, revise, or revert? |
| -------- | --------------------------- | -------- | ------------------------ |
| [Change] | [Status]                    | [Result] | [Decision]               |

### Individual contributions during that week

| Member | Work completed  | Result (worked / partly / failed) | Evidence | Next step   |
| ------ | --------------- | --------------------------------- | -------- | ----------- |
| [Name] | [Specific work] | [Status]                          | [Link]   | [Next step] |

---

## Week 9 — Responsible AI, Peer Testing, and Final Delivery

**Deliverable during Week 10 Lab:** Document peer-testing feedback, Responsible AI concerns, changes made in response, and the final runnable system. Submit the final project report as instructed.

### Peer testing

- **Team evaluated:** [Team ID]
- **One thing that worked well:** [Evidence]
- **One meaningful failure:** [Reproduction steps or example]
- **One actionable recommendation:** [Specific fix]
- **Feedback received from another team:** [Summary]

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
