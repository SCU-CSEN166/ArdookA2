# Ardook grant-match classifier v2: protocol (written before any v2 experiment)

Committed before `src/tune_match_classifier.py` was run. v1 (`results/ardook_match_v1/`,
`results/ardook_match_protocol.md`) is kept unchanged as the historical checkpoint. This file is
not edited after the run except for the "Deviations" section at the end.

## Audit findings that motivate v2 (from v1 artifacts only)

* **Labels.** `week8_hidden_test/final/expected_labels.csv` is the TA file
  (`ta_expected_labels.csv`, removed in `0ea915e`) with `Good`/`Bad` renamed to
  `Relevant`/`Not relevant`; every row agrees. The deleted `expected_labels_estimate.csv` was
  written for a different applicant (an Armenian laundry social enterprise) and is not used.
  Tracker labels are the v1 mapping of `Fit`; conditional ratings stay unscored.
* **Applicant facts differ between earlier comparisons.** The rules pipeline (9/10) used
  `entity_type = for-profit`; the v1 classifier used the README profile with entity type and
  revenue unknown. The TA reasons also assume a for-profit, pre-/low-revenue applicant. v2 keeps
  the documented provisional profile (`region = US`, `woman_led = yes`, others unknown) and
  reports a for-profit sensitivity run separately, never as the default.
* **The rules pipeline is not a general method.** Its hidden-test rules were transcribed by
  program name from the hidden workbook itself, so its 9/10 is not comparable evidence. The
  pairing model's weights (`models/`, git-ignored) are not in the repository, so its F1 0.83
  cannot be reproduced from committed artifacts.
* **v1 failure causes (dev data and post-exposure hidden rows only).** (1) The conflict stage
  learns word weights from 19 real dev grants that each weigh as much as ~52 synthetic pairs;
  generic words get large weights (`req:women-led` -0.36, `fact:annual_revenue=unknown|and`
  -0.43), which pushes Women Who Tech to 0.18. (2) The shared-focus stage only knows synthetic
  topic vocabulary, so cloud-credit wording (AWS Activate) scores as unrelated. (3) The model
  has no explicit way to apply a stated requirement that an applicant fact contradicts: 11
  synthetic-validation `ineligible` pairs are predicted `match`.

## Data and splits

Clean synthetic database and `Ardook_data.xlsx` only; never the messy synthetic data. Splits are
exactly the v1 split manifest (`results/ardook_match_v1/split_manifest.json`; the run asserts
equality). Selection uses synthetic validation and out-of-fold real dev (GroupKFold, 5 folds by
grant family, as v1). Synthetic test and real held-out test are scored once after selection.

**All hidden-test rows and the real held-out rows have already been examined. Their v2 results
are post-exposure regression checks, not independent unseen accuracy.** They are never used to
select a candidate, a threshold or a filter.

## Candidates

1. **v1 configuration, retrained in this run** (twostage:bge + logreg, related C 0.1, conflict
   C 1.0, real weight share 0.10) - the reference; the saved v1 model is also scored as-is.
2. **Regularization grid** on the same model: related C in {0.03, 0.1, 0.3}, conflict C in
   {0.1, 0.3, 1.0}, real weight share in {0.05, 0.10, 0.20}; threshold from 0.05 to 0.95 in steps
   of 0.05, chosen on the selection score as in v1. Nothing else changes.
3. **Requirement filter** (`src/eligibility_filter.py`, written from general requirement
   language, no grant names): parses entity type, location, annual revenue bounds, woman-led
   and operating-age clauses, handles negation and "or"-alternatives, skips hedged clauses
   ("typically", "usually", "call-specific"), and returns `pass`, `fail` or `unknown`; a missing
   applicant fact is always `unknown`. It never turns `not_match` into `match`; a `fail` turns a
   model `match` into a final `not_match` with the quoted clause as evidence. Social impact is
   not treated as nonprofit; woman-led is not treated as majority or percentage ownership.
   No deadline or open/closed filtering. Applied on top of candidates 1 and the best of 2.

## Selection objective (unchanged from v1)

Selection score = mean of synthetic-validation macro-F1 and real-dev out-of-fold macro-F1.

## Promotion rule (fixed now)

A candidate replaces the v1 inference default only if, against the v1 configuration retrained in
this run, all of these hold:

1. its selection score is higher by at least 0.02;
2. real-dev out-of-fold macro-F1 is not lower, and real-dev match precision and match recall
   each drop by no more than 0.05;
3. synthetic-validation match precision and match recall each drop by no more than 0.05.

Ties or failures keep v1 (smallest change). The filter is integrated only if, on top of the
promoted (or v1) model, it does not lower real-dev out-of-fold macro-F1, flips no correct real
dev prediction to wrong, and raises synthetic-validation macro-F1. Because the filter is written
from requirement language and the synthetic requirements are templated, its synthetic gain is
expected to be optimistic and is reported as such.

## Reporting

For v1 (saved), the v1 retrain, the grid winner, and each with/without filter: exact counts,
accuracy, macro-F1, match precision/recall/F1, false positives and false negatives per grant on
synthetic validation (selection), synthetic test, real dev out-of-fold, real held-out test and the
10 hidden grants, plus filter coverage (how many pairs got `pass`/`fail`/`unknown` per rule) and
the conditional tracker grants' predictions (unscored). Real-dev numbers used for selection are
optimistic because 27 configurations and a threshold were chosen on 19 grants.

## Deviations

1. Before the full run, the script was smoke-tested with one or two grid configurations into a
   scratch folder to check it ran end to end; those numbers were not used or kept.
2. Retraining the v1 configuration does not reproduce the saved v1 model bit for bit (largest
   P(match) difference 0.024 on the tracker; Tara's Circle falls below 0.25 in the retrain). The
   promotion rule compared against the retrain, as declared; the saved v1 model is reported too.
3. The requirement filter failed its integration check (it flips Women TechEU, correct under the
   promoted model, to `not_match`), so the v2 config keeps it off. The CLI exposes it only as an
   explicit opt-in (`--requirement-filter`); the app does not use it.
