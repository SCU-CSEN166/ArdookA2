import unittest
from datetime import date

import numpy as np
import pandas as pd

from src.fit_judge import (SYSTEM_PROMPT, FitJudgeRanker, adapter_base_model, judge_messages, judge_prompt,
                            model_cache_dir, skip_reasoning)
from src.pipeline import EligibilityChecker, EligibilityRule, OrganizationProfile


def grant(row, program, requirements, deadline=None, status="Open"):
    return {
        "source_row": row,
        "program": program,
        "type": "Grant",
        "core_requirements": requirements,
        "deadline_date": pd.Timestamp(deadline) if deadline else pd.NaT,
        "deadline": deadline or "Rolling",
        "current_status": status,
        "concerns": "",
        "amount_benefit": "$10,000",
        "official_source": f"https://example.org/{row}",
        "fit": "Strong",
    }


class KeywordEncoder:
    """Deterministic 3-d vectors: health / arts / anything else."""

    model_name = "fake-encoder"

    def _vec(self, text):
        t = text.casefold()
        v = np.array([("health" in t) * 1.0, ("arts" in t) * 1.0, 0.1])
        return v / np.linalg.norm(v)

    def encode_queries(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def encode_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])


class FitJudgeRankerTests(unittest.TestCase):
    def setUp(self):
        self.grants = pd.DataFrame([
            grant(2, "Health Fund", "Supports community health; registered nonprofit"),
            grant(3, "Arts Fund", "Supports community arts"),
            grant(4, "Closed Health Fund", "Supports community health", deadline="2026-01-01"),
            grant(5, "Health For-Profit Fund", "Supports community health; for-profit company"),
            grant(6, "General Fund", "Any mission"),
        ])
        self.checker = EligibilityChecker({
            "Health For-Profit Fund": [EligibilityRule("entity_type", "in", ["for-profit"], "for-profit company")],
        })
        self.profile = OrganizationProfile("A nonprofit running community health clinics", {"entity_type": "nonprofit"})
        self.prompts = []

    def judge(self, prompts):
        self.prompts += prompts
        # The fake judge prefers the arts grant, to show it can override the first stage.
        return [0.9 if "Arts Fund" in p else 0.2 for p in prompts]

    def ranker(self, depth=30):
        return FitJudgeRanker(self.grants, self.checker, encoder=KeywordEncoder(), judge=self.judge,
                              adapter_dir="fake-adapter", first_stage_depth=depth)

    def test_only_eligible_grants_are_judged_and_returned(self):
        results = self.ranker().recommend(self.profile, as_of=date(2026, 10, 1), top_k=10)
        rows = [r["source_row"] for r in results]
        self.assertNotIn(4, rows)  # past deadline
        self.assertNotIn(5, rows)  # known rule fails
        self.assertEqual(sorted(rows), [2, 3, 6])
        self.assertEqual(len(self.prompts), 3)

    def test_judge_orders_shortlist_and_scores_are_probabilities(self):
        results = self.ranker().recommend(self.profile, as_of=date(2026, 10, 1), top_k=3)
        self.assertEqual(results[0]["source_row"], 3)
        self.assertEqual(results[0]["score"], 90.0)
        self.assertEqual(results[0]["ranker"]["fit_probability"], 0.9)
        self.assertTrue(all(r["eligibility"] == "needs_verification" for r in results))
        self.assertEqual(results[0]["source_url"], "https://example.org/3")

    def test_grants_beyond_shortlist_keep_first_stage_order_below_judged(self):
        results = self.ranker(depth=1).recommend(self.profile, as_of=date(2026, 10, 1), top_k=3)
        self.assertEqual(results[0]["source_row"], 2)  # best first-stage match is the only judged grant
        self.assertTrue(results[0]["ranker"]["judged"])
        self.assertFalse(results[1]["ranker"]["judged"])
        self.assertIsNone(results[1]["score"])
        self.assertEqual(len(self.prompts), 1)

    def test_prompt_includes_attributes_but_not_fit_labels(self):
        prompt = judge_prompt(self.profile, self.grants.to_dict("records")[0])
        self.assertIn('"entity_type": "nonprofit"', prompt)
        self.assertIn("core_requirements: Supports community health", prompt)
        self.assertNotIn("Strong", prompt)


class DeepSeekJudgeTests(unittest.TestCase):
    DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

    def test_deepseek_gets_instruction_in_user_turn(self):
        self.assertEqual(judge_messages(self.DEEPSEEK, "Q"), [{"role": "user", "content": SYSTEM_PROMPT + "\n\nQ"}])
        self.assertEqual(judge_messages("Qwen/Qwen2.5-7B-Instruct", "Q")[0], {"role": "system", "content": SYSTEM_PROMPT})

    def test_reasoning_block_is_closed_so_next_token_is_the_answer(self):
        self.assertEqual(skip_reasoning(self.DEEPSEEK, "<｜Assistant｜><think>\n"), "<｜Assistant｜><think>\n\n</think>\n\n")
        self.assertEqual(skip_reasoning(self.DEEPSEEK, "<｜Assistant｜>"), "<｜Assistant｜><think>\n\n</think>\n\n")
        self.assertEqual(skip_reasoning("Qwen/Qwen2.5-7B-Instruct", "<|im_start|>assistant\n"), "<|im_start|>assistant\n")

    def test_base_model_comes_from_adapter_and_large_cache_is_found(self):
        import json, os, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": self.DEEPSEEK}))
            self.assertEqual(adapter_base_model(tmp), self.DEEPSEEK)
            self.assertIsNone(adapter_base_model(Path(tmp) / "missing"))
            (Path(tmp) / "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B").mkdir()
            old = os.environ.get("LARGE_MODEL_CACHE")
            os.environ["LARGE_MODEL_CACHE"] = tmp
            try:
                self.assertEqual(model_cache_dir(self.DEEPSEEK), tmp)
                self.assertIsNone(model_cache_dir("Qwen/Qwen2.5-7B-Instruct"))
            finally:
                os.environ.pop("LARGE_MODEL_CACHE") if old is None else os.environ.update(LARGE_MODEL_CACHE=old)


if __name__ == "__main__":
    unittest.main()
