import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from readpaper.llm import InvalidModelOutput, validate_related_selection  # noqa: E402


class RelatedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {"id": "paper-a", "kind": "related", "title": "A Study"},
            {"id": "paper-b", "kind": "recent", "title": "Another Study"},
        ]

    def test_keeps_model_rank_order_and_reasons_for_known_candidates(self):
        selected = validate_related_selection({"selections": [
            {"id": "paper-b", "reason": "Its method is closely related."},
            {"id": "paper-a", "reason": "It studies the same research problem."},
        ]}, self.candidates, limit=8)

        self.assertEqual([item["id"] for item in selected], ["paper-b", "paper-a"])
        self.assertEqual([item["llm_relevance_rank"] for item in selected], [1, 2])

    def test_rejects_unknown_or_duplicate_candidate_ids(self):
        with self.assertRaises(InvalidModelOutput):
            validate_related_selection({"selections": [
                {"id": "unknown", "reason": "Not a candidate."},
                {"id": "paper-a", "reason": "Known candidate."},
            ]}, self.candidates, limit=8)


if __name__ == "__main__":
    unittest.main()
