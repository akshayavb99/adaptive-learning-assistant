import unittest

from retrieval_evaluation import calculate_metrics


class RetrievalEvaluationTests(unittest.TestCase):
    def test_hit_rate_and_mrr(self):
        cases = [
            {"id": "q1", "query": "one", "relevant_sources": ["docs/a.md"]},
            {"id": "q2", "query": "two", "relevant_sources": ["docs/z.md"]},
            {"id": "q3", "query": "three", "relevant_sources": ["docs/c.md"]},
        ]
        retrieved = {
            "q1": [{"source_path": "docs/x.md"}, {"source_path": "docs/a.md"}],
            "q2": [{"source_path": "docs/y.md"}],
            "q3": [{"source_path": "docs/c.md"}],
        }

        hit_rate, mrr, rows = calculate_metrics(cases, retrieved, 2)

        self.assertAlmostEqual(hit_rate, 2 / 3)
        self.assertAlmostEqual(mrr, (0.5 + 0 + 1.0) / 3)
        self.assertTrue(rows[1]["is_relevant"])
        self.assertEqual(rows[1]["reciprocal_rank"], 0.5)

    def test_chunk_level_relevance(self):
        cases = [{
            "id": "q1",
            "query": "one",
            "relevant_sources": ["docs/a.md"],
            "relevant_chunks": [{"source_path": "docs/a.md", "chunk_index": 2}],
        }]
        retrieved = {
            "q1": [
                {"source_path": "docs/a.md", "chunk_index": 1},
                {"source_path": "docs/a.md", "chunk_index": 2},
            ]
        }

        hit_rate, mrr, rows = calculate_metrics(cases, retrieved, 2)

        self.assertEqual(hit_rate, 1.0)
        self.assertEqual(mrr, 0.5)
        self.assertFalse(rows[0]["is_relevant"])
        self.assertTrue(rows[1]["is_relevant"])


if __name__ == "__main__":
    unittest.main()
