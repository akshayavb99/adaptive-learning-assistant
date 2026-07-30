import tempfile
import unittest
from pathlib import Path

from generate_gold_truth import validate_record
from retrieval_evaluation import normalize_source_path


class GoldTruthTests(unittest.TestCase):
    def test_normalizes_container_absolute_source(self):
        self.assertEqual(normalize_source_path("/app/data/index.md"), "data/index.md")
        self.assertEqual(normalize_source_path('D:/Projects/adaptive-testing-assistant/data/index.md'), 'data/index.md')

    def test_validates_evidence_and_provenance(self):
        chunks = {("data/index.md", 3): {"content": "Vector search uses embeddings."}}
        record = {
            "id": "gold-001",
            "query": "What does vector search use?",
            "question": "What does vector search use?",
            "answer": "Embeddings.",
            "relevant_sources": ["data/index.md"],
            "relevant_chunks": [{"source_path": "data/index.md", "chunk_index": 3}],
            "evidence": "Vector search uses embeddings.",
        }
        validate_record(record, chunks)

    def test_rejects_evidence_outside_chunk(self):
        chunks = {("data/index.md", 3): {"content": "Vector search uses embeddings."}}
        record = {
            "id": "gold-001",
            "query": "What does vector search use?",
            "question": "What does vector search use?",
            "answer": "A database.",
            "relevant_sources": ["data/index.md"],
            "relevant_chunks": [{"source_path": "data/index.md", "chunk_index": 3}],
            "evidence": "This is not in the chunk.",
        }
        with self.assertRaises(ValueError):
            validate_record(record, chunks)


if __name__ == "__main__":
    unittest.main()

