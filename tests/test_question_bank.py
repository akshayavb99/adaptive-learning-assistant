import unittest
from unittest.mock import patch

from question_bank import QuestionBankStore


class FakeCursor:
    def __init__(self, nearest):
        self.nearest = iter(nearest)
        self.executions = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, statement, parameters=None):
        self.executions.append((statement, parameters))

    def fetchone(self):
        return next(self.nearest)


class FakeConnection:
    def __init__(self, nearest):
        self.cursor_instance = FakeCursor(nearest)
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class FakeEmbedder:
    def encode(self, text):
        return [1.0, 0.0]


def record(text, difficulty=3):
    return {
        "question": {
            "question_type": "short_answer",
            "question": text,
            "options": [],
            "expected_answer": ["answer"],
            "explanation": "because",
        },
        "generation_difficulty": difficulty,
        "source_paths": ["notes/topic.md", "notes/topic.md"],
    }


class QuestionBankStoreTests(unittest.TestCase):
    @patch("question_bank.ensure_question_bank_schema")
    def test_inserts_below_threshold_and_skips_at_or_above(self, ensure_schema):
        connection = FakeConnection([None, (0.95,), (0.949,)])
        store = QuestionBankStore(connection, FakeEmbedder())

        result = store.store_questions([record("first"), record("duplicate"), record("different")])

        self.assertEqual(result, {"inserted": 2, "skipped": 1})
        self.assertEqual(connection.commits, 1)
        self.assertEqual(connection.rollbacks, 0)
        inserts = [
            call for call in connection.cursor_instance.executions
            if "INSERT INTO question_bank" in call[0]
        ]
        self.assertEqual(len(inserts), 2)
        self.assertEqual(inserts[0][1][5], ["notes/topic.md"])

    @patch("question_bank.ensure_question_bank_schema")
    def test_invalid_difficulty_rolls_back(self, ensure_schema):
        connection = FakeConnection([])
        store = QuestionBankStore(connection, FakeEmbedder())

        with self.assertRaises(ValueError):
            store.store_questions([record("invalid", difficulty=6)])

        self.assertEqual(connection.commits, 0)
        self.assertEqual(connection.rollbacks, 1)


if __name__ == "__main__":
    unittest.main()
