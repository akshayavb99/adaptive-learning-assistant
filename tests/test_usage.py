import os
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from usage import UsageRecorder


class FakeCursor:
    def __init__(self):
        self.executions = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, statement, parameters=None):
        self.executions.append((statement, parameters))


class FakeConnection:
    def __init__(self):
        self.cursor_instance = FakeCursor()
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class UsageRecorderTests(unittest.TestCase):
    def test_records_tokens_and_cost_with_cached_input(self):
        connection = FakeConnection()
        with patch.dict(os.environ, {
            "OPENAI_INPUT_COST_PER_MILLION_TOKENS": "5",
            "OPENAI_CACHED_INPUT_COST_PER_MILLION_TOKENS": "0.5",
            "OPENAI_OUTPUT_COST_PER_MILLION_TOKENS": "30",
        }, clear=False):
            recorder = UsageRecorder(connection)
            response = SimpleNamespace(
                id="resp_1",
                usage=SimpleNamespace(
                    input_tokens=1000,
                    output_tokens=500,
                    total_tokens=1500,
                    input_tokens_details=SimpleNamespace(cached_tokens=200),
                ),
            )
            self.assertTrue(recorder.record(response, "test-model", "test_question"))

        insert = connection.cursor_instance.executions[-1][1]
        self.assertEqual(insert[:7], ("test_question", "test-model", "resp_1", 1000, 200, 500, 1500))
        self.assertEqual(insert[7], Decimal("0.004"))
        self.assertEqual(insert[8], Decimal("0.0001"))
        self.assertEqual(insert[9], Decimal("0.015"))
        self.assertEqual(insert[10], Decimal("0.0191"))
        self.assertEqual(connection.commits, 2)

    def test_missing_usage_is_ignored(self):
        connection = FakeConnection()
        recorder = UsageRecorder(connection)

        self.assertFalse(recorder.record(SimpleNamespace(id="resp_2"), "test-model", "answer_grade"))
        self.assertEqual(len(connection.cursor_instance.executions), 4)


if __name__ == "__main__":
    unittest.main()