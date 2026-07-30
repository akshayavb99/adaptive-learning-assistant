import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from usage import UsageRecorder


class Cursor:
    def __init__(self):
        self.calls = []
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def execute(self, query, params=None): self.calls.append((query, params))

class Connection:
    def __init__(self): self.cursor_instance = Cursor(); self.commits = 0
    def cursor(self): return self.cursor_instance
    def commit(self): self.commits += 1

class UsageRecorderTests(unittest.TestCase):
    def test_records_tokens_and_configured_costs(self):
        connection = Connection()
        response = SimpleNamespace(
            id="resp-1",
            usage=SimpleNamespace(
                input_tokens=1000,
                output_tokens=500,
                total_tokens=1500,
                input_tokens_details=SimpleNamespace(cached_tokens=200),
            ),
        )
        with patch.dict(os.environ, {
            "OPENAI_INPUT_COST_PER_MILLION_TOKENS": "5",
            "OPENAI_CACHED_INPUT_COST_PER_MILLION_TOKENS": "1",
            "OPENAI_OUTPUT_COST_PER_MILLION_TOKENS": "10",
        }, clear=False):
            recorder = UsageRecorder(connection)
            self.assertTrue(recorder.record(response, "test-model", "answer_grade"))
        params = connection.cursor_instance.calls[-1][1]
        self.assertEqual(params[:6], ("answer_grade", "test-model", "resp-1", 1000, 200, 500))
        self.assertAlmostEqual(float(params[-1]), 0.0092)

    def test_model_specific_pricing_overrides_global_rates(self):
        connection = Connection()
        response = SimpleNamespace(
            id="resp-model",
            usage=SimpleNamespace(
                input_tokens=1000,
                output_tokens=500,
                total_tokens=1500,
                input_tokens_details=SimpleNamespace(cached_tokens=200),
            ),
        )
        with patch.dict(os.environ, {
            "OPENAI_INPUT_COST_PER_MILLION_TOKENS": "5",
            "OPENAI_CACHED_INPUT_COST_PER_MILLION_TOKENS": "1",
            "OPENAI_OUTPUT_COST_PER_MILLION_TOKENS": "10",
            "OPENAI_MODEL_PRICING": json.dumps({
                "test-model": {"input": 2, "cached_input": 0.2, "output": 8},
            }),
        }, clear=False):
            recorder = UsageRecorder(connection)
            self.assertTrue(recorder.record(response, "test-model", "answer_grade"))
        params = connection.cursor_instance.calls[-1][1]
        self.assertAlmostEqual(float(params[7]), 0.0016)
        self.assertAlmostEqual(float(params[8]), 0.00004)
        self.assertAlmostEqual(float(params[9]), 0.004)
        self.assertAlmostEqual(float(params[10]), 0.00564)

    def test_unknown_model_uses_global_rates(self):
        connection = Connection()
        with patch.dict(os.environ, {"OPENAI_MODEL_PRICING": json.dumps({
            "other-model": {"input": 1, "cached_input": 0.1, "output": 1},
        })}, clear=False):
            recorder = UsageRecorder(connection)
        self.assertEqual(recorder._rates_for_model("unknown-model"), recorder.default_rates)

    def test_invalid_model_pricing_is_rejected(self):
        connection = Connection()
        with patch.dict(os.environ, {"OPENAI_MODEL_PRICING": '{"test-model": {}}'}, clear=False):
            with self.assertRaisesRegex(ValueError, "missing"):
                UsageRecorder(connection)

    def test_missing_usage_is_not_persisted(self):
        connection = Connection()
        recorder = UsageRecorder(connection)
        self.assertFalse(recorder.record(SimpleNamespace(), "model", "operation"))
        self.assertEqual(len(connection.cursor_instance.calls), 4)

if __name__ == "__main__": unittest.main()