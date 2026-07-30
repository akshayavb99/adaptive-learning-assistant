import json
import math
import os
from typing import Any

from database import ensure_usage_schema


class UsageRecorder:
    """Persist OpenAI token usage and configurable estimated costs."""

    def __init__(self, connection: Any):
        self.connection = connection
        self.default_rates = (
            self._rate("OPENAI_INPUT_COST_PER_MILLION_TOKENS", "5"),
            self._rate("OPENAI_CACHED_INPUT_COST_PER_MILLION_TOKENS", "0.5"),
            self._rate("OPENAI_OUTPUT_COST_PER_MILLION_TOKENS", "30"),
        )
        self.model_pricing = self._load_model_pricing()
        ensure_usage_schema(connection)

    @staticmethod
    def _rate(name: str, default: str) -> float:
        value = float(os.getenv(name, default))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite, non-negative number")
        return value

    @classmethod
    def _validate_model_rates(cls, model: str, rates: Any) -> tuple[float, float, float]:
        if not isinstance(rates, dict):
            raise ValueError(f"Pricing for model {model!r} must be an object")
        required = ("input", "cached_input", "output")
        missing = [name for name in required if name not in rates]
        if missing:
            raise ValueError(f"Pricing for model {model!r} is missing: {", ".join(missing)}")
        values = []
        for name in required:
            value = rates[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Pricing for model {model!r} field {name!r} must be a finite, non-negative number")
            values.append(float(value))
        return tuple(values)

    @classmethod
    def _load_model_pricing(cls) -> dict[str, tuple[float, float, float]]:
        raw = os.getenv("OPENAI_MODEL_PRICING", "").strip()
        if not raw:
            return {}
        try:
            configured = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("OPENAI_MODEL_PRICING must be valid JSON") from exc
        if not isinstance(configured, dict):
            raise ValueError("OPENAI_MODEL_PRICING must be a JSON object keyed by model name")
        pricing = {}
        for model, rates in configured.items():
            if not isinstance(model, str) or not model.strip():
                raise ValueError("OPENAI_MODEL_PRICING model names must be non-empty strings")
            pricing[model] = cls._validate_model_rates(model, rates)
        return pricing

    def _rates_for_model(self, model: str) -> tuple[float, float, float]:
        return self.model_pricing.get(model, self.default_rates)

    @staticmethod
    def _get(value: Any, key: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(key, default)
        return getattr(value, key, default)

    def record(self, response: Any, model: str, operation: str) -> bool:
        usage = self._get(response, "usage")
        if usage is None:
            return False
        input_tokens = int(self._get(usage, "input_tokens", 0) or 0)
        output_tokens = int(self._get(usage, "output_tokens", 0) or 0)
        total_tokens = int(self._get(usage, "total_tokens", input_tokens + output_tokens) or 0)
        details = self._get(usage, "input_tokens_details")
        cached_tokens = int(self._get(details, "cached_tokens", 0) or 0)
        input_rate, cached_input_rate, output_rate = self._rates_for_model(model)
        billable_input_tokens = max(0, input_tokens - cached_tokens)
        input_cost = billable_input_tokens * input_rate / 1_000_000
        cached_cost = cached_tokens * cached_input_rate / 1_000_000
        output_cost = output_tokens * output_rate / 1_000_000
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO openai_usage (
                    operation, model, response_id, input_tokens, cached_input_tokens,
                    output_tokens, total_tokens, input_cost_usd, cached_input_cost_usd,
                    output_cost_usd, total_cost_usd
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    operation, model, self._get(response, "id"), input_tokens,
                    cached_tokens, output_tokens, total_tokens, input_cost,
                    cached_cost, output_cost, input_cost + cached_cost + output_cost,
                ),
            )
        self.connection.commit()
        return True