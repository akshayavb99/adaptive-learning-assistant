import logging
import os
from decimal import Decimal
from typing import Any

from database import ensure_usage_schema


logger = logging.getLogger(__name__)


class UsageRecorder:
    """Persist OpenAI response usage and estimated cost in PostgreSQL."""

    def __init__(self, connection: Any):
        self.connection = connection
        self.input_rate = self._rate("OPENAI_INPUT_COST_PER_MILLION_TOKENS", "5")
        self.cached_input_rate = self._rate("OPENAI_CACHED_INPUT_COST_PER_MILLION_TOKENS", "0.5")
        self.output_rate = self._rate("OPENAI_OUTPUT_COST_PER_MILLION_TOKENS", "30")
        ensure_usage_schema(connection)

    @staticmethod
    def _rate(name: str, default: str) -> Decimal:
        try:
            value = Decimal(os.getenv(name, default))
        except Exception as exc:
            raise ValueError(f"{name} must be a non-negative decimal") from exc
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value

    @staticmethod
    def _get(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    def record(self, response: Any, model: str, operation: str) -> bool:
        """Persist usage from a Responses API response; return whether data existed."""
        usage = self._get(response, "usage")
        if usage is None:
            return False

        input_tokens = int(self._get(usage, "input_tokens", 0) or 0)
        output_tokens = int(self._get(usage, "output_tokens", 0) or 0)
        total_tokens = int(self._get(usage, "total_tokens", input_tokens + output_tokens) or 0)
        input_details = self._get(usage, "input_tokens_details")
        cached_input_tokens = int(self._get(input_details, "cached_tokens", 0) or 0)
        cached_input_tokens = min(max(cached_input_tokens, 0), input_tokens)
        uncached_input_tokens = input_tokens - cached_input_tokens

        input_cost = Decimal(uncached_input_tokens) / Decimal(1_000_000) * self.input_rate
        cached_cost = Decimal(cached_input_tokens) / Decimal(1_000_000) * self.cached_input_rate
        output_cost = Decimal(output_tokens) / Decimal(1_000_000) * self.output_rate
        total_cost = input_cost + cached_cost + output_cost
        response_id = self._get(response, "id")

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO openai_usage (
                        operation, model, response_id, input_tokens,
                        cached_input_tokens, output_tokens, total_tokens,
                        input_cost_usd, cached_input_cost_usd,
                        output_cost_usd, total_cost_usd
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        operation,
                        model,
                        response_id,
                        input_tokens,
                        cached_input_tokens,
                        output_tokens,
                        total_tokens,
                        input_cost,
                        cached_cost,
                        output_cost,
                        total_cost,
                    ),
                )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return True