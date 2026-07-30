import json
import logging
import os
import random
from collections.abc import Sequence
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)


QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "question_type": {
            "type": "string",
            "enum": ["short_answer", "single_choice", "multiple_choice"],
        },
        "question": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}},
        "expected_answer": {"type": "array", "items": {"type": "string"}},
        "explanation": {"type": "string"},
    },
    "required": ["question_type", "question", "options", "expected_answer", "explanation"],
    "additionalProperties": False,
}

QUALITY_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "groundedness_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "answer_correctness_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "clarity_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "difficulty_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "groundedness_explanation": {"type": "string"},
        "answer_correctness_explanation": {"type": "string"},
        "clarity_explanation": {"type": "string"},
        "difficulty_explanation": {"type": "string"},
    },
    "required": ["groundedness_score", "answer_correctness_score", "clarity_score", "difficulty_score", "groundedness_explanation", "answer_correctness_explanation", "clarity_explanation", "difficulty_explanation"],
    "additionalProperties": False,
}
GRADING_SCHEMA = {
    "type": "object",
    "properties": {
        "is_correct": {"type": "boolean"},
        "correct_answer": {"type": "array", "items": {"type": "string"}},
        "feedback": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["is_correct", "correct_answer", "feedback", "explanation"],
    "additionalProperties": False,
}


def parse_topic_input(value: str) -> list[str]:
    """Parse a comma-separated topic string into unique, trimmed topics."""
    if not isinstance(value, str):
        raise ValueError("topic input must be a string")
    return list(dict.fromkeys(
        topic.strip()
        for topic in value.split(",")
        if topic.strip()
    ))

class OpenAIRAGClient:
    """Retrieve knowledge-base context and run an adaptive OpenAI-generated test."""

    def __init__(
        self,
        retriever: Any,
        client: Any | None = None,
        model: str | None = None,
        search_mode: str | None = None,
        num_results: int | None = None,
        rng: random.Random | None = None,
        usage_recorder: Any | None = None,
        question_bank: Any | None = None,
    ):
        load_dotenv()
        self.retriever = retriever
        self.client = client if client is not None else OpenAI()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
        self.search_mode = search_mode or os.getenv("RAG_SEARCH_MODE", "hybrid")
        configured_results = (
            num_results if num_results is not None else int(os.getenv("RAG_TOP_K", "5"))
        )
        if self.search_mode not in {"hybrid", "vector"}:
            raise ValueError("search_mode must be 'hybrid' or 'vector'")
        if (
            isinstance(configured_results, bool)
            or not isinstance(configured_results, int)
            or configured_results <= 0
        ):
            raise ValueError("num_results must be a positive integer")
        self.num_results = configured_results
        self.rng = rng or random.Random()
        self.usage_recorder = usage_recorder
        self.question_bank = question_bank
        self.judge_model = os.getenv("OPENAI_JUDGE_MODEL", self.model)
        self.judge_max_attempts = int(os.getenv("QUESTION_JUDGE_MAX_ATTEMPTS", "3"))
        self.judge_pass_threshold = float(os.getenv("QUESTION_JUDGE_PASS_THRESHOLD", "4.0"))
        if self.judge_max_attempts <= 0:
            raise ValueError("QUESTION_JUDGE_MAX_ATTEMPTS must be positive")
        if not 1 <= self.judge_pass_threshold <= 5:
            raise ValueError("QUESTION_JUDGE_PASS_THRESHOLD must be between 1 and 5")
        self.judgment_connection = getattr(retriever, "connection", None)

    @staticmethod
    def _normalize_topics(topics: str | Sequence[str]) -> list[str]:
        if isinstance(topics, str):
            values = [topics]
        elif isinstance(topics, Sequence):
            values = list(topics)
        else:
            raise ValueError("topics must be a non-empty string or sequence of strings")

        normalized: list[str] = []
        for topic in values:
            if not isinstance(topic, str):
                raise ValueError("each topic must be a string")
            topic = topic.strip()
            if topic and topic not in normalized:
                normalized.append(topic)
        if not normalized:
            raise ValueError("topics must contain at least one non-empty topic")
        return normalized

    @staticmethod
    def _normalize_question_text(question: str) -> str:
        return " ".join(question.split()).casefold()

    @staticmethod
    def _topic_label(topics: str | Sequence[str]) -> str:
        return "; ".join(OpenAIRAGClient._normalize_topics(topics))

    def retrieve(
        self,
        query: str | Sequence[str],
        num_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """Embed and search each topic independently, then merge its chunks."""
        topics = self._normalize_topics(query)
        search = (
            self.retriever.search_hybrid
            if self.search_mode == "hybrid"
            else self.retriever.search_index
        )
        if num_results is not None and (
            isinstance(num_results, bool)
            or not isinstance(num_results, int)
            or num_results <= 0
        ):
            raise ValueError("num_results must be a positive integer")

        merged: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
        if num_results is None:
            topic_limits = [self.num_results] * len(topics)
        else:
            topic_limits = [
                num_results // len(topics) + (index < num_results % len(topics))
                for index in range(len(topics))
            ]

        for topic, topic_limit in zip(topics, topic_limits):
            if topic_limit == 0:
                continue
            for chunk in search(topic, topic_limit):
                key = (chunk.get("source_path"), chunk.get("chunk_index"), chunk.get("content"))
                existing = merged.get(key)
                if existing is None:
                    existing = dict(chunk)
                    existing["matched_topics"] = [topic]
                    merged[key] = existing
                elif topic not in existing["matched_topics"]:
                    existing["matched_topics"].append(topic)

        return list(merged.values())

    @staticmethod
    def _chunk_key(chunk: dict[str, Any]) -> tuple[Any, Any, Any]:
        """Return a stable identity for a retrieved chunk."""
        return (chunk.get("source_path"), chunk.get("chunk_index"), chunk.get("content"))

    @classmethod
    def _schedule_chunks(
        cls, chunks: list[dict[str, Any]], count: int
    ) -> list[dict[str, Any]]:
        """Build a balanced round-robin schedule of retrieved chunks."""
        unique_chunks: list[dict[str, Any]] = []
        seen: set[tuple[Any, Any, Any]] = set()
        for chunk in chunks:
            key = cls._chunk_key(chunk)
            if key not in seen:
                seen.add(key)
                unique_chunks.append(chunk)

        if not unique_chunks or count <= 0:
            return []
        limit = min(count, 2) if len(unique_chunks) == 1 else count
        return [unique_chunks[index % len(unique_chunks)] for index in range(limit)]

    def select_random_topic(self) -> str:
        """Select a topic from Markdown sources available to the retriever."""
        if hasattr(self.retriever, "list_topics"):
            topics = self.retriever.list_topics()
        else:
            input_path = getattr(self.retriever, "input_path", None)
            topics = (
                sorted(str(path.relative_to(input_path).with_suffix(""))
                       for path in input_path.rglob("*.md") if path.is_file())
                if input_path is not None else []
            )
        if not topics:
            raise RuntimeError("No Markdown knowledge-base topics are available")
        return self.rng.choice(topics)

    @staticmethod
    def _context(chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            raise RuntimeError("The retriever returned no document chunks")
        return "\n\n".join(
            f"Topics: {', '.join(chunk.get('matched_topics', []))}\n"
            f"Source: {chunk.get('source_path', 'unknown')}\n"
            f"{chunk.get('content', '')}"
            for chunk in chunks
        )

    def _structured_response(self, prompt: str, name: str, schema: dict[str, Any], model: str | None = None) -> dict[str, Any]:
        response = self.client.responses.create(
            model=model or self.model,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": name,
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        if self.usage_recorder is not None:
            try:
                self.usage_recorder.record(response, model or self.model, name)
            except Exception:
                logger.warning("Could not persist OpenAI usage metrics", exc_info=True)

        output = getattr(response, "output_text", None)
        if not isinstance(output, str) or not output.strip():
            raise RuntimeError(f"OpenAI returned no structured {name} output")
        try:
            result = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"OpenAI returned invalid {name} JSON") from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"OpenAI returned an invalid {name} object")
        return result

    def generate_question(
        self,
        topic: str | Sequence[str],
        chunks: list[dict[str, Any]],
        difficulty: int,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate one grounded question at the requested difficulty."""
        self._normalize_topics(topic)
        if difficulty not in range(1, 6):
            raise ValueError("difficulty must be between 1 and 5")
        history_text = json.dumps(
            [item.get("question", "") for item in (history or [])],
            ensure_ascii=False,
        )
        prompt = f"""You are an adaptive test question writer.
Create exactly one short question about the topic using only the supplied knowledge-base context.
Difficulty is {difficulty} on a scale from 1 (introductory) to 5 (advanced).
Choose exactly one question type: short_answer, single_choice, or multiple_choice.
Keep the question concise and unambiguous. For short_answer, options must be an empty array.
For single_choice and multiple_choice, provide 2-5 options. expected_answer must contain the exact
answer text or answer texts from the options for choice questions. Do not reveal the answer in the
question. Avoid repeating any question from the test history.

Topics: {self._topic_label(topic)}
Previous questions in this test (do not repeat any of them): {history_text}
Knowledge-base context:
{self._context(chunks)}
"""
        question = self._structured_response(prompt, "test_question", QUESTION_SCHEMA)
        self._validate_question(question)
        return question

    def _persist_judgment(self, question: dict[str, Any], chunks: list[dict[str, Any]],
                          assigned_difficulty: int, attempt_number: int,
                          judgment: dict[str, Any] | None, status: str) -> None:
        if self.judgment_connection is None:
            return
        scores = judgment or {}
        source_paths = sorted({chunk.get("source_path") for chunk in chunks if chunk.get("source_path")})
        with self.judgment_connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO question_quality_judgments (
                    question_text, question_type, options, expected_answer, source_paths,
                    assigned_difficulty, groundedness_score, answer_correctness_score,
                    clarity_score, difficulty_score, overall_score, passed, judge_status,
                    judge_model, attempt_number, groundedness_explanation,
                    answer_correctness_explanation, clarity_explanation, difficulty_explanation
                )
                VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    question.get("question", ""), question.get("question_type", ""),
                    json.dumps(question.get("options", []), ensure_ascii=False),
                    json.dumps(question.get("expected_answer", []), ensure_ascii=False),
                    source_paths, assigned_difficulty, scores.get("groundedness_score"),
                    scores.get("answer_correctness_score"), scores.get("clarity_score"),
                    scores.get("difficulty_score"), scores.get("overall_score"),
                    scores.get("passed", False), status, self.judge_model, attempt_number,
                    scores.get("groundedness_explanation", ""),
                    scores.get("answer_correctness_explanation", ""),
                    scores.get("clarity_explanation", ""), scores.get("difficulty_explanation", ""),
                ),
            )
        self.judgment_connection.commit()

    def judge_question(self, question: dict[str, Any], chunks: list[dict[str, Any]],
                       assigned_difficulty: int, attempt_number: int = 1) -> dict[str, Any]:
        """Evaluate a generated question-answer pair, not a learner answer."""
        prompt = f"""You are a strict quality reviewer for generated adaptive-test questions.
Review the generated question and expected answer using only the supplied knowledge-base context.
Score each criterion from 1 (poor) to 5 (excellent).
Groundedness means the question and answer are supported by the context.
Answer correctness means the expected answer is correct according to the context.
Clarity means the question is concise, unambiguous, and answerable.
Difficulty means the actual difficulty matches the assigned difficulty.
Do not evaluate a learner answer; there is no learner answer in this review.
Return concise explanations for every score.

Assigned difficulty: {assigned_difficulty}/5
Question JSON:
{json.dumps(question, ensure_ascii=False)}
Knowledge-base context:
{self._context(chunks)}
"""
        judgment = self._structured_response(prompt, "question_quality_judge", QUALITY_JUDGE_SCHEMA, self.judge_model)
        score_fields = ("groundedness_score", "answer_correctness_score", "clarity_score", "difficulty_score")
        for field in score_fields:
            value = judgment.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value not in range(1, 6):
                raise RuntimeError(f"OpenAI returned an invalid {field}")
        judgment["overall_score"] = sum(judgment[field] for field in score_fields) / 4
        judgment["passed"] = judgment["overall_score"] >= self.judge_pass_threshold
        self._persist_judgment(question, chunks, assigned_difficulty, attempt_number, judgment,
                               "passed" if judgment["passed"] else "failed")
        return judgment

    def generate_validated_question(self, topic: str | Sequence[str], chunks: list[dict[str, Any]],
                                    difficulty: int, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Generate and judge attempts until a passing pair is available."""
        previous_questions = {
            self._normalize_question_text(item.get("question", ""))
            for item in (history or [])
            if isinstance(item, dict) and isinstance(item.get("question"), str)
        }
        last_error: Exception | None = None
        for attempt in range(1, self.judge_max_attempts + 1):
            question = None
            try:
                question = self.generate_question(topic, chunks, difficulty, history)
                normalized_question = self._normalize_question_text(question["question"])
                if normalized_question in previous_questions:
                    raise RuntimeError("OpenAI generated a duplicate question")
                judgment = self.judge_question(question, chunks, difficulty, attempt)
                if judgment["passed"]:
                    question["quality_judgment"] = judgment
                    return question
            except Exception as exc:
                last_error = exc
                if question is not None:
                    try:
                        self._persist_judgment(question, chunks, difficulty, attempt, None, "error")
                    except Exception:
                        logger.warning("Could not persist failed question judgment", exc_info=True)
        raise RuntimeError("Could not generate a quality-approved question") from last_error
    @staticmethod
    def _validate_question(question: dict[str, Any]) -> None:
        question_type = question.get("question_type")
        options = question.get("options")
        expected = question.get("expected_answer")
        if question_type not in {"short_answer", "single_choice", "multiple_choice"}:
            raise RuntimeError("OpenAI returned an unsupported question type")
        if not isinstance(question.get("question"), str) or not question["question"].strip():
            raise RuntimeError("OpenAI returned an empty question")
        if not isinstance(options, list) or not all(isinstance(value, str) for value in options):
            raise RuntimeError("OpenAI returned invalid question options")
        if not isinstance(expected, list) or not expected or not all(isinstance(value, str) for value in expected):
            raise RuntimeError("OpenAI returned an invalid expected answer")
        if question_type == "short_answer" and options:
            raise RuntimeError("Short-answer questions must not have options")
        if question_type != "short_answer" and not 2 <= len(options) <= 5:
            raise RuntimeError("Choice questions must have between 2 and 5 options")
        if question_type == "single_choice" and len(expected) != 1:
            raise RuntimeError("Single-choice questions must have one expected answer")

    def grade_answer(
        self,
        question: dict[str, Any],
        user_answer: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Grade an answer against the generated question and its source context."""
        if not isinstance(user_answer, str) or not user_answer.strip():
            raise ValueError("user_answer must be a non-empty string")
        prompt = f"""You are grading an adaptive test answer.
Grade the user's answer using the question, expected answer, and knowledge-base context.
For multiple-choice questions, the answer is correct only when the selected set exactly matches
the expected set. Ignore letter casing and surrounding whitespace. Be fair with short answers when
the meaning is equivalent. Return concise feedback and explanation grounded in the context.

Question JSON:
{json.dumps(question, ensure_ascii=False)}
User answer:
{user_answer}
Knowledge-base context:
{self._context(chunks)}
"""
        result = self._structured_response(prompt, "answer_grade", GRADING_SCHEMA)
        if not isinstance(result.get("is_correct"), bool):
            raise RuntimeError("OpenAI returned an invalid correctness value")
        return result

    def persist_questions(self, results: list[dict[str, Any]]) -> dict[str, int] | None:
        """Persist generated questions when a question-bank store is configured."""
        if self.question_bank is None:
            return None
        approved = [record for record in results if record.get("question", {}).get("quality_judgment", {}).get("passed") is True]
        return self.question_bank.store_questions(approved)
    def run_test(
        self,
        topic: str | Sequence[str] | None = None,
        num_questions: int = 20,
        answer_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> dict[str, Any]:
        """Run an interactive adaptive test and return its summary."""
        if topic is not None and (not isinstance(topic, str) or not topic.strip()):
            raise ValueError("topic must be a non-empty string when provided")
        if isinstance(num_questions, bool) or not isinstance(num_questions, int) or num_questions <= 0:
            raise ValueError("num_questions must be a positive integer")
        topic_provided = bool(topic)
        selected_topics = (
            self._normalize_topics(topic)
            if topic_provided
            else self._normalize_topics(self.retriever.list_topics())
        )
        difficulty = 3
        history: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        chunks = self.retrieve(
            selected_topics,
            num_results=num_questions if not topic_provided else None,
        )
        if not chunks:
            raise RuntimeError("The retriever returned no document chunks")

        if not topic_provided:
            chunk_schedule = self._schedule_chunks(chunks, num_questions)
            if len(chunk_schedule) != num_questions:
                raise RuntimeError(
                    f"Requested {num_questions} questions but retrieved "
                    f"only {len(chunk_schedule)} distinct chunks"
                )
        else:
            chunk_schedule = self._schedule_chunks(chunks, num_questions)

        total_questions = len(chunk_schedule)
        for question_number, chunk in enumerate(chunk_schedule, 1):
            question_chunks = [chunk]
            question = self.generate_validated_question(
                selected_topics, question_chunks, difficulty, history
            )
            output_fn(f"\nQuestion {question_number}/{total_questions} (difficulty {difficulty})")
            output_fn(question["question"])
            if question["options"]:
                output_fn("\n".join(f"{index}. {option}" for index, option in enumerate(question["options"], 1)))
            user_answer = answer_fn("Answer: ")
            grade = self.grade_answer(question, user_answer, question_chunks)
            output_fn("Correct!" if grade["is_correct"] else "Incorrect.")
            output_fn(f"Correct answer: {', '.join(grade['correct_answer'])}")
            output_fn(grade["explanation"])
            difficulty = max(1, min(5, difficulty + (1 if grade["is_correct"] else -1)))
            result = {
                "question": question,
                "user_answer": user_answer,
                "grade": grade,
                "difficulty": difficulty,
                "generation_difficulty": max(1, min(5, difficulty - (1 if grade["is_correct"] else -1))),
                "source_paths": sorted({chunk.get("source_path") for chunk in question_chunks if chunk.get("source_path")}),
            }
            results.append(result)
            history.append({"question": question["question"], "answer": user_answer, "correct": grade["is_correct"]})

        if self.question_bank is not None:
            try:
                self.persist_questions(results)
            except Exception:
                logger.warning("Could not persist question-bank entries", exc_info=True)

        correct = sum(1 for result in results if result["grade"]["is_correct"])
        output_fn(f"\nFinal score: {correct}/{total_questions}")
        return {"topic": selected_topics, "total": total_questions, "correct": correct, "results": results}


