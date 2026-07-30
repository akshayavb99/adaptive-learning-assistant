import json
from typing import Any

from database import ensure_question_bank_schema


DEFAULT_SIMILARITY_THRESHOLD = 0.95


class QuestionBankStore:
    """Persist generated questions while rejecting semantic duplicates."""

    def __init__(
        self,
        connection: Any,
        embedder: Any,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        if not 0 < similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        self.connection = connection
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        ensure_question_bank_schema(connection)

    @staticmethod
    def _vector_to_string(vector: Any) -> str:
        values = vector.tolist() if hasattr(vector, "tolist") else vector
        return "[" + ",".join(str(value) for value in values) + "]"

    @staticmethod
    def _source_paths(record: dict[str, Any]) -> list[str]:
        paths = record.get("source_paths")
        if paths is None:
            paths = [chunk.get("source_path") for chunk in record.get("chunks", [])]
        return sorted({path for path in paths if isinstance(path, str) and path})

    def list_questions(
        self,
        question_type: str | None = None,
        difficulty: int | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        """Return recent question-bank entries without exposing embeddings."""
        if question_type not in {None, "short_answer", "single_choice", "multiple_choice"}:
            raise ValueError("invalid question_type")
        if difficulty is not None and difficulty not in range(1, 6):
            raise ValueError("difficulty must be between 1 and 5")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")

        conditions = []
        parameters: list[Any] = []
        if question_type is not None:
            conditions.append("question_type = %s")
            parameters.append(question_type)
        if difficulty is not None:
            conditions.append("difficulty = %s")
            parameters.append(difficulty)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT id, question_type, question_text, options, correct_answer,
                       explanation, source_paths, difficulty, created_at,
                       COUNT(*) OVER() AS total_count
                FROM question_bank
                {where}
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (*parameters, limit),
            )
            rows = cursor.fetchall()

        questions = [
            {
                "id": row[0],
                "question_type": row[1],
                "question": row[2],
                "options": row[3],
                "correct_answer": row[4],
                "explanation": row[5],
                "source_paths": row[6],
                "difficulty": row[7],
                "created_at": row[8],
            }
            for row in rows
        ]
        return {"questions": questions, "total": int(rows[0][9]) if rows else 0}

    def store_questions(self, records: list[dict[str, Any]]) -> dict[str, int]:
        inserted = 0
        skipped = 0
        try:
            with self.connection.cursor() as cursor:
                for record in records:
                    question = record.get("question", record)
                    question_text = question.get("question")
                    if not isinstance(question_text, str) or not question_text.strip():
                        raise ValueError("question records must contain non-empty question text")
                    difficulty = record.get("generation_difficulty", record.get("difficulty"))
                    if isinstance(difficulty, bool) or difficulty not in range(1, 6):
                        raise ValueError("question difficulty must be between 1 and 5")

                    vector = self._vector_to_string(self.embedder.encode(question_text))
                    cursor.execute(
                        """
                        SELECT 1 - (embedding <=> %s::vector) AS similarity
                        FROM question_bank
                        ORDER BY embedding <=> %s::vector
                        LIMIT 1
                        """,
                        (vector, vector),
                    )
                    nearest = cursor.fetchone()
                    if nearest is not None and float(nearest[0]) >= self.similarity_threshold:
                        skipped += 1
                        continue

                    cursor.execute(
                        """
                        INSERT INTO question_bank (
                            question_type, question_text, options, correct_answer,
                            explanation, source_paths, difficulty, embedding
                        )
                        VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s::vector)
                        """,
                        (
                            question.get("question_type"),
                            question_text.strip(),
                            json.dumps(question.get("options", []), ensure_ascii=False),
                            json.dumps(question.get("expected_answer", []), ensure_ascii=False),
                            question.get("explanation", ""),
                            self._source_paths(record),
                            difficulty,
                            vector,
                        ),
                    )
                    inserted += 1
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return {"inserted": inserted, "skipped": skipped}
