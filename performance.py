import uuid
from typing import Any, Sequence

from database import ensure_performance_schema


class PerformanceRecorder:
    """Persist anonymous test-session and aggregate answer performance."""

    def __init__(self, connection: Any):
        self.connection = connection
        ensure_performance_schema(connection)

    def start_test(self, anonymous_session_id: str, topics: Sequence[str], requested_questions: int) -> str:
        test_id = str(uuid.uuid4())
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO test_sessions (id, anonymous_session_id, topics, requested_questions)
                VALUES (%s, %s, %s, %s)
                """,
                (test_id, anonymous_session_id, list(topics), requested_questions),
            )
        self.connection.commit()
        return test_id

    def record_answer(
        self, test_id: str, question_number: int, question_type: str,
        assigned_difficulty: int, next_difficulty: int, is_correct: bool,
    ) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO test_answers (
                    session_id, question_number, question_type, assigned_difficulty,
                    next_difficulty, is_correct
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id, question_number) DO NOTHING
                """,
                (test_id, question_number, question_type, assigned_difficulty,
                 next_difficulty, is_correct),
            )
            cursor.execute(
                """
                UPDATE test_sessions
                SET answered_questions = (SELECT COUNT(*) FROM test_answers WHERE session_id = %s),
                    correct_answers = (SELECT COUNT(*) FROM test_answers WHERE session_id = %s AND is_correct)
                WHERE id = %s
                """,
                (test_id, test_id, test_id),
            )
        self.connection.commit()

    def finish_test(self, test_id: str, completed: bool, ended_early: bool) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE test_sessions
                SET completed_at = NOW(), completed = %s, ended_early = %s
                WHERE id = %s AND completed_at IS NULL
                """,
                (completed, ended_early, test_id),
            )
        self.connection.commit()