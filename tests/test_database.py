import os
import unittest
from unittest.mock import patch

from database import initialize_database


class FakeCursor:
    def __init__(self, fail=False):
        self.statements = []
        self.fail = fail

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, statement, *parameters):
        self.statements.append(statement)
        if self.fail and len(self.statements) == 2:
            raise RuntimeError("schema failure")


class FakeConnection:
    def __init__(self, fail=False):
        self.cursor_instance = FakeCursor(fail)
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


class DatabaseInitializationTests(unittest.TestCase):
    def test_initializes_all_schema_objects_in_one_transaction(self):
        connection = FakeConnection()
        with patch.dict(os.environ, {"VECTOR_DIMENSIONS": "384"}, clear=False):
            initialize_database(connection)

        self.assertEqual(connection.commits, 1)
        self.assertEqual(connection.rollbacks, 0)
        sql = "\n".join(connection.cursor_instance.statements)
        for name in (
            "CREATE EXTENSION IF NOT EXISTS vector",
            "CREATE TABLE IF NOT EXISTS documents",
            "CREATE TABLE IF NOT EXISTS openai_usage",
            "CREATE TABLE IF NOT EXISTS question_bank",
            "CREATE TABLE IF NOT EXISTS retrieval_evaluation_runs",
            "CREATE TABLE IF NOT EXISTS retrieval_evaluation_results",
            "CREATE TABLE IF NOT EXISTS question_quality_judgments",
        ):
            self.assertIn(name, sql)

    def test_rolls_back_when_schema_creation_fails(self):
        connection = FakeConnection(fail=True)
        with self.assertRaises(RuntimeError):
            initialize_database(connection)
        self.assertEqual(connection.commits, 0)
        self.assertEqual(connection.rollbacks, 1)

    def test_rejects_invalid_vector_dimensions(self):
        connection = FakeConnection()
        with patch.dict(os.environ, {"VECTOR_DIMENSIONS": "0"}, clear=False):
            with self.assertRaises(ValueError):
                initialize_database(connection)
        self.assertEqual(connection.commits, 0)


if __name__ == "__main__":
    unittest.main()



