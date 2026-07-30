import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import psycopg
    from database import get_connection
    from embedding import FastEmbedder
    from rag_retriever import RAGRetriever
except ImportError:
    psycopg = None


@unittest.skipUnless(
    os.getenv("RUN_PGVECTOR_INTEGRATION") == "1",
    "set RUN_PGVECTOR_INTEGRATION=1 to run Docker pgvector integration tests",
)
class PgvectorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if psycopg is None:
            raise unittest.SkipTest("database/embedding dependencies are unavailable")
        try:
            cls.connection = get_connection()
            cls.connection.execute("SELECT 1")
            cls.embedder = FastEmbedder()
        except Exception as exc:
            raise unittest.SkipTest(f"pgvector or FastEmbed unavailable: {exc}")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "connection"):
            cls.connection.close()

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.file = Path(self.directory.name, "integration.md")
        self.file.write_text("# Initial\n" + ("initial content " * 100), encoding="utf-8")
        self.source_path = str(self.file.resolve())
        self.retriever = RAGRetriever(self.directory.name, self.connection, self.embedder)
        self.connection.execute("DELETE FROM documents WHERE source_path = %s", (self.source_path,))
        self.connection.commit()

    def tearDown(self):
        self.connection.execute("DELETE FROM documents WHERE source_path = %s", (self.source_path,))
        self.connection.commit()
        self.directory.cleanup()

    def test_insert_and_refresh_replace_source_chunks(self):
        self.retriever.update_list = [self.source_path]
        inserted = self.retriever.insert_into_index()
        self.assertEqual(inserted, [self.source_path])

        initial_count = self.connection.execute(
            "SELECT count(*) FROM documents WHERE source_path = %s",
            (self.source_path,),
        ).fetchone()[0]
        self.assertGreater(initial_count, 0)
        dimensions = self.connection.execute(
            "SELECT vector_dims(embedding) FROM documents WHERE source_path = %s LIMIT 1",
            (self.source_path,),
        ).fetchone()[0]
        self.assertEqual(dimensions, int(os.getenv("VECTOR_DIMENSIONS", "384")))

        self.file.write_text("# Updated\nupdated content", encoding="utf-8")
        now = time.time()
        os.utime(self.file, (now, now))
        refreshed = self.retriever.refresh_index()
        self.assertEqual(refreshed, [self.source_path])

        rows = self.connection.execute(
            "SELECT count(*), string_agg(content, ' ') FROM documents WHERE source_path = %s",
            (self.source_path,),
        ).fetchone()
        self.assertEqual(rows[0], 1)
        self.assertIn("updated content", rows[1])

    def test_unchanged_source_is_not_refreshed(self):
        self.retriever.update_list = [self.source_path]
        self.retriever.insert_into_index()
        old_time = time.time() - (4 * 24 * 60 * 60)
        os.utime(self.file, (old_time, old_time))
        self.assertEqual(self.retriever.refresh_index(), [])


if __name__ == "__main__":
    unittest.main()

