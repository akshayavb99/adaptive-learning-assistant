import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rag_retriever import RAGRetriever


class FakeCursor:
    def __init__(self, indexed_paths: set[str]):
        self.indexed_paths = indexed_paths
        self.parameters = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, query, parameters):
        self.parameters.append(parameters)

    def fetchone(self):
        path = self.parameters[-1][0]
        return (path in self.indexed_paths,)


class FakeConnection:
    def __init__(self, indexed_paths: set[str]):
        self.cursor_instance = FakeCursor(indexed_paths)

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        pass

    def rollback(self):
        pass


class FakeSearchCursor:
    def __init__(self, rows):
        self.rows = rows
        self.query = None
        self.parameters = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, query, parameters):
        self.query = query
        self.parameters = parameters

    def fetchall(self):
        return self.rows


class SearchConnection:
    def __init__(self, rows):
        self.cursor_instance = FakeSearchCursor(rows)

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        pass

    def rollback(self):
        pass


class FakeEmbedder:
    def __init__(self, vector):
        self.vector = vector
        self.queries = []

    def encode(self, query):
        self.queries.append(query)
        return self.vector


class RefreshIndexTests(unittest.TestCase):
    def test_recent_file_is_selected_even_when_indexed(self):
        with TemporaryDirectory() as directory:
            file = Path(directory, "recent.md")
            file.write_text("recent")
            source_path = str(file.resolve())
            connection = FakeConnection({source_path})

            with patch.dict(os.environ, {"DAYS_AGO": "3"}, clear=False):
                result = RAGRetriever(directory, connection, FakeEmbedder([1])).refresh_index()

            self.assertEqual(result, [source_path])
            self.assertEqual(len(connection.cursor_instance.parameters), 2)

    def test_old_missing_file_is_selected(self):
        with TemporaryDirectory() as directory:
            file = Path(directory, "old.md")
            file.write_text("old")
            old_time = time.time() - (4 * 24 * 60 * 60)
            os.utime(file, (old_time, old_time))
            source_path = str(file.resolve())
            connection = FakeConnection(set())

            with patch.dict(os.environ, {"DAYS_AGO": "3"}, clear=False):
                result = RAGRetriever(directory, connection, FakeEmbedder([1])).refresh_index()

            self.assertEqual(result, [source_path])
            self.assertEqual(len(connection.cursor_instance.parameters), 3)

    def test_old_indexed_file_is_skipped(self):
        with TemporaryDirectory() as directory:
            file = Path(directory, "old.md")
            file.write_text("old")
            old_time = time.time() - (4 * 24 * 60 * 60)
            os.utime(file, (old_time, old_time))
            source_path = str(file.resolve())
            connection = FakeConnection({source_path})

            with patch.dict(os.environ, {"DAYS_AGO": "3"}, clear=False):
                result = RAGRetriever(directory, connection, FakeEmbedder([1])).refresh_index()

            self.assertEqual(result, [])

    def test_invalid_days_ago_uses_default(self):
        with TemporaryDirectory() as directory:
            file = Path(directory, "old.md")
            file.write_text("old")
            old_time = time.time() - (4 * 24 * 60 * 60)
            os.utime(file, (old_time, old_time))
            source_path = str(file.resolve())
            connection = FakeConnection({source_path})

            with patch.dict(os.environ, {"DAYS_AGO": "not-an-integer"}, clear=False):
                result = RAGRetriever(directory, connection, FakeEmbedder([1])).refresh_index()

            self.assertEqual(result, [])


class SearchTests(unittest.TestCase):
    def test_search_returns_ranked_result_dicts(self):
        rows = [("docs/a.md", 0, "Python content", 0.91), ("docs/b.md", 1, "Other content", 0.72)]
        connection = SearchConnection(rows)
        embedder = FakeEmbedder([0.1, -0.2, 0.3])

        result = RAGRetriever(".", connection, embedder).search("how do I use Python?", 2)

        self.assertEqual(result, [
            {"source_path": "docs/a.md", "chunk_index": 0, "content": "Python content", "similarity": 0.91},
            {"source_path": "docs/b.md", "chunk_index": 1, "content": "Other content", "similarity": 0.72},
        ])
        self.assertEqual(embedder.queries, ["how do I use Python?"])
        self.assertEqual(connection.cursor_instance.parameters, ("[0.1,-0.2,0.3]", "[0.1,-0.2,0.3]", 2))
        self.assertIn("<=>", connection.cursor_instance.query)
        self.assertIn("LIMIT %s", connection.cursor_instance.query)

    def test_search_supports_array_like_embeddings(self):
        class ArrayLike:
            def tolist(self):
                return [1, 2]

        connection = SearchConnection([])
        RAGRetriever(".", connection, FakeEmbedder(ArrayLike())).search("query")
        self.assertEqual(connection.cursor_instance.parameters[0], "[1,2]")

    def test_search_rejects_invalid_arguments(self):
        retriever = RAGRetriever(".", SearchConnection([]), FakeEmbedder([]))
        for query in ("", "   ", None):
            with self.subTest(query=query):
                with self.assertRaises(ValueError):
                    retriever.search(query)
        for limit in (0, -1, 1.5, True):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    retriever.search("query", limit)


if __name__ == "__main__":
    unittest.main()

class InsertionCursor:
    def __init__(self, fail_on_insert=False):
        self.calls = []
        self.fail_on_insert = fail_on_insert

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, query, parameters):
        self.calls.append((query, parameters))
        if self.fail_on_insert and query.lstrip().startswith("INSERT"):
            raise RuntimeError("insert failed")


class InsertionConnection:
    def __init__(self, fail_on_insert=False):
        self.cursor_instance = InsertionCursor(fail_on_insert)
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        pass

    def rollback(self):
        pass

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class InsertionTests(unittest.TestCase):
    def test_empty_update_list_discovers_and_inserts_all_markdown_files(self):
        with TemporaryDirectory() as directory:
            first = Path(directory, "first.md")
            second = Path(directory, "nested", "second.md")
            second.parent.mkdir()
            first.write_text("# First\nFirst content", encoding="utf-8")
            second.write_text("# Second\nSecond content", encoding="utf-8")
            connection = InsertionConnection()
            embedder = FakeEmbedder([0.1, 0.2])
            retriever = RAGRetriever(directory, connection, embedder)

            result = retriever.insert_into_index()

            self.assertEqual(result, sorted([str(first.resolve()), str(second.resolve())]))
            self.assertEqual(embedder.queries, ["# First\n\nFirst content", "# Second\n\nSecond content"])
            self.assertEqual(connection.commits, 2)
            self.assertEqual(connection.rollbacks, 0)
            insert_calls = [call for call in connection.cursor_instance.calls if call[0].lstrip().startswith("INSERT")]
            self.assertEqual(len(insert_calls), 2)
            self.assertIn("ON CONFLICT (source_path, chunk_index)", insert_calls[0][0])

    def test_nonempty_update_list_processes_only_selected_file(self):
        with TemporaryDirectory() as directory:
            selected = Path(directory, "selected.md")
            ignored = Path(directory, "ignored.md")
            selected.write_text("selected", encoding="utf-8")
            ignored.write_text("ignored", encoding="utf-8")
            connection = InsertionConnection()
            retriever = RAGRetriever(directory, connection, FakeEmbedder([1]))
            retriever.update_list = [str(selected)]

            result = retriever.insert_into_index()

            self.assertEqual(result, [str(selected.resolve())])
            self.assertEqual(connection.commits, 1)

    def test_structure_aware_chunking_preserves_heading_context_and_paragraphs(self):
        content = "# Intro\nintro text\n\n## Details\nfirst paragraph\n\nsecond paragraph"

        chunks = RAGRetriever._chunk_document(content)

        self.assertEqual(chunks, [
            "# Intro\n\nintro text",
            "# Intro\n## Details\n\nfirst paragraph",
            "# Intro\n## Details\n\nsecond paragraph",
        ])

    def test_structure_aware_chunking_keeps_lists_and_code_fences_atomic(self):
        content = (
            "# Examples\n\n"
            "- first item\n\n"
            "- second item\n\n"
            "```python\n"
            "first = 1\n\n"
            "second = 2\n"
            "```"
        )

        chunks = RAGRetriever._chunk_document(content)

        self.assertEqual(len(chunks), 2)
        self.assertIn("- first item\n\n- second item", chunks[0])
        self.assertIn("first = 1\n\nsecond = 2", chunks[1])

    def test_low_information_fragments_do_not_create_chunks(self):
        chunks = RAGRetriever._chunk_document("# Topic\n\nwhere")
        self.assertEqual(chunks, [])

    def test_structural_separators_do_not_create_heading_only_chunks(self):
        content = "# Logistic Regression\n\n## Definition\n\n---\n\nLogistic regression predicts probabilities for a binary outcome."

        chunks = RAGRetriever._chunk_document(content)

        self.assertEqual(chunks, [
            "# Logistic Regression\n## Definition\n\nLogistic regression predicts probabilities for a binary outcome."
        ])
        self.assertTrue(all("---" not in chunk for chunk in chunks))
    def test_long_paragraph_fallback_uses_sentence_boundaries(self):
        sentence = "This is a complete sentence. "
        content = "# Details\n\n" + (sentence * 30).strip()

        chunks = RAGRetriever._chunk_document(content)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.startswith("# Details\n\n") for chunk in chunks))
        self.assertTrue(all(chunk.endswith(".") for chunk in chunks))

    def test_insert_failure_rolls_back_and_does_not_report_file(self):
        with TemporaryDirectory() as directory:
            file = Path(directory, "broken.md")
            file.write_text("broken", encoding="utf-8")
            connection = InsertionConnection(fail_on_insert=True)
            retriever = RAGRetriever(directory, connection, FakeEmbedder([1]))
            retriever.update_list = [str(file)]

            with self.assertRaises(RuntimeError):
                retriever.insert_into_index()

            self.assertEqual(connection.commits, 0)
            self.assertEqual(connection.rollbacks, 1)

class HybridSearchTests(unittest.TestCase):
    def test_hybrid_search_returns_rrf_ranked_results(self):
        rows = [
            ("docs/b.md", 1, "Both matches", 0.8, 0.5, 1, 1, 0.03, 1),
            ("docs/a.md", 0, "Vector only", 0.7, None, 2, None, 0.016, 2),
            ("docs/c.md", 0, "Lexical only", None, 0.9, None, 2, 0.015, 3),
        ]
        connection = SearchConnection(rows)
        embedder = FakeEmbedder([0.1, 0.2, 0.3])

        result = RAGRetriever(".", connection, embedder).search_hybrid("python vectors", 3)

        self.assertEqual(result[0], {
            "source_path": "docs/b.md",
            "chunk_index": 1,
            "content": "Both matches",
            "vector_similarity": 0.8,
            "lexical_score": 0.5,
            "vector_rank": 1,
            "lexical_rank": 1,
            "hybrid_score": 0.03,
            "hybrid_rank": 1,
        })
        self.assertEqual(embedder.queries, ["python vectors"])
        self.assertEqual(connection.cursor_instance.parameters[0], "[0.1,0.2,0.3]")
        self.assertEqual(connection.cursor_instance.parameters[1], 20)
        self.assertEqual(connection.cursor_instance.parameters[2:4], ("python vectors", "python vectors"))
        self.assertEqual(connection.cursor_instance.parameters[-1], 3)
        self.assertIn("websearch_to_tsquery", connection.cursor_instance.query)
        self.assertIn("ts_rank_cd", connection.cursor_instance.query)
        self.assertIn("rrf_score", connection.cursor_instance.query)

    def test_hybrid_search_rejects_invalid_arguments(self):
        retriever = RAGRetriever(".", SearchConnection([]), FakeEmbedder([]))
        with self.assertRaises(ValueError):
            retriever.search_hybrid(" ")
        with self.assertRaises(ValueError):
            retriever.search_hybrid("query", 0)

class RefreshWriteTests(unittest.TestCase):
    def test_refresh_deletes_and_reinserts_recent_file(self):
        with TemporaryDirectory() as directory:
            file = Path(directory, "changed.md")
            file.write_text("# Changed\nnew content", encoding="utf-8")
            connection = InsertionConnection()
            retriever = RAGRetriever(directory, connection, FakeEmbedder([1]))

            result = retriever.refresh_index()

            self.assertEqual(result, [str(file.resolve())])
            self.assertEqual(connection.commits, 1)
            calls = connection.cursor_instance.calls
            self.assertIn("DELETE FROM documents WHERE source_path = %s", calls[0][0])
            self.assertTrue(calls[1][0].lstrip().startswith("INSERT"))
    def test_refresh_with_no_updates_is_a_noop(self):
        with TemporaryDirectory() as directory:
            file = Path(directory, "old.md")
            file.write_text("old", encoding="utf-8")
            old_time = time.time() - (4 * 24 * 60 * 60)
            os.utime(file, (old_time, old_time))
            source_path = str(file.resolve())
            connection = FakeConnection({source_path})
            retriever = RAGRetriever(directory, connection, FakeEmbedder([1]))

            result = retriever.refresh_index()

            self.assertEqual(result, [])
            self.assertEqual(connection.cursor_instance.parameters, [(source_path,)])





