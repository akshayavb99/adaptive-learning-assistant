import unittest

from performance import PerformanceRecorder


class Cursor:
    def __init__(self): self.calls = []
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def execute(self, query, params=None): self.calls.append((query, params))

class Connection:
    def __init__(self): self.cursor_instance = Cursor(); self.commits = 0
    def cursor(self): return self.cursor_instance
    def commit(self): self.commits += 1

class PerformanceRecorderTests(unittest.TestCase):
    def test_records_session_answer_and_completion(self):
        connection = Connection()
        recorder = PerformanceRecorder(connection)
        test_id = recorder.start_test("00000000-0000-0000-0000-000000000001", ["Python"], 5)
        recorder.record_answer(test_id, 1, "short_answer", 3, 4, True)
        recorder.finish_test(test_id, completed=True, ended_early=False)
        queries = "\n".join(query for query, _ in connection.cursor_instance.calls)
        self.assertIn("INSERT INTO test_sessions", queries)
        self.assertIn("INSERT INTO test_answers", queries)
        self.assertIn("UPDATE test_sessions", queries)
        self.assertEqual(connection.commits, 4)

if __name__ == "__main__": unittest.main()