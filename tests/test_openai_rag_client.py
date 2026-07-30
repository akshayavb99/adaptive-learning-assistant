import json
import os
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from openai_rag_client import OpenAIRAGClient, parse_topic_input


class FakeResponses:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.outputs:
            raise AssertionError("unexpected OpenAI call")
        return SimpleNamespace(output_text=self.outputs.pop(0))


class FakeOpenAI:
    def __init__(self, outputs):
        self.responses = FakeResponses(outputs)


class FakeRetriever:
    input_path = None

    def __init__(self, topics=None):
        self.topics = ["python/basics"] if topics is None else topics
        self.calls = []

    def list_topics(self):
        return self.topics

    def search_hybrid(self, query, num_results):
        self.calls.append(("hybrid", query, num_results))
        return [{"source_path": f"{query}.md", "content": f"{query} context", "hybrid_score": 0.5}]

    def search_index(self, query, num_results):
        self.calls.append(("vector", query, num_results))
        return [{"source_path": f"{query}.md", "content": f"{query} context", "hybrid_score": 0.5}]


def question(question_type="short_answer"):
    options = ["list", "tuple"] if question_type != "short_answer" else []
    expected = [options[0]] if options else ["A list is mutable"]
    return {
        "question_type": question_type,
        "question": "What is the answer?",
        "options": options,
        "expected_answer": expected,
        "explanation": "The context supports this answer.",
    }


def grade(correct):
    return {
        "is_correct": correct,
        "correct_answer": ["A list is mutable"],
        "feedback": "Good." if correct else "Review this concept.",
        "explanation": "The context supports the expected answer.",
    }



def judge_payload(scores=(5, 5, 5, 5)):
    return {
        "groundedness_score": scores[0],
        "answer_correctness_score": scores[1],
        "clarity_score": scores[2],
        "difficulty_score": scores[3],
        "groundedness_explanation": "Grounded.",
        "answer_correctness_explanation": "Correct.",
        "clarity_explanation": "Clear.",
        "difficulty_explanation": "Appropriate.",
    }

class TopicInputTests(unittest.TestCase):
    def test_parse_topic_input_trims_deduplicates_and_ignores_empty_values(self):
        self.assertEqual(
            parse_topic_input(" Python, PostgreSQL, , Docker, Python "),
            ["Python", "PostgreSQL", "Docker"],
        )

    def test_parse_topic_input_returns_empty_for_blank_input(self):
        self.assertEqual(parse_topic_input(" ,  , "), [])

class OpenAIRAGClientTests(unittest.TestCase):
    def test_judge_model_can_be_configured_independently(self):
        with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-5.4-nano", "OPENAI_JUDGE_MODEL": "gpt-5.4-mini"}, clear=False):
            wrapper = OpenAIRAGClient(FakeRetriever(), client=FakeOpenAI([]))

        self.assertEqual(wrapper.model, "gpt-5.4-nano")
        self.assertEqual(wrapper.judge_model, "gpt-5.4-mini")

    def test_retrieve_calls_retriever_directly(self):
        retriever = FakeRetriever()
        wrapper = OpenAIRAGClient(retriever, client=FakeOpenAI([]), num_results=3)

        result = wrapper.retrieve("Python lists")

        self.assertEqual(result[0]["content"], "Python lists context")
        self.assertEqual(retriever.calls, [("hybrid", "Python lists", 3)])

    def test_multiple_topics_search_separately_and_deduplicate_topics(self):
        retriever = FakeRetriever()
        wrapper = OpenAIRAGClient(retriever, client=FakeOpenAI([]), num_results=5)

        result = wrapper.retrieve(["first", "second", "first", " "])

        self.assertEqual([chunk["content"] for chunk in result], ["first context", "second context"])
        self.assertEqual(retriever.calls, [("hybrid", "first", 5), ("hybrid", "second", 5)])

    def test_multiple_topics_keep_results_per_topic_and_merge_shared_chunks(self):
        class WideRetriever(FakeRetriever):
            def search_hybrid(self, query, num_results):
                self.calls.append(("hybrid", query, num_results))
                chunks = [
                    {
                        "source_path": f"{query}/{index}.md",
                        "chunk_index": index,
                        "content": f"{query} context {index}",
                        "hybrid_score": 1.0 - index / 10,
                    }
                    for index in range(num_results)
                ]
                if query in {"first", "second"}:
                    chunks[0] = {
                        "source_path": "shared.md",
                        "chunk_index": 0,
                        "content": "shared context",
                        "hybrid_score": 1.0,
                    }
                return chunks

        retriever = WideRetriever()
        wrapper = OpenAIRAGClient(retriever, client=FakeOpenAI([]), num_results=3)

        result = wrapper.retrieve(["first", "second"])

        self.assertEqual(len(result), 5)
        self.assertEqual(retriever.calls, [("hybrid", "first", 3), ("hybrid", "second", 3)])
        shared = next(chunk for chunk in result if chunk["source_path"] == "shared.md")
        self.assertEqual(shared["matched_topics"], ["first", "second"])

    def test_context_includes_matched_topics(self):
        context = OpenAIRAGClient._context([{
            "source_path": "python.md",
            "content": "Python context",
            "matched_topics": ["Python", "Programming"],
        }])
        self.assertIn("Topics: Python, Programming", context)
        self.assertIn("Python context", context)

    def test_vector_mode_dispatches_to_vector_search(self):
        retriever = FakeRetriever()
        wrapper = OpenAIRAGClient(
            retriever,
            client=FakeOpenAI([]),
            search_mode="vector",
        )

        wrapper.retrieve("exact query")

        self.assertEqual(retriever.calls, [("vector", "exact query", 5)])

    def test_random_topic_selection_is_deterministic_with_injected_rng(self):
        retriever = FakeRetriever(["first", "second"])
        wrapper = OpenAIRAGClient(
            retriever,
            client=FakeOpenAI([]),
            rng=random.Random(1),
        )

        self.assertEqual(wrapper.select_random_topic(), "first")

    def test_question_and_grading_use_structured_outputs(self):
        fake = FakeOpenAI([json.dumps(question("single_choice")), json.dumps(grade(True))])
        wrapper = OpenAIRAGClient(FakeRetriever(), client=fake)
        chunks = wrapper.retrieve("Python")

        generated = wrapper.generate_question(["Python", "Databases"], chunks, 3)
        result = wrapper.grade_answer(generated, "list", chunks)

        self.assertEqual(generated["question_type"], "single_choice")
        self.assertTrue(result["is_correct"])
        self.assertEqual(fake.responses.calls[0]["text"]["format"]["name"], "test_question")
        self.assertEqual(fake.responses.calls[1]["text"]["format"]["name"], "answer_grade")
        self.assertIn("Python context", fake.responses.calls[0]["input"])
        self.assertIn("Python; Databases", fake.responses.calls[0]["input"])

    def test_failed_pair_is_regenerated_before_returning(self):
        outputs = [json.dumps(question()), json.dumps(judge_payload((3, 3, 3, 3))), json.dumps(question()), json.dumps(judge_payload())]
        wrapper = OpenAIRAGClient(FakeRetriever(), client=FakeOpenAI(outputs))
        chunks = wrapper.retrieve("Python")
        result = wrapper.generate_validated_question("Python", chunks, 3)
        self.assertTrue(result["quality_judgment"]["passed"])
        self.assertEqual(result["quality_judgment"]["overall_score"], 5.0)

    def test_duplicate_question_is_rejected_and_regenerated(self):
        duplicate = question()
        unique = question()
        unique["question"] = "What is a different answer?"
        outputs = [
            json.dumps(duplicate),
            json.dumps(unique),
            json.dumps(judge_payload()),
        ]
        wrapper = OpenAIRAGClient(FakeRetriever(), client=FakeOpenAI(outputs))
        chunks = wrapper.retrieve("Python")

        result = wrapper.generate_validated_question(
            "Python", chunks, 3,
            history=[{"question": "  what   IS the answer? "}],
        )

        self.assertEqual(result["question"], "What is a different answer?")
        self.assertEqual(len(wrapper.client.responses.calls), 3)

    def test_persist_questions_excludes_failed_judgments(self):
        class Store:
            def store_questions(self, records):
                self.records = records
                return {"inserted": len(records), "skipped": 0}
        store = Store()
        wrapper = OpenAIRAGClient(FakeRetriever(), client=FakeOpenAI([]), question_bank=store)
        failed = {"question": dict(question(), quality_judgment={"passed": False})}
        passed = {"question": dict(question(), quality_judgment={"passed": True})}
        wrapper.persist_questions([failed, passed])
        self.assertEqual(store.records, [passed])

    def test_run_test_adapts_difficulty_and_defaults_are_supported(self):
        second_question = question()
        second_question["question"] = "What is a different answer?"
        outputs = [
            json.dumps(question()), json.dumps(judge_payload()), json.dumps(grade(True)),
            json.dumps(second_question), json.dumps(judge_payload()), json.dumps(grade(False)),
        ]
        wrapper = OpenAIRAGClient(FakeRetriever(), client=FakeOpenAI(outputs))
        answers = iter(["A list is mutable", "wrong"])
        seen = []

        summary = wrapper.run_test(
            topic="python/basics",
            num_questions=2,
            answer_fn=lambda prompt: next(answers),
            output_fn=seen.append,
        )

        self.assertEqual(summary["topic"], ["python/basics"])
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual([item["difficulty"] for item in summary["results"]], [4, 3])
        self.assertEqual(len(wrapper.retriever.calls), 1)

    def test_run_test_rejects_invalid_count_and_empty_topics(self):
        wrapper = OpenAIRAGClient(FakeRetriever(), client=FakeOpenAI([]))
        with self.assertRaises(ValueError):
            wrapper.run_test(num_questions=0)
        with self.assertRaises(ValueError):
            wrapper.run_test(topic=" ")

    def test_missing_topics_fail_clearly(self):
        wrapper = OpenAIRAGClient(FakeRetriever([]), client=FakeOpenAI([]))
        with self.assertRaisesRegex(RuntimeError, "No Markdown"):
            wrapper.select_random_topic()


    def test_schedule_chunks_is_round_robin_and_balanced(self):
        chunks = [
            {"source_path": "a.md", "chunk_index": 0, "content": "a0"},
            {"source_path": "a.md", "chunk_index": 1, "content": "a1"},
            {"source_path": "b.md", "chunk_index": 0, "content": "b0"},
        ]

        schedule = OpenAIRAGClient._schedule_chunks(chunks, 8)

        self.assertEqual(
            [(chunk["source_path"], chunk["chunk_index"]) for chunk in schedule],
            [
                ("a.md", 0), ("a.md", 1), ("b.md", 0),
                ("a.md", 0), ("a.md", 1), ("b.md", 0),
                ("a.md", 0), ("a.md", 1),
            ],
        )
        self.assertEqual(
            {key: sum(1 for chunk in schedule if (chunk["source_path"], chunk["chunk_index"]) == key)
             for key in [("a.md", 0), ("a.md", 1), ("b.md", 0)]},
            {("a.md", 0): 3, ("a.md", 1): 3, ("b.md", 0): 2},
        )

    def test_schedule_single_chunk_stops_after_two_questions(self):
        chunk = {"source_path": "only.md", "chunk_index": 0, "content": "only"}

        schedule = OpenAIRAGClient._schedule_chunks([chunk], 10)

        self.assertEqual(schedule, [chunk, chunk])

    def test_run_test_passes_one_scheduled_chunk_per_question(self):
        class WideRetriever(FakeRetriever):
            def search_hybrid(self, query, num_results):
                self.calls.append(("hybrid", query, num_results))
                return [
                    {"source_path": "a.md", "chunk_index": 0, "content": "a0"},
                    {"source_path": "b.md", "chunk_index": 0, "content": "b0"},
                    {"source_path": "c.md", "chunk_index": 0, "content": "c0"},
                ]

        class SchedulingClient(OpenAIRAGClient):
            def __init__(self, retriever):
                super().__init__(retriever, client=FakeOpenAI([]))
                self.generated_chunks = []
                self.graded_chunks = []

            def generate_validated_question(self, topic, chunks, difficulty, history=None):
                self.generated_chunks.append(chunks[0]["source_path"])
                return question()

            def grade_answer(self, generated_question, user_answer, chunks):
                self.graded_chunks.append(chunks[0]["source_path"])
                return grade(True)

        retriever = WideRetriever(["topic"])
        wrapper = SchedulingClient(retriever)
        answers = iter(["answer"] * 5)

        summary = wrapper.run_test(
            topic="topic",
            num_questions=5,
            answer_fn=lambda prompt: next(answers),
            output_fn=lambda output: None,
        )

        self.assertEqual(wrapper.generated_chunks, ["a.md", "b.md", "c.md", "a.md", "b.md"])
        self.assertEqual(wrapper.graded_chunks, wrapper.generated_chunks)
        self.assertEqual(summary["total"], 5)

    def test_run_test_without_topic_retrieves_one_chunk_per_question(self):
        class NoTopicRetriever(FakeRetriever):
            def search_hybrid(self, query, num_results):
                self.calls.append(("hybrid", query, num_results))
                return [
                    {
                        "source_path": f"{query}/{index}.md",
                        "chunk_index": index,
                        "content": f"{query} context {index}",
                    }
                    for index in range(num_results)
                ]

        class SchedulingClient(OpenAIRAGClient):
            def __init__(self, retriever):
                super().__init__(retriever, client=FakeOpenAI([]))
                self.generated_chunks = []

            def generate_validated_question(self, topic, chunks, difficulty, history=None):
                self.generated_chunks.append(
                    (chunks[0]["source_path"], chunks[0]["chunk_index"])
                )
                return question()

            def grade_answer(self, generated_question, user_answer, chunks):
                return grade(True)

        retriever = NoTopicRetriever(["first", "second"])
        wrapper = SchedulingClient(retriever)
        answers = iter(["answer"] * 5)

        summary = wrapper.run_test(
            topic=None,
            num_questions=5,
            answer_fn=lambda prompt: next(answers),
            output_fn=lambda output: None,
        )

        self.assertEqual(summary["total"], 5)
        self.assertEqual(len(wrapper.generated_chunks), 5)
        self.assertEqual(len(set(wrapper.generated_chunks)), 5)
        self.assertEqual(
            retriever.calls,
            [("hybrid", "first", 3), ("hybrid", "second", 2)],
        )


if __name__ == "__main__":
    unittest.main()


