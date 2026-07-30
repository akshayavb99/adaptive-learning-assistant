import uuid


import streamlit as st

from database import get_connection
from embedding import FastEmbedder
from openai_rag_client import OpenAIRAGClient, parse_topic_input
from performance import PerformanceRecorder
from question_bank import QuestionBankStore
from usage import UsageRecorder
from rag_retriever import RAGRetriever


st.set_page_config(page_title="Adaptive Testing Assistant", layout="centered")


@st.cache_resource
def get_runtime():
    connection = get_connection()
    embedder = FastEmbedder()
    retriever = RAGRetriever("data", connection, embedder)
    processed = retriever.refresh_index()
    question_bank = QuestionBankStore(connection, embedder)
    client = OpenAIRAGClient(retriever, usage_recorder=UsageRecorder(connection), question_bank=question_bank)
    performance = PerformanceRecorder(connection)
    return retriever, client, performance, processed


def initialize_state() -> None:
    defaults = {
        "anonymous_session_id": str(uuid.uuid4()),
        "test_id": None,
        "test_active": False,
        "ended_early": False,
        "selected_topics": None,
        "num_questions": 20,
        "question_number": 0,
        "scheduled_chunks": [],
        "difficulty": 3,
        "current_question": None,
        "current_chunks": None,
        "current_answer": None,
        "current_grade": None,
        "answer_submitted": False,
        "history": [],
        "results": [],
        "question_bank_persisted": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_current_question() -> None:
    answer_key = f"answer_{st.session_state.question_number}"
    st.session_state.pop(answer_key, None)
    st.session_state.current_question = None
    st.session_state.current_chunks = None
    st.session_state.current_answer = None
    st.session_state.current_grade = None
    st.session_state.answer_submitted = False


def prepare_question(client: OpenAIRAGClient) -> None:
    topic = st.session_state.selected_topics
    chunk_index = st.session_state.question_number - 1
    schedule = st.session_state.scheduled_chunks
    if not schedule or chunk_index >= len(schedule):
        raise RuntimeError("No scheduled chunk is available for this question")
    chunks = [schedule[chunk_index]]
    question = client.generate_validated_question(
        topic,
        chunks,
        st.session_state.difficulty,
        st.session_state.history,
    )
    st.session_state.current_chunks = chunks
    st.session_state.current_question = question
    st.session_state.current_answer = None
    st.session_state.current_grade = None
    st.session_state.answer_submitted = False


def reset_test() -> None:
    st.session_state.test_active = False
    st.session_state.ended_early = False
    st.session_state.selected_topics = None
    st.session_state.scheduled_chunks = []
    st.session_state.test_id = None
    st.session_state.num_questions = 20
    st.session_state.question_number = 0
    st.session_state.difficulty = 3
    st.session_state.history = []
    st.session_state.results = []
    st.session_state.question_bank_persisted = False
    clear_current_question()


def persist_test_questions(client: OpenAIRAGClient, include_current: bool = False) -> None:
    if st.session_state.question_bank_persisted:
        return
    records = list(st.session_state.results)
    if include_current and st.session_state.current_question is not None:
        records.append({
            "question": st.session_state.current_question,
            "generation_difficulty": st.session_state.difficulty,
            "source_paths": sorted({
                chunk.get("source_path")
                for chunk in (st.session_state.current_chunks or [])
                if chunk.get("source_path")
            }),
        })
    try:
        stats = client.persist_questions(records)
        if stats is not None:
            st.info(
                f"Question bank: stored {stats['inserted']} new question(s); "
                f"skipped {stats['skipped']} duplicate(s)."
            )
        st.session_state.question_bank_persisted = True
    except Exception:
        st.warning("Question-bank persistence failed; the test result was preserved.")
        import logging
        logging.getLogger(__name__).warning(
            "Could not persist question-bank entries", exc_info=True
        )


def end_test(client: OpenAIRAGClient, performance: PerformanceRecorder) -> None:
    persist_test_questions(client, include_current=True)
    if st.session_state.test_id:
        performance.finish_test(st.session_state.test_id, completed=False, ended_early=True)
    st.session_state.test_active = False
    st.session_state.ended_early = True
    clear_current_question()


def answer_widget(question: dict) -> str:
    question_type = question["question_type"]
    options = question["options"]
    key = f"answer_{st.session_state.question_number}"
    if question_type == "short_answer":
        return st.text_input("Your answer", key=key)
    if question_type == "single_choice":
        selected = st.radio("Choose one answer", options, key=key)
        return selected or ""
    selected = st.multiselect("Choose all that apply", options, key=key)
    return ", ".join(selected)


def render_submitted_feedback() -> None:
    grade = st.session_state.current_grade
    if grade is None:
        return
    with st.chat_message("user"):
        st.markdown("**Your answer**")
        st.write(st.session_state.current_answer)
    with st.chat_message("assistant"):
        label = "Correct" if grade["is_correct"] else "Incorrect"
        st.markdown(f"**{label}**")
        st.markdown("**Correct answer**")
        st.write(", ".join(grade["correct_answer"]))
        st.markdown("**Explanation**")
        st.write(grade["explanation"])
        if grade.get("feedback"):
            st.write(grade["feedback"])


def render_summary() -> None:
    correct = sum(1 for result in st.session_state.results if result["grade"]["is_correct"])
    answered = len(st.session_state.results)
    if st.session_state.ended_early:
        st.warning(
            f"Test ended early. Score: {correct}/{answered} "
            f"({answered}/{st.session_state.num_questions} answered)"
        )
    else:
        st.success(f"Test complete. Final score: {correct}/{st.session_state.num_questions}")


def render_question_bank(question_bank: QuestionBankStore) -> None:
    st.subheader("Question bank")
    filter_col, difficulty_col, refresh_col = st.columns([2, 1, 1])
    with filter_col:
        question_type = st.selectbox(
            "Question type",
            ["All", "short_answer", "single_choice", "multiple_choice"],
            format_func=lambda value: "All types" if value == "All" else value.replace("_", " ").title(),
        )
    with difficulty_col:
        difficulty_label = st.selectbox("Difficulty", ["All", "1", "2", "3", "4", "5"])
    with refresh_col:
        st.write("")
        refresh = st.button("Refresh", key="refresh_question_bank")
    if refresh:
        st.rerun()

    try:
        result = question_bank.list_questions(
            question_type=None if question_type == "All" else question_type,
            difficulty=None if difficulty_label == "All" else int(difficulty_label),
        )
    except Exception as exc:
        st.warning(f"Could not load the question bank: {exc}")
        return

    questions = result["questions"]
    st.caption(f"{result['total']} question(s) match the selected filters.")
    if not questions:
        st.info("No questions have been stored yet.")
        return

    table_rows = [
        {
            "ID": item["id"],
            "Question": item["question"],
            "Type": item["question_type"].replace("_", " ").title(),
            "Difficulty": item["difficulty"],
            "Sources": ", ".join(item["source_paths"] or []),
            "Created": item["created_at"],
        }
        for item in questions
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    st.subheader("Question details")
    for item in questions:
        title = f"#{item['id']} Ã‚Â· Difficulty {item['difficulty']} Ã‚Â· {item['question']}"
        with st.expander(title):
            st.markdown(f"**Type:** {item['question_type'].replace('_', ' ').title()}")
            if item["options"]:
                st.markdown("**Options**")
                for option in item["options"]:
                    st.write(f"- {option}")
            st.markdown("**Correct answer**")
            st.write(", ".join(item["correct_answer"]))
            st.markdown("**Explanation**")
            st.write(item["explanation"])
            st.markdown("**Sources**")
            st.write(", ".join(item["source_paths"] or []))


initialize_state()

st.title("Adaptive Testing Assistant")
st.write("Answer grounded questions from your knowledge base. The difficulty adapts after every answer.")

try:
    retriever, client, performance, startup_processed = get_runtime()
except Exception as exc:
    st.error(f"Application initialization failed: {exc}")
    st.stop()

with st.sidebar:
    st.header("Test setup")
    topic_input = st.text_input(
        "Topics (optional)",
        placeholder="Python, PostgreSQL, Docker",
        help="Enter one or more topics separated by commas.",
    )
    question_count = st.number_input("Number of questions", min_value=1, value=20, step=1)

    if st.button("Refresh index"):
        try:
            processed = retriever.refresh_index()
            st.success(f"Refreshed {len(processed)} file(s).")
        except Exception as exc:
            st.error(f"Index refresh failed: {exc}")

    if st.button("Start test", type="primary"):
        try:
            reset_test()
            st.session_state.test_active = True
            selected_topics = parse_topic_input(topic_input)
            st.session_state.selected_topics = selected_topics or [client.select_random_topic()]
            st.session_state.num_questions = int(question_count)
            candidate_chunks = client.retrieve(st.session_state.selected_topics)
            st.session_state.scheduled_chunks = client._schedule_chunks(
                candidate_chunks, st.session_state.num_questions
            )
            if not st.session_state.scheduled_chunks:
                raise RuntimeError("The retriever returned no document chunks")
            st.session_state.question_number = 1
            prepare_question(client)
            st.session_state.test_id = performance.start_test(
                st.session_state.anonymous_session_id,
                st.session_state.selected_topics,
                st.session_state.num_questions,
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Could not start the test: {exc}")

    if st.session_state.test_active and st.button("End test"):
        end_test(client, performance)
        st.rerun()

    if st.button("Restart"):
        reset_test()
        st.rerun()

if startup_processed:
    st.info(f"Indexed {len(startup_processed)} updated file(s) at startup.")

test_tab, question_bank_tab = st.tabs(["Adaptive Test", "Question Bank"])

with question_bank_tab:
    render_question_bank(client.question_bank)

with test_tab:
    if not st.session_state.test_active:
        if st.session_state.results:
            render_summary()
        else:
            st.info("Choose one or more topics and a question count, then select Start test.")
    else:
        question = st.session_state.current_question
        if question is None:
            st.error("No question is available. Restart the test.")
        else:
            st.progress(
                st.session_state.question_number / st.session_state.num_questions,
                text=f"Question {st.session_state.question_number} of {st.session_state.num_questions}",
            )
            topic_label = "; ".join(st.session_state.selected_topics)
            st.caption(f"Topics: {topic_label} - Difficulty: {st.session_state.difficulty}/5")

            with st.chat_message("assistant"):
                st.write(question["question"])
                if question["options"]:
                    for index, option in enumerate(question["options"], 1):
                        st.write(f"{index}. {option}")

            if st.session_state.answer_submitted:
                render_submitted_feedback()
                if st.button(
                    "Next question",
                    type="primary",
                    key=f"next_{st.session_state.question_number}",
                ):
                    was_last_question = (
                        st.session_state.question_number >= st.session_state.num_questions
                    )
                    clear_current_question()
                    if was_last_question:
                        persist_test_questions(client)
                        if st.session_state.test_id:
                            performance.finish_test(st.session_state.test_id, completed=True, ended_early=False)
                        st.session_state.test_active = False
                    else:
                        st.session_state.question_number += 1
                        prepare_question(client)
                    st.rerun()
            else:
                with st.form(f"answer_form_{st.session_state.question_number}"):
                    answer = answer_widget(question)
                    submitted = st.form_submit_button("Submit answer")

                if submitted:
                    if not answer.strip():
                        st.warning("Please provide an answer before submitting.")
                    else:
                        try:
                            assigned_difficulty = st.session_state.difficulty
                            grade = client.grade_answer(
                                question,
                                answer,
                                st.session_state.current_chunks,
                            )
                            is_correct = grade["is_correct"]
                            next_difficulty = max(1, min(5, assigned_difficulty + (1 if is_correct else -1)))
                            if st.session_state.test_id:
                                performance.record_answer(
                                    st.session_state.test_id,
                                    st.session_state.question_number,
                                    question["question_type"],
                                    assigned_difficulty,
                                    next_difficulty,
                                    is_correct,
                                )
                            st.session_state.current_answer = answer
                            st.session_state.current_grade = grade
                            st.session_state.answer_submitted = True
                            st.session_state.results.append({
                                "question": question,
                                "answer": answer,
                                "grade": grade,
                                "difficulty": st.session_state.difficulty,
                                "generation_difficulty": st.session_state.difficulty,
                                "source_paths": sorted({
                                    chunk.get("source_path")
                                    for chunk in st.session_state.current_chunks
                                    if chunk.get("source_path")
                                }),
                            })
                            st.session_state.history.append({
                                "question": question["question"],
                                "answer": answer,
                                "correct": is_correct,
                            })
                            st.session_state.difficulty = next_difficulty
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not grade the answer: {exc}")

