import argparse

from database import get_connection
from embedding import FastEmbedder
from openai_rag_client import OpenAIRAGClient
from question_bank import QuestionBankStore
from rag_retriever import RAGRetriever
from usage import UsageRecorder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an adaptive knowledge-base test")
    parser.add_argument("--topic", help="Test topic; a random knowledge-base topic is used when omitted")
    parser.add_argument(
        "--questions",
        type=int,
        default=20,
        help="Number of questions to ask (default: 20)",
    )
    args = parser.parse_args()
    if args.questions <= 0:
        parser.error("--questions must be a positive integer")

    connection = get_connection()
    try:
        retriever = RAGRetriever("data", connection, FastEmbedder())
        processed = retriever.refresh_index()
        print(f"Refreshed {len(processed)} files")
        question_bank = QuestionBankStore(connection, retriever.embedder)
        client = OpenAIRAGClient(retriever, usage_recorder=UsageRecorder(connection), question_bank=question_bank)
        client.run_test(topic=args.topic, num_questions=args.questions)
    finally:
        connection.close()


if __name__ == "__main__":
    main()