import argparse
import json

from database import get_connection
from embedding import FastEmbedder
from openai_rag_client import OpenAIRAGClient
from rag_retriever import RAGRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the knowledge base with OpenAI tool calling")
    parser.add_argument("query", nargs="?", help="Question or search query")
    args = parser.parse_args()
    query = args.query or input("Query: ").strip()

    connection = get_connection()
    try:
        retriever = RAGRetriever("data", connection, FastEmbedder())
        chunks = OpenAIRAGClient(retriever).retrieve(query)
        print(json.dumps(chunks, indent=2, ensure_ascii=False))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
