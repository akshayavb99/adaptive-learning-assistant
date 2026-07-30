import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

from database import ensure_retrieval_evaluation_schema, get_connection
from rag_retriever import RRF_K, RAGRetriever
from embedding import FastEmbedder

def normalize_source_path(path: str) -> str:
    """Normalize host/container/relative paths to a corpus-relative path."""
    text = str(path).replace("\\", "/")
    cwd = Path.cwd().as_posix().rstrip("/") + "/"
    for prefix in (cwd, "/app/"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    marker = "/data/"
    if marker in text:
        text = "data/" + text.rsplit(marker, 1)[1]
    return text.lstrip("./")



def load_cases(path: str) -> list[dict[str, Any]]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            case = json.loads(line)
            if not case.get("id") or not case.get("query"):
                raise ValueError("each evaluation case needs id and query")
            if not case.get("relevant_sources"):
                raise ValueError("each evaluation case needs relevant_sources")
            cases.append(case)
    if not cases:
        raise ValueError("evaluation dataset is empty")
    return cases


def _is_relevant(result: dict[str, Any], case: dict[str, Any]) -> bool:
    source_path = normalize_source_path(result.get("source_path", ""))
    relevant_sources = {
        normalize_source_path(source) for source in case["relevant_sources"]
    }
    if source_path not in relevant_sources:
        return False
    relevant_chunks = case.get("relevant_chunks")
    if not relevant_chunks:
        return True
    return any(
        normalize_source_path(chunk.get("source_path", "")) == source_path
        and chunk.get("chunk_index") == result.get("chunk_index")
        for chunk in relevant_chunks
    )

def calculate_metrics(
    cases: Iterable[dict[str, Any]],
    retrieved: dict[str, list[dict[str, Any]]],
    top_k: int,
) -> tuple[float, float, list[dict[str, Any]]]:
    total = 0
    hits = 0
    reciprocal_sum = 0.0
    result_rows = []
    for case in cases:
        total += 1
        first_rank = None
        results = retrieved[case["id"]][:top_k]
        for rank, result in enumerate(results, 1):
            relevant = _is_relevant(result, case)
            if relevant and first_rank is None:
                first_rank = rank
            result_rows.append({
                "query_id": case["id"],
                "result_rank": rank,
                "source_path": result.get("source_path", ""),
                "chunk_index": result.get("chunk_index"),
                "is_relevant": relevant,
                "reciprocal_rank": 0.0,
                "vector_rank": result.get("vector_rank"),
                "lexical_rank": result.get("lexical_rank"),
                "rrf_score": result.get("hybrid_score"),
                "vector_similarity": result.get("vector_similarity"),
                "lexical_score": result.get("lexical_score"),
            })
        if first_rank is not None:
            hits += 1
            reciprocal_sum += 1.0 / first_rank
            for row in result_rows:
                if row["query_id"] == case["id"] and row["result_rank"] == first_rank:
                    row["reciprocal_rank"] = 1.0 / first_rank
                    break
    if total == 0:
        raise ValueError("at least one evaluation case is required")
    return hits / total, reciprocal_sum / total, result_rows


def evaluate_retrieval(
    retriever: RAGRetriever,
    connection: Any,
    cases: list[dict[str, Any]],
    modes: tuple[str, ...] = ("vector", "hybrid"),
    top_k: int = 5,
    dataset_name: str = "retrieval_cases",
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if any(mode not in {"vector", "hybrid"} for mode in modes):
        raise ValueError("modes must contain only vector or hybrid")
    ensure_retrieval_evaluation_schema(connection)
    summaries = []
    try:
        for mode in modes:
            started = time.perf_counter()
            search = retriever.search_index if mode == "vector" else retriever.search_hybrid
            retrieved = {case["id"]: search(case["query"], top_k) for case in cases}
            hit_rate, mrr, result_rows = calculate_metrics(cases, retrieved, top_k)
            duration_ms = (time.perf_counter() - started) * 1000
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO retrieval_evaluation_runs (
                        dataset_name, search_mode, top_k, rrf_k, query_count,
                        hit_rate, mrr, duration_ms
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        dataset_name,
                        mode,
                        top_k,
                        RRF_K if mode == "hybrid" else None,
                        len(cases),
                        hit_rate,
                        mrr,
                        duration_ms,
                    ),
                )
                run_id = cursor.fetchone()[0]
                for row in result_rows:
                    cursor.execute(
                        """
                        INSERT INTO retrieval_evaluation_results (
                            run_id, query_id, result_rank, source_path, chunk_index,
                            is_relevant, reciprocal_rank, vector_rank, lexical_rank,
                            rrf_score, vector_similarity, lexical_score
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            run_id,
                            row["query_id"],
                            row["result_rank"],
                            row["source_path"],
                            row["chunk_index"],
                            row["is_relevant"],
                            row["reciprocal_rank"],
                            row["vector_rank"],
                            row["lexical_rank"],
                            row["rrf_score"],
                            row["vector_similarity"],
                            row["lexical_score"],
                        ),
                    )
            connection.commit()
            summaries.append({
                "run_id": run_id,
                "search_mode": mode,
                "hit_rate": hit_rate,
                "mrr": mrr,
                "query_count": len(cases),
                "duration_ms": duration_ms,
            })
    except Exception:
        connection.rollback()
        raise
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate vector and hybrid retrieval")
    parser.add_argument("--dataset", default="evaluation/retrieval_cases.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    cases = load_cases(args.dataset)
    connection = get_connection()
    try:
        retriever = RAGRetriever("data", connection, FastEmbedder())
        retriever.refresh_index()
        for summary in evaluate_retrieval(retriever, connection, cases, top_k=args.top_k):
            print(json.dumps(summary, indent=2))
    finally:
        connection.close()


if __name__ == "__main__":
    main()


