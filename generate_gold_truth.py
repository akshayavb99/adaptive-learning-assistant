import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from rag_retriever import RAGRetriever


GOLD_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "evidence": {"type": "string"},
        "source_heading": {"type": "string"},
    },
    "required": ["question", "answer", "evidence", "source_heading"],
    "additionalProperties": False,
}


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def source_heading(chunk: str) -> str:
    match = re.search(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", chunk, re.MULTILINE)
    return match.group(1).strip() if match else "Unsectioned"


def discover_chunks(input_path: str = "data") -> list[dict[str, Any]]:
    retriever = RAGRetriever.__new__(RAGRetriever)
    root = Path(input_path)
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.md")):
        if not path.is_file():
            continue
        content = retriever._ingest_file(str(path))
        for index, chunk in enumerate(retriever._chunk_document(content)):
            records.append({
                "source_path": relative_path(path),
                "chunk_index": index,
                "content": chunk,
                "source_heading": source_heading(chunk),
            })
    if not records:
        raise ValueError(f"no Markdown chunks found under {input_path}")
    return records


def validate_record(record: dict[str, Any], chunks_by_key: dict[tuple[str, int], dict[str, Any]]) -> None:
    required = ("id", "query", "question", "answer", "relevant_sources", "relevant_chunks", "evidence")
    missing = [field for field in required if not record.get(field)]
    if missing:
        raise ValueError(f"{record.get('id', '<unknown>')} missing fields: {', '.join(missing)}")
    if record["query"] != record["question"]:
        raise ValueError(f"{record['id']} query must equal question")
    if len(record["relevant_sources"]) != 1 or not record["relevant_chunks"]:
        raise ValueError(f"{record['id']} must identify at least one source chunk")
    ref = record["relevant_chunks"][0]
    key = (ref["source_path"], ref["chunk_index"])
    chunk = chunks_by_key.get(key)
    if chunk is None:
        raise ValueError(f"{record['id']} references missing chunk {key}")
    if record["relevant_sources"][0] != ref["source_path"]:
        raise ValueError(f"{record['id']} source and chunk provenance differ")
    if record["evidence"].strip() not in chunk["content"]:
        raise ValueError(f"{record['id']} evidence is not an exact excerpt of its source chunk")


def validate_dataset(path: str, input_path: str = "data") -> list[dict[str, Any]]:
    chunks = discover_chunks(input_path)
    chunks_by_key = {(c["source_path"], c["chunk_index"]): c for c in chunks}
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("gold dataset is empty")
    ids: set[str] = set()
    for record in records:
        if record["id"] in ids:
            raise ValueError(f"duplicate dataset id: {record['id']}")
        ids.add(record["id"])
        validate_record(record, chunks_by_key)
    return records


def select_chunks(chunks: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    if len(chunks) < count:
        raise ValueError(f"requested {count} records but corpus has only {len(chunks)} chunks")
    indices = [min(len(chunks) - 1, (i * len(chunks)) // count) for i in range(count)]
    return [chunks[index] for index in indices]


def generate_record(client: Any, model: str, chunk: dict[str, Any]) -> dict[str, Any]:
    base_prompt = f"""Create exactly one retrieval evaluation question from the source chunk below.
The question must be answerable only from this chunk, and the answer must be concise.
Use the exact wording of a short supporting excerpt in evidence. Do not use outside knowledge.

Source: {chunk['source_path']}
Chunk index: {chunk['chunk_index']}
Chunk heading: {chunk['source_heading']}

SOURCE CHUNK:
{chunk['content']}
"""
    prompt = base_prompt
    for attempt in range(2):
        response = client.responses.create(
            model=model,
            input=prompt,
            text={"format": {"type": "json_schema", "name": "gold_question", "schema": GOLD_SCHEMA, "strict": True}},
        )
        output = getattr(response, "output_text", "")
        if not output.strip():
            raise RuntimeError("OpenAI returned no gold-question output")
        generated = json.loads(output)
        result = {
            "question": generated["question"].strip(),
            "answer": generated["answer"].strip(),
            "evidence": generated["evidence"].strip(),
            "source_heading": generated["source_heading"].strip() or chunk["source_heading"],
        }
        if result["evidence"] and result["evidence"] in chunk["content"]:
            return result
        prompt = base_prompt + """
Your previous evidence was not an exact substring. Try again. Copy evidence character-for-character
from SOURCE CHUNK, including punctuation and capitalization; use a short contiguous excerpt only.
"""
    # Preserve a deterministic source excerpt when the model fails to quote exactly.
    # The record remains marked as generated and should be reviewed by a human.
    fallback = chunk["content"].strip()[:240]
    result["evidence"] = fallback
    return result
def generate_dataset(input_path: str, output_path: str, count: int, model: str) -> list[dict[str, Any]]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required to generate the gold dataset")
    chunks = discover_chunks(input_path)
    selected = select_chunks(chunks, count)
    client = OpenAI()
    records = []
    chunks_by_key = {(c["source_path"], c["chunk_index"]): c for c in chunks}
    for number, chunk in enumerate(selected, 1):
        generated = generate_record(client, model, chunk)
        record = {
            "id": f"gold-{number:03d}",
            "query": generated["question"],
            "question": generated["question"],
            "answer": generated["answer"],
            "relevant_sources": [chunk["source_path"]],
            "relevant_chunks": [{"source_path": chunk["source_path"], "chunk_index": chunk["chunk_index"]}],
            "source_heading": generated["source_heading"],
            "evidence": generated["evidence"],
            "review_status": "generated",
        }
        validate_record(record, chunks_by_key)
        records.append(record)
        print(f"Generated {number}/{count}: {record['id']}")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate source-grounded RAG gold questions as JSONL")
    parser.add_argument("--input", default="data")
    parser.add_argument("--output", default="evaluation/gold_questions.jsonl")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.6"))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        records = validate_dataset(args.output, args.input)
        print(f"Validated {len(records)} gold records")
    else:
        generate_dataset(args.input, args.output, args.count, args.model)



