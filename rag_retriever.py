import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


DEFAULT_DAYS_AGO = 3
CHUNK_SIZE = 500
RRF_K = 60
HYBRID_CANDIDATE_MULTIPLIER = 4
HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+.+$")
FENCE_PATTERN = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
HORIZONTAL_RULE_PATTERN = re.compile(r"^\s{0,3}(?:(?:-\s*){3,}|(?:\*\s*){3,}|(?:_\s*){3,})$")
LOW_INFORMATION_FRAGMENTS = {"where", "and", "or", "because", "therefore"}


class RAGRetriever:
    """Discover, refresh, insert, and search source files indexed in pgvector."""

    def __init__(self, input_path: str, connection: Any, embedder: Any):
        self.input_path = Path(input_path)
        self.connection = connection
        self.embedder = embedder
        self.update_list: list[str] = []

    @staticmethod
    def _days_ago() -> int:
        value = os.getenv("DAYS_AGO")
        try:
            days = int(value) if value is not None else DEFAULT_DAYS_AGO
        except ValueError:
            return DEFAULT_DAYS_AGO
        return days if days >= 0 else DEFAULT_DAYS_AGO

    @staticmethod
    def _vector_to_string(vector: Any) -> str:
        values = vector.tolist() if hasattr(vector, "tolist") else vector
        return "[" + ",".join(str(value) for value in values) + "]"

    def _discover_markdown_files(self) -> list[str]:
        return sorted(
            str(file.resolve())
            for file in self.input_path.rglob("*.md")
            if file.is_file()
        )

    def list_topics(self) -> list[str]:
        """Return relative Markdown paths usable as random test topics."""
        return sorted(
            str(file.relative_to(self.input_path).with_suffix(""))
            for file in self.input_path.rglob("*.md")
            if file.is_file()
        )

    @staticmethod
    def _ingest_file(file_path: str) -> str:
        return Path(file_path).read_text(encoding="utf-8").replace("\r\n", "\n")

    @classmethod
    def _chunk_document(cls, content: str) -> list[str]:
        """Split Markdown into semantically coherent, structure-aware chunks.

        Headings provide context, while blank lines delimit paragraphs. Fenced
        code and Markdown lists are kept as atomic blocks. Character-based
        splitting is used only as a safety fallback for an unusually long
        prose paragraph.
        """
        lines = content.splitlines()
        heading_stack: list[tuple[int, str]] = []
        chunks: list[str] = []
        current: list[str] = []
        in_fence = False

        def heading_context() -> str:
            return "\n".join(line for _, line in heading_stack)

        def emit(block: list[str]) -> None:
            text = "\n".join(block).strip()
            if not text:
                return

            # Markdown separators are layout, not learning content. They can
            # otherwise become a chunk when they follow a heading.
            substantive_lines = [
                line for line in text.splitlines()
                if line.strip()
                and not HEADING_PATTERN.match(line)
                and not HORIZONTAL_RULE_PATTERN.match(line)
            ]
            substantive_text = " ".join(substantive_lines).strip()
            if not substantive_text or substantive_text.casefold() in LOW_INFORMATION_FRAGMENTS:
                return

            context = heading_context()
            prefix = f"{context}\n\n" if context else ""
            is_atomic = (
                bool(FENCE_PATTERN.match(text))
                or any(LIST_ITEM_PATTERN.match(line) for line in block)
            )
            pieces = [text] if is_atomic else cls._split_long_paragraph(text)
            chunks.extend(prefix + piece for piece in pieces)

        def flush() -> None:
            nonlocal current
            if current:
                emit(current)
                current = []

        for index, line in enumerate(lines):
            if not in_fence and HEADING_PATTERN.match(line):
                flush()
                marker = re.match(r"^\s{0,3}(#{1,6})\s+", line)
                if marker:
                    level = len(marker.group(1))
                    heading_stack = [item for item in heading_stack if item[0] < level]
                    heading_stack.append((level, line.strip()))
                continue

            if not in_fence and HORIZONTAL_RULE_PATTERN.match(line):
                flush()
                continue

            fence = FENCE_PATTERN.match(line)
            if fence:
                in_fence = not in_fence
                current.append(line)
                continue

            if not line.strip() and not in_fence:
                next_line = next((candidate for candidate in lines[index + 1:] if candidate.strip()), "")
                current_is_list = any(LIST_ITEM_PATTERN.match(item) for item in current)
                next_is_list = bool(LIST_ITEM_PATTERN.match(next_line))
                if not (current_is_list and next_is_list):
                    flush()
                    continue

            current.append(line)

        flush()
        return chunks

    @staticmethod
    def _split_long_paragraph(text: str) -> list[str]:
        if len(text) <= CHUNK_SIZE:
            return [text]

        sentences = re.split(r"(?<=[.!?])\s+", text)
        pieces: list[str] = []
        current: list[str] = []
        current_length = 0

        def flush() -> None:
            nonlocal current, current_length
            if current:
                pieces.append(" ".join(current).strip())
                current = []
                current_length = 0

        for sentence in sentences:
            if len(sentence) > CHUNK_SIZE:
                flush()
                words = sentence.split()
                word_chunk: list[str] = []
                word_length = 0
                for word in words:
                    extra = len(word) + (1 if word_chunk else 0)
                    if word_chunk and word_length + extra > CHUNK_SIZE:
                        pieces.append(" ".join(word_chunk))
                        word_chunk = []
                        word_length = 0
                    word_chunk.append(word)
                    word_length += len(word) + (1 if len(word_chunk) > 1 else 0)
                if word_chunk:
                    pieces.append(" ".join(word_chunk))
                continue

            extra = len(sentence) + (1 if current else 0)
            if current and current_length + extra > CHUNK_SIZE:
                flush()
            current.append(sentence)
            current_length += len(sentence) + (1 if len(current) > 1 else 0)

        flush()
        return pieces

    def _is_indexed(self, source_path: str) -> bool:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS(
                    SELECT 1
                    FROM documents
                    WHERE source_path = %s
                )
                """,
                (source_path,),
            )
            result = cursor.fetchone()

        return bool(result[0])

    def refresh_index(self) -> list[str]:
        """Return files that are recent or absent from the vector index."""
        load_dotenv()
        cutoff_time = time.time() - (self._days_ago() * 24 * 60 * 60)
        self.update_list = []

        for file in self.input_path.rglob("*.md"):
            if not file.is_file():
                continue

            source_path = str(file.resolve())
            is_recent = file.stat().st_mtime >= cutoff_time
            if is_recent or not self._is_indexed(source_path):
                self.update_list.append(source_path)

        if not self.update_list:
            return []
        return self.insert_into_index()

    def insert_into_index(self) -> list[str]:
        """Ingest, chunk, embed, and upsert the selected Markdown files."""
        load_dotenv()
        if not self.update_list:
            self.update_list = self._discover_markdown_files()

        processed: list[str] = []
        for file_path in self.update_list:
            source_path = str(Path(file_path).resolve())
            try:
                content = self._ingest_file(source_path)
                chunks = self._chunk_document(content)
                source_updated = datetime.fromtimestamp(
                    Path(source_path).stat().st_mtime, tz=timezone.utc
                )
                processed_at = datetime.now(timezone.utc)

                with self.connection.cursor() as cursor:
                    cursor.execute(
                        "DELETE FROM documents WHERE source_path = %s",
                        (source_path,),
                    )
                    for chunk_index, chunk in enumerate(chunks):
                        embedding = self._vector_to_string(self.embedder.encode(chunk))
                        cursor.execute(
                            """
                            INSERT INTO documents (
                                source_path,
                                chunk_index,
                                content,
                                embedding,
                                source_last_updated_date,
                                last_processed_date
                            )
                            VALUES (%s, %s, %s, %s::vector, %s, %s)
                            ON CONFLICT (source_path, chunk_index)
                            DO UPDATE SET
                                content = EXCLUDED.content,
                                embedding = EXCLUDED.embedding,
                                source_last_updated_date = EXCLUDED.source_last_updated_date,
                                last_processed_date = EXCLUDED.last_processed_date
                            """,
                            (
                                source_path,
                                chunk_index,
                                chunk,
                                embedding,
                                source_updated,
                                processed_at,
                            ),
                        )
                self.connection.commit()
                processed.append(source_path)
            except Exception:
                self.connection.rollback()
                raise

        return processed

    def search_index(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        """Return the closest indexed document chunks for ``query``."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(num_results, bool) or not isinstance(num_results, int) or num_results <= 0:
            raise ValueError("num_results must be a positive integer")

        query_vector = self._vector_to_string(self.embedder.encode(query))
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT source_path,
                       chunk_index,
                       content,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vector, query_vector, num_results),
            )
            rows = cursor.fetchall()

        return [
            {
                "source_path": row[0],
                "chunk_index": row[1],
                "content": row[2],
                "similarity": float(row[3]),
            }
            for row in rows
        ]

    def search_hybrid(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        """Search using vector similarity and PostgreSQL full-text relevance."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(num_results, bool) or not isinstance(num_results, int) or num_results <= 0:
            raise ValueError("num_results must be a positive integer")

        query_vector = self._vector_to_string(self.embedder.encode(query))
        candidate_limit = max(num_results * HYBRID_CANDIDATE_MULTIPLIER, 20)
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                WITH vector_scored AS (
                    SELECT source_path, chunk_index, content,
                           1 - (embedding <=> %s::vector) AS vector_similarity
                    FROM documents
                    WHERE embedding IS NOT NULL
                ),
                vector_results AS (
                    SELECT source_path, chunk_index, content, vector_similarity,
                           ROW_NUMBER() OVER (ORDER BY vector_similarity DESC) AS vector_rank
                    FROM vector_scored
                    ORDER BY vector_similarity DESC
                    LIMIT %s
                ),
                lexical_scored AS (
                    SELECT source_path, chunk_index, content,
                           ts_rank_cd(
                               to_tsvector('english', content),
                               websearch_to_tsquery('english', %s)
                           ) AS lexical_score
                    FROM documents
                    WHERE to_tsvector('english', content) @@
                          websearch_to_tsquery('english', %s)
                ),
                lexical_results AS (
                    SELECT source_path, chunk_index, content, lexical_score,
                           ROW_NUMBER() OVER (ORDER BY lexical_score DESC) AS lexical_rank
                    FROM lexical_scored
                    ORDER BY lexical_score DESC
                    LIMIT %s
                ),
                ranked AS (
                    SELECT source_path, chunk_index, content, vector_similarity,
                           vector_rank, NULL::double precision AS lexical_score,
                           NULL::bigint AS lexical_rank,
                           1.0 / (%s + vector_rank) AS rrf_score
                    FROM vector_results
                    UNION ALL
                    SELECT source_path, chunk_index, content,
                           NULL::double precision AS vector_similarity,
                           NULL::bigint AS vector_rank, lexical_score,
                           lexical_rank,
                           1.0 / (%s + lexical_rank) AS rrf_score
                    FROM lexical_results
                ),
                aggregated AS (
                    SELECT source_path, chunk_index, content,
                           MAX(vector_similarity) AS vector_similarity,
                           MAX(lexical_score) AS lexical_score,
                           MAX(vector_rank) AS vector_rank,
                           MAX(lexical_rank) AS lexical_rank,
                           SUM(rrf_score) AS hybrid_score
                    FROM ranked
                    GROUP BY source_path, chunk_index, content
                )
                SELECT source_path, chunk_index, content, vector_similarity,
                       lexical_score, vector_rank, lexical_rank, hybrid_score,
                       ROW_NUMBER() OVER (ORDER BY hybrid_score DESC) AS hybrid_rank
                FROM aggregated
                ORDER BY hybrid_score DESC
                LIMIT %s
                """,
                (
                    query_vector,
                    candidate_limit,
                    query,
                    query,
                    candidate_limit,
                    RRF_K,
                    RRF_K,
                    num_results,
                ),
            )
            rows = cursor.fetchall()

        return [
            {
                "source_path": row[0],
                "chunk_index": row[1],
                "content": row[2],
                "vector_similarity": float(row[3]) if row[3] is not None else None,
                "lexical_score": float(row[4]) if row[4] is not None else None,
                "vector_rank": int(row[5]) if row[5] is not None else None,
                "lexical_rank": int(row[6]) if row[6] is not None else None,
                "hybrid_score": float(row[7]),
                "hybrid_rank": int(row[8]),
            }
            for row in rows
        ]

    search = search_index











