import os

import psycopg
from dotenv import load_dotenv


def get_connection():
    """Create a PostgreSQL connection from DATABASE_URL or component settings."""
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return psycopg.connect(database_url)

    return psycopg.connect(
        dbname=os.getenv("POSTGRES_DB", "adaptive_testing"),
        user=os.getenv("POSTGRES_USER", "adaptive_testing"),
        password=os.getenv("POSTGRES_PASSWORD", "adaptive_testing"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
    )



def ensure_usage_schema(connection, commit: bool = True) -> None:
    """Create the OpenAI usage table and indexes if needed."""
    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS openai_usage (
                id BIGSERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                operation TEXT NOT NULL,
                model TEXT NOT NULL,
                response_id TEXT,
                input_tokens BIGINT NOT NULL,
                cached_input_tokens BIGINT NOT NULL DEFAULT 0,
                output_tokens BIGINT NOT NULL,
                total_tokens BIGINT NOT NULL,
                input_cost_usd NUMERIC(18, 10) NOT NULL DEFAULT 0,
                cached_input_cost_usd NUMERIC(18, 10) NOT NULL DEFAULT 0,
                output_cost_usd NUMERIC(18, 10) NOT NULL DEFAULT 0,
                total_cost_usd NUMERIC(18, 10) NOT NULL DEFAULT 0
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS openai_usage_created_at_idx ON openai_usage (created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS openai_usage_operation_idx ON openai_usage (operation)")
        cursor.execute("CREATE INDEX IF NOT EXISTS openai_usage_model_idx ON openai_usage (model)")
    if commit:
        connection.commit()

def ensure_question_bank_schema(connection) -> None:
    """Create the generated-question table and indexes if needed."""
    dimensions = int(os.getenv("VECTOR_DIMENSIONS", "384"))
    if dimensions <= 0:
        raise ValueError("VECTOR_DIMENSIONS must be positive")
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS question_bank (
                id BIGSERIAL PRIMARY KEY,
                question_type TEXT NOT NULL,
                question_text TEXT NOT NULL,
                options JSONB NOT NULL DEFAULT '[]'::jsonb,
                correct_answer JSONB NOT NULL,
                explanation TEXT NOT NULL,
                source_paths TEXT[] NOT NULL DEFAULT '{{}}',
                difficulty SMALLINT NOT NULL CHECK (difficulty BETWEEN 1 AND 5),
                embedding vector({dimensions}) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS question_bank_embedding_hnsw_idx "
            "ON question_bank USING hnsw (embedding vector_cosine_ops)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS question_bank_created_at_idx "
            "ON question_bank (created_at)"
        )
    connection.commit()



def ensure_performance_schema(connection, commit: bool = True) -> None:
    """Create anonymous test-session and answer-performance tables."""
    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_sessions (
                id UUID PRIMARY KEY,
                anonymous_session_id UUID NOT NULL,
                started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                completed_at TIMESTAMPTZ,
                topics TEXT[] NOT NULL DEFAULT '{}',
                requested_questions INTEGER NOT NULL CHECK (requested_questions > 0),
                answered_questions INTEGER NOT NULL DEFAULT 0 CHECK (answered_questions >= 0),
                correct_answers INTEGER NOT NULL DEFAULT 0 CHECK (correct_answers >= 0),
                completed BOOLEAN NOT NULL DEFAULT FALSE,
                ended_early BOOLEAN NOT NULL DEFAULT FALSE
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS test_sessions_anonymous_idx ON test_sessions (anonymous_session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS test_sessions_started_at_idx ON test_sessions (started_at)")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_answers (
                id BIGSERIAL PRIMARY KEY,
                session_id UUID NOT NULL REFERENCES test_sessions(id) ON DELETE CASCADE,
                question_number INTEGER NOT NULL CHECK (question_number > 0),
                question_type TEXT NOT NULL,
                assigned_difficulty SMALLINT NOT NULL CHECK (assigned_difficulty BETWEEN 1 AND 5),
                next_difficulty SMALLINT NOT NULL CHECK (next_difficulty BETWEEN 1 AND 5),
                is_correct BOOLEAN NOT NULL,
                answered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (session_id, question_number)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS test_answers_session_idx ON test_answers (session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS test_answers_answered_at_idx ON test_answers (answered_at)")
    if commit:
        connection.commit()

def initialize_database(connection=None) -> None:
    """Create the complete application schema in one idempotent transaction."""
    dimensions = int(os.getenv("VECTOR_DIMENSIONS", "384"))
    if dimensions <= 0:
        raise ValueError("VECTOR_DIMENSIONS must be positive")
    owns_connection = connection is None
    connection = connection or get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id BIGSERIAL PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({dimensions}),
                    source_last_updated_date TIMESTAMPTZ NOT NULL,
                    last_processed_date TIMESTAMPTZ NOT NULL,
                    UNIQUE (source_path, chunk_index)
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS documents_source_path_idx ON documents (source_path)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx "
                "ON documents USING hnsw (embedding vector_cosine_ops)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS documents_content_fts_idx "
                "ON documents USING gin (to_tsvector('english', content))"
            )

            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS question_bank (
                    id BIGSERIAL PRIMARY KEY,
                    question_type TEXT NOT NULL,
                    question_text TEXT NOT NULL,
                    options JSONB NOT NULL DEFAULT '[]'::jsonb,
                    correct_answer JSONB NOT NULL,
                    explanation TEXT NOT NULL,
                    source_paths TEXT[] NOT NULL DEFAULT '{{}}',
                    difficulty SMALLINT NOT NULL CHECK (difficulty BETWEEN 1 AND 5),
                    embedding vector({dimensions}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS question_bank_embedding_hnsw_idx "
                "ON question_bank USING hnsw (embedding vector_cosine_ops)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS question_bank_created_at_idx ON question_bank (created_at)")

        ensure_usage_schema(connection, commit=False)
        ensure_performance_schema(connection, commit=False)
        ensure_question_quality_schema(connection, commit=False)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        if owns_connection:
            connection.close()


def ensure_question_quality_schema(connection, commit: bool = True) -> None:
    """Create the generated-question quality audit table and indexes."""
    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS question_quality_judgments (
                id BIGSERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                question_text TEXT NOT NULL,
                question_type TEXT NOT NULL,
                options JSONB NOT NULL DEFAULT '[]'::jsonb,
                expected_answer JSONB NOT NULL,
                source_paths TEXT[] NOT NULL DEFAULT '{}',
                assigned_difficulty SMALLINT NOT NULL CHECK (assigned_difficulty BETWEEN 1 AND 5),
                groundedness_score SMALLINT CHECK (groundedness_score BETWEEN 1 AND 5),
                answer_correctness_score SMALLINT CHECK (answer_correctness_score BETWEEN 1 AND 5),
                clarity_score SMALLINT CHECK (clarity_score BETWEEN 1 AND 5),
                difficulty_score SMALLINT CHECK (difficulty_score BETWEEN 1 AND 5),
                overall_score DOUBLE PRECISION,
                passed BOOLEAN NOT NULL DEFAULT FALSE,
                judge_status TEXT NOT NULL,
                judge_model TEXT NOT NULL,
                attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
                groundedness_explanation TEXT NOT NULL DEFAULT '',
                answer_correctness_explanation TEXT NOT NULL DEFAULT '',
                clarity_explanation TEXT NOT NULL DEFAULT '',
                difficulty_explanation TEXT NOT NULL DEFAULT ''
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS question_quality_created_at_idx ON question_quality_judgments (created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS question_quality_passed_idx ON question_quality_judgments (passed)")
        cursor.execute("CREATE INDEX IF NOT EXISTS question_quality_model_idx ON question_quality_judgments (judge_model)")
        cursor.execute("CREATE INDEX IF NOT EXISTS question_quality_difficulty_idx ON question_quality_judgments (assigned_difficulty)")
    if commit:
        connection.commit()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Initialize the PostgreSQL/pgvector application schema")
    parser.add_argument("--init", action="store_true", help="create all application tables and indexes")
    args = parser.parse_args()
    if not args.init:
        parser.error("use --init to initialize the database schema")
    initialize_database()




