# Adaptive Testing Assistant

Adaptive Testing Assistant turns a Markdown knowledge base into grounded adaptive tests. It retrieves relevant document chunks, generates and grades structured questions with OpenAI, adapts difficulty after each answer, persists approved questions, records usage and anonymous performance metrics, and provides Grafana dashboards.

## What it does

- Indexes Markdown files recursively with local FastEmbed embeddings.
- Retrieves context with vector or hybrid PostgreSQL/pgvector search.
- Generates short-answer, single-choice, and multiple-choice questions.
- Uses a separate quality judge before showing a generated question.
- Prevents normalized duplicate questions within the current test.
- Adapts difficulty from 1 to 5 based on the learner answer.
- Stores approved questions with semantic duplicate filtering.
- Records OpenAI token usage, estimated cost, and anonymous test performance.
- Provides Streamlit UI, CLI, Docker Compose, and Grafana dashboards.

## Architecture

    Markdown files
          |
          v
    RAGRetriever ----> FastEmbed
          |
          v
    PostgreSQL + pgvector <---- UsageRecorder
          ^                         ^
          |                         |
    QuestionBankStore        PerformanceRecorder
          ^                         ^
          |                         |
    Streamlit / CLI ------> OpenAI Responses API
                                  |
                         question generation
                         quality judging
                         answer grading

## Quick start

Prerequisites:

- Docker Desktop with Docker Compose
- An OpenAI API key
- At least one Markdown file under data/

1. Copy the environment template:

       Copy-Item .env.example .env

2. Set the API key in .env:

       OPENAI_API_KEY=your_openai_api_key

3. Add or edit Markdown files under data/.

4. Start the stack:

       docker compose up --build

Open:

- Streamlit: http://localhost:8501
- Grafana: http://localhost:3000
- PostgreSQL: localhost:5432

The first startup may take longer while the local embedding model downloads.

The schema-init service is one-shot. After schema changes or when using an existing database volume, rerun it explicitly:

    docker compose build schema-init streamlit
    docker compose run --rm schema-init
    docker compose up -d grafana streamlit

To stop services while preserving volumes:

    docker compose down

## Using the application

In Streamlit:

1. Enter comma-separated topics, or leave the topic field blank for a random topic.
2. Choose the number of questions.
3. Select Start test.
4. Submit answers and review the feedback.
5. Select Next question to continue or End test to stop early.
6. Use the Question Bank tab to browse persisted approved questions.

Retrieved chunks are scheduled once per test and distributed across questions. The generator receives the complete prior-question history, and normalized duplicate generations are rejected and retried.

The CLI uses the same retrieval and adaptive-test client:

    uv sync
    python main.py --topic "Python data structures" --questions 5
    python main.py --questions 10

The CLI requires a reachable PostgreSQL instance and an initialized schema.

## Quality judging

Generated questions are reviewed before display. The judge scores groundedness, expected-answer correctness, clarity, and difficulty alignment. Failed attempts are regenerated up to QUESTION_JUDGE_MAX_ATTEMPTS.

The judge model is configured independently with OPENAI_JUDGE_MODEL. If it is unset, it falls back to OPENAI_MODEL.

## Usage and cost tracking

UsageRecorder stores one record per OpenAI structured response in openai_usage. Costs are estimates, not invoices.

Global fallback rates are configured per million tokens:

- OPENAI_INPUT_COST_PER_MILLION_TOKENS
- OPENAI_CACHED_INPUT_COST_PER_MILLION_TOKENS
- OPENAI_OUTPUT_COST_PER_MILLION_TOKENS

For model-specific pricing, set OPENAI_MODEL_PRICING to a JSON object. Each model requires input, cached_input, and output rates:

    OPENAI_MODEL_PRICING={"gpt-5.4-nano":{"input":5,"cached_input":0.5,"output":30},"gpt-5.4-mini":{"input":5,"cached_input":0.5,"output":30}}

Rates are USD per one million tokens. The input rate applies to non-cached input tokens; cached_input applies to cached input tokens; output applies to output tokens. An unknown model uses the global fallback rates.

## Grafana dashboards

Grafana is provisioned with the Adaptive PostgreSQL datasource and two dashboards:

- OpenAI Usage and Cost: token totals, estimated cost, cost over time, and usage by operation/model.
- User Test Performance: tests started, overall accuracy, accuracy over time, accuracy by difficulty, and recent sessions.

If Grafana reports an authentication or empty-dashboard error:

    docker compose run --rm schema-init
    docker compose up -d --force-recreate grafana

The datasource uses the same POSTGRES_DB, POSTGRES_USER, and POSTGRES_PASSWORD values as the application. A new test must generate usage or performance rows before the corresponding panels contain non-zero data.

## Configuration

Copy .env.example to .env. Important variables:

| Variable | Default/example | Purpose |
| --- | --- | --- |
| OPENAI_API_KEY | empty | Required for generation and grading |
| OPENAI_MODEL | gpt-5.4-nano | Generation and grading model |
| OPENAI_JUDGE_MODEL | gpt-5.4-mini | Quality-judge model |
| OPENAI_MODEL_PRICING | empty | JSON model-to-rate overrides |
| QUESTION_JUDGE_MAX_ATTEMPTS | 3 | Quality-gate attempts |
| QUESTION_JUDGE_PASS_THRESHOLD | 4.0 | Minimum average judge score |
| DATABASE_URL | local PostgreSQL URL | Local Python connection |
| POSTGRES_DB | adaptive_testing | Database name |
| POSTGRES_USER | adaptive_testing | Database user |
| POSTGRES_PASSWORD | adaptive_testing | Database password |
| POSTGRES_HOST | localhost | Host for local Python execution |
| POSTGRES_PORT | 5432 | Host PostgreSQL port |
| VECTOR_DIMENSIONS | 384 | Embedding vector size |
| FASTEMBED_MODEL | BAAI/bge-small-en-v1.5 | Local embedding model |
| FASTEMBED_CACHE_PATH | .fastembed locally, /models in Docker | Model cache |
| DAYS_AGO | 3 | Recent-file refresh window |
| RAG_SEARCH_MODE | hybrid | hybrid or vector |
| RAG_TOP_K | 5 | Retrieved chunks per topic |
| GRAFANA_PORT | 3000 | Host Grafana port |
| GRAFANA_ADMIN_USER | admin | Grafana administrator |
| GRAFANA_ADMIN_PASSWORD | change-me | Grafana administrator password |

Docker services use pgvector as the PostgreSQL hostname. Local Python execution normally uses localhost.

## Testing

Run unit tests:

    uv sync
    uv run python -m unittest discover -s tests -v

The suite covers topic parsing, retrieval, structure-aware chunking, question generation, duplicate prevention, grading, question persistence, usage/cost recording, performance recording, and schema initialization.

Validate Compose:

    docker compose config --quiet

Optional pgvector integration tests require the Compose PostgreSQL service:

    docker compose up -d --build --wait pgvector schema-init
    $env:RUN_PGVECTOR_INTEGRATION="1"
    docker compose run --rm streamlit python -m unittest tests.test_pgvector_integration -v
    docker compose down

## Data and persistence

- data/ is mounted read-only into the application container.
- PostgreSQL stores document chunks, generated questions, usage records, quality judgments, test sessions, and test answers.
- Learner answer text is not stored in the performance tables.
- docker compose down preserves named volumes.
- docker compose down -v removes PostgreSQL, Grafana, and model-cache volumes.

## Project structure

    data/                 Markdown knowledge base
    database.py           PostgreSQL connection and schema initialization
    embedding.py          FastEmbed ONNX wrapper
    rag_retriever.py      Ingestion, chunking, indexing, and retrieval
    openai_rag_client.py  Structured generation, judging, grading, and tests
    question_bank.py      Question persistence and semantic duplicate filtering
    usage.py              Token and model-specific cost recording
    performance.py        Anonymous test-performance recording
    streamlit_app.py      Streamlit UI
    main.py               CLI entry point
    docker-compose.yml    PostgreSQL, Streamlit, and Grafana services
    grafana/              Provisioned datasource and dashboards
    tests/                Unit and optional integration tests

## Troubleshooting

### PostgreSQL connection failures

    docker compose ps
    docker compose logs pgvector
    docker compose logs streamlit

Ensure pgvector is healthy and that local Python uses localhost while Docker services use pgvector.

### Grafana has no data

Check datasource health and table existence:

    docker compose run --rm schema-init
    docker compose up -d --force-recreate grafana
    docker compose exec -T pgvector psql -U adaptive_testing -d adaptive_testing -c "SELECT COUNT(*) FROM openai_usage;"
    docker compose exec -T pgvector psql -U adaptive_testing -d adaptive_testing -c "SELECT COUNT(*) FROM test_sessions;"

### Question bank is empty

Complete or end a test after a question is generated, then refresh the Question Bank tab. Only approved questions are persisted.

### OpenAI authentication failures

Set OPENAI_API_KEY in .env and recreate Streamlit:

    docker compose up -d --build --force-recreate streamlit
    docker compose logs streamlit

### Vector dimension mismatch

VECTOR_DIMENSIONS must match the selected embedding model. Recreate local state only when a full reindex is acceptable:

    docker compose down -v
    docker compose up --build

## Limitations

- The default deployment is for local development, not hardened production use.
- There is no authentication or multi-user isolation.
- Generated content can still be incorrect despite retrieval and quality judging.
- Cost values depend on configured pricing and model metadata.
- Markdown is the canonical knowledge-base format.
- Topic names containing commas cannot be entered as a single topic.