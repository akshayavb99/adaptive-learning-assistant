FROM python:3.13-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir fastembed openai psycopg[binary] python-dotenv requests streamlit
COPY rag_retriever.py database.py embedding.py openai_rag_client.py performance.py usage.py question_bank.py main.py chat.py streamlit_app.py ./
COPY tests ./tests

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]




