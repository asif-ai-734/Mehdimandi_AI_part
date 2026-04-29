# Document RAG System - FastAPI Backend

A simple user/project-scoped Retrieval-Augmented Generation backend. There is no auth and no project management endpoint. The client sends string `user_id` and `project_id` scope values directly when uploading files, chatting, or running analysis.

## Features

- Upload multiple PDF, DOCX, or TXT files in one request
- Extract text from PDF, DOCX, and TXT files, then chunk, embed, and store document chunks in Qdrant
- Hybrid retrieval combines semantic vectors, BM25 keyword scoring, and exact ID boosts for references like `AR-402`, `3.23`, `W126`, and `D101`
- Retrieve knowledge by `user_id` and `project_id`
- Generate answers with OpenAI using retrieved document context and citations
- Save chat history to SQLite in `chat_history`

## App Structure

```text
app/
  main.py
  config.py
  database.py
  models.py              # Document, ChatHistory
  schemas/               # Pydantic schemas
  routes/
    documents.py         # upload/list/delete documents
    chat.py              # chat and chat history
  services/
    chunker.py
    embeddings.py
    file_extractor.py
    openai_service.py
    qdrant_service.py
    rag_service.py
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then start Qdrant:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

Run the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger docs:

```text
http://localhost:8000/docs
```

## API

### Upload Documents

```http
POST /documents/upload
Content-Type: multipart/form-data
```

Multipart fields:

```text
user_id=user-1
project_id=project-10
project_name=Cedar Ridge Exterior
project_address=225 Confederation Drive, Toronto
files=@document1.pdf
files=@document2.docx
files=@notes.txt
```

The upload endpoint processes documents synchronously. Files are text-extracted, split into chunks, embedded, and stored in Qdrant for that `user_id` and `project_id`. `project_name` and `project_address` are stored with the document and vector metadata so later chat and analysis can use the same project context.

### List Documents

```http
GET /documents?user_id=user-1&project_id=project-10
```

### Delete Document

```http
DELETE /documents/{document_id}?user_id=user-1&project_id=project-10
```

### Chat

```http
POST /chat
Content-Type: application/json

{
  "user_id": "user-1",
  "project_id": "project-10",
  "message": "What are the main topics in these documents?"
}
```

### Chat History

```http
GET /chat/history?user_id=user-1&project_id=project-10&limit=50&offset=0
DELETE /chat/history?user_id=user-1&project_id=project-10
```

### Tender Analysis

```http
POST /analysis/tender
Content-Type: application/json

{
  "user_id": "user-1",
  "project_id": "project-10",
  "divisions": ["06", "08", "09"],
  "instructions": "Focus on material costs for windows and doors. Exclude painting work."
}
```

The analysis endpoint uses the existing RAG index for the uploaded project documents and returns a structured preview with metrics, scope, risks, pricing impacts, selected divisions, and sources.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///./rag.db` | SQLite database path |
| `OPENAI_API_KEY` | empty | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Text model for chat responses |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | empty | Optional Qdrant API key |
| `QDRANT_COLLECTION_NAME` | `project_documents` | Qdrant collection |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-base-en-v1.5` | Embedding model |
| `HYBRID_CANDIDATE_MULTIPLIER` | `4` | Vector candidate expansion before reranking |
| `KEYWORD_SCAN_LIMIT` | `500` | Scoped payload scan limit for exact ID retrieval |

## Tests

```bash
pytest tests/ -v
```

## Notes

`project_id` is just a string scope value. The app does not create, update, list, or delete projects.
