# Document RAG System - FastAPI Backend

A simple user/project-scoped Retrieval-Augmented Generation backend. There is no auth and no project management endpoint. The client sends `user_id` and `project_id` directly when uploading files or chatting.

## Features

- Upload multiple PDF, DOCX, or TXT files in one request
- For PDFs, render pages, analyze sheets with OpenAI vision, extract sheet/keynote/schedule entities, then embed structured chunks
- For DOCX/TXT, extract, chunk, embed, and store document chunks in Qdrant
- Hybrid retrieval combines semantic vectors, BM25 keyword scoring, and exact ID boosts for references like `AR-402`, `3.23`, `W126`, and `D101`
- Retrieve knowledge by `user_id` and `project_id`
- Generate answers with OpenAI using retrieved document context and sheet/page citations
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
    drawing_processor.py
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
user_id=1
project_id=10
files=@document1.pdf
files=@document2.docx
files=@notes.txt
```

The upload endpoint processes documents synchronously. PDF drawing sets are split into page images, analyzed into sheet-aware JSON, synthesized at document level, chunked by semantic type, embedded, and stored in Qdrant for that `user_id` and `project_id`.

### List Documents

```http
GET /documents?user_id=1&project_id=10
```

### Delete Document

```http
DELETE /documents/{document_id}?user_id=1&project_id=10
```

### Chat

```http
POST /chat
Content-Type: application/json

{
  "user_id": 1,
  "project_id": 10,
  "message": "What are the main topics in these documents?"
}
```

### Chat History

```http
GET /chat/history?user_id=1&project_id=10&limit=50&offset=0
DELETE /chat/history?user_id=1&project_id=10
```

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///./rag.db` | SQLite database path |
| `OPENAI_API_KEY` | empty | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Text model for chat and document synthesis |
| `OPENAI_VISION_MODEL` | `gpt-4o` | Vision model for PDF sheet analysis |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | empty | Optional Qdrant API key |
| `QDRANT_COLLECTION_NAME` | `project_documents` | Qdrant collection |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-base-en-v1.5` | Embedding model |
| `ENABLE_PDF_DRAWING_ANALYSIS` | `true` | Enables OpenAI structured analysis for PDFs |
| `PDF_RENDER_DPI` | `200` | Starting DPI for rendered page images |
| `MAX_PAGE_IMAGE_BYTES` | `8388608` | Per-page image byte cap; DPI is reduced if needed |
| `HYBRID_CANDIDATE_MULTIPLIER` | `4` | Vector candidate expansion before reranking |
| `KEYWORD_SCAN_LIMIT` | `500` | Scoped payload scan limit for exact ID retrieval |

## Tests

```bash
pytest tests/ -v
```

## Notes

`project_id` is just a scope value. The app does not create, update, list, or delete projects.
