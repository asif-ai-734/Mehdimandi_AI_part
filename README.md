# Document RAG API

FastAPI backend for a user/project-scoped document RAG workflow. Callers provide plain string `user_id` and `project_id` values on every request. The app does not include authentication or project-management endpoints.

Uploaded tender documents and addenda are extracted, chunked, embedded, stored in Qdrant, and linked to SQLite metadata. Chat and analysis routes then retrieve scoped project context and use OpenAI to return answers or structured estimating data.

## Features

- Upload multiple PDF, DOCX, and TXT files in one request
- Upload addenda separately through the `addendum` multipart field
- Store document metadata in SQLite, including `source_type`, `divisions`, `instructions`, `file_size`, `page_count`, and chunk count
- Store document chunks and project metadata in Qdrant
- Retrieve by `user_id` and `project_id`
- Hybrid retrieval with semantic search, keyword scoring, and exact reference boosts
- Chat against uploaded project documents with saved history
- Generate analysis screens for summary, scope, pricing, risks, clarifications, assumptions, exclusions, addenda, quote draft, and resources
- Group all analysis endpoints under one OpenAPI tag: `analysis`

## Project Structure

```text
app/
  main.py                 # FastAPI app, middleware, startup, router registration
  config.py               # Environment-driven settings
  database.py             # SQLAlchemy session and schema patching
  models.py               # Document, ChatHistory
  routes/
    documents.py          # Upload, list, delete documents
    chat.py               # Chat and chat history
    summary.py            # Project dashboard summary
    scope.py              # Scope of work items
    pricing.py            # Pricing impacts
    risks.py              # Risk and coordination items
    clarifications.py     # Clarification questions
    assumptions.py        # Bid assumptions
    exclusions.py         # Explicit exclusions
    addenda.py            # Addenda changes
    quote_draft.py        # Suggested quote information
    resources.py          # Source document list
  schemas/                # Pydantic request and response models
  services/
    file_extractor.py     # PDF, DOCX, TXT extraction and PDF page count
    chunker.py            # Text chunking
    embeddings.py         # Sentence Transformer embeddings
    qdrant_service.py     # Vector storage and scoped retrieval helpers
    openai_service.py     # OpenAI JSON and chat generation
    rag_service.py        # Upload processing, retrieval, chat orchestration
    analysis_service.py   # Summary analysis
    *_service.py          # Per-screen analysis services
  utils/
    analysis_inputs.py    # Division normalization and validation
    scopes.py             # user_id/project_id normalization
```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy the environment file and set your OpenAI key:

```bash
cp .env.example .env
```

Start Qdrant:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

Run the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

## Core API

### Health

```http
GET /
GET /health
```

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
divisions=["06", "08", "09"]
instructions=Focus on material costs for windows and doors.
files=@specifications.pdf
files=@drawings.docx
files=@notes.txt
addendum=@addendum-01.pdf
addendum=@addendum-02.docx
```

The upload route processes files synchronously. When the response returns, successful files have already been saved, extracted, chunked, embedded, and inserted into Qdrant.

Saved document metadata includes:

```text
filename
file_path
file_type
source_type        # document or addendum
user_id
project_id
project_name
project_address
divisions          # JSON string in SQLite, normalized to list in responses
instructions
file_size
page_count         # PDF only when available
total_chunks
created_at
```

### List Documents

```http
GET /documents?user_id=user-1&project_id=project-10
```

### Delete Document

```http
DELETE /documents/{document_id}?user_id=user-1&project_id=project-10
```

Deletes the SQLite row, saved file if present, and matching Qdrant vectors.

### Chat

```http
POST /chat
Content-Type: application/json

{
  "user_id": "user-1",
  "project_id": "project-10",
  "message": "What are the main pricing risks?"
}
```

### Chat History

```http
GET /chat/history?user_id=user-1&project_id=project-10&limit=50&offset=0
DELETE /chat/history?user_id=user-1&project_id=project-10
```

## Analysis API

All routes in this group are listed under the `analysis` tag in Swagger.

Each analysis route accepts only:

```text
user_id
project_id
```

Most analysis routes require at least one uploaded document and at least one saved division value containing a CSI-style numeric code. `divisions` and `instructions` are read from the latest uploaded document metadata for the same user/project scope.

| Route | Purpose |
| --- | --- |
| `GET /summary` | Dashboard summary with value, duration, labor, highlights, and selected divisions |
| `GET /scope` | Structured scope of work items with quantities and references |
| `GET /pricing` | Pricing impacts and cost factors |
| `GET /risks` | Risk and coordination items |
| `GET /clarifications` | Questions to clarify before pricing |
| `GET /assumptions` | Bid assumptions supported by tender context |
| `GET /exclusions` | Explicit exclusions and by-others items |
| `GET /addenda` | Addenda-driven changes and impacts |
| `GET /quote-draft` | Suggested quote data for review |
| `GET /resources` | Uploaded source document list |

Example:

```http
GET /scope?user_id=user-1&project_id=project-10
```

`/resources` returns the uploaded document list and can return an empty list for a valid scope. The AI-backed analysis routes return `404` when no uploaded documents exist for the scope and `400` when saved divisions are missing or invalid.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///./rag.db` | SQLAlchemy database URL |
| `OPENAI_API_KEY` | empty | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model for chat and JSON generation |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | empty | Optional Qdrant API key |
| `QDRANT_COLLECTION_NAME` | `project_documents` | Qdrant collection name |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-base-en-v1.5` | Sentence Transformer embedding model |
| `HYBRID_CANDIDATE_MULTIPLIER` | `4` | Vector candidate expansion before reranking |
| `KEYWORD_SCAN_LIMIT` | `500` | Scoped payload scan limit for keyword and exact-ID retrieval |

Other upload settings are defined in `app/config.py`:

```text
max_file_size = 10 MB
allowed_file_types = pdf, docx, txt
upload_dir = uploads
```

## Tests

Run the test suite:

```bash
.\venv\Scripts\python.exe -m pytest -v
```

Compile-check the app:

```bash
python -m compileall -q app
```

Current local result after the latest route/database updates:

```text
10 passed
```

## Notes

- `project_id` is a scope string, not a real project record.
- Upload processing is synchronous.
- The API stores files on local disk under `uploads/{user_id}/{project_id}/{source_type}`.
- Qdrant and OpenAI must be reachable for live chat and live analysis.
- Tests use fake services for external dependencies where needed.
