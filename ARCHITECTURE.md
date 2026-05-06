# Architecture

This backend is a FastAPI document RAG service for tender and estimating workflows. The system is intentionally scoped by caller-provided `user_id` and `project_id` values. There is no auth layer and no project table.

## Runtime Components

```text
Client
  -> FastAPI routes
  -> SQLite metadata and chat history
  -> Local uploaded files
  -> Qdrant vector database
  -> OpenAI chat/JSON generation
```

Startup initializes SQLite tables, applies small schema additions for existing SQLite databases, loads the embedding model, and creates the Qdrant collection if needed.

## API Layer

Routes are registered in `app/main.py`.

```text
documents tag
  POST   /documents/upload
  GET    /documents
  DELETE /documents/{document_id}

chat tag
  POST   /chat
  GET    /chat/history
  DELETE /chat/history

analysis tag
  GET /summary
  GET /scope
  GET /pricing
  GET /risks
  GET /clarifications
  GET /assumptions
  GET /exclusions
  GET /addenda
  GET /quote-draft
  GET /resources

health
  GET /
  GET /health
```

All analysis endpoints are grouped under one OpenAPI tag: `analysis`.

## Service Layer

```text
app/services/file_extractor.py
  Extracts text from PDF, DOCX, and TXT files.
  Counts PDF pages when available.

app/services/chunker.py
  Splits extracted text into overlapping chunks.

app/services/embeddings.py
  Loads the Sentence Transformer model and creates vectors.

app/services/qdrant_service.py
  Creates the Qdrant collection, stores vectors, searches scoped vectors,
  scrolls scoped payloads, and deletes vectors for removed documents.

app/services/rag_service.py
  Orchestrates document processing, chunk metadata, chat retrieval,
  reranking, OpenAI answer generation, and chat-history persistence.

app/services/openai_service.py
  Wraps OpenAI response generation and JSON generation.

app/services/analysis_service.py
  Builds the dashboard summary payload.

app/services/scope_service.py
app/services/pricing_service.py
app/services/risks_service.py
app/services/clarifications_service.py
app/services/assumptions_service.py
app/services/exclusions_service.py
app/services/addenda_service.py
app/services/quote_draft_service.py
  Build per-screen retrieval queries, call OpenAI for JSON, and normalize
  model output into stable response shapes.
```

## Data Model

Default database:

```env
DATABASE_URL=sqlite:///./rag.db
```

### `documents`

```text
id
filename
file_path
file_type
source_type        # document or addendum
project_id
user_id
project_name
project_address
divisions          # stored as JSON text
instructions
file_size
page_count
total_chunks
created_at
```

`ensure_database_schema()` adds missing columns for older local SQLite databases:

```text
project_name
project_address
source_type
divisions
instructions
file_size
page_count
```

### `chat_history`

```text
id
user_id
project_id
user_message
assistant_response
sources
created_at
```

## Qdrant Payload

Each stored chunk carries scope and source metadata:

```json
{
  "user_id": "user-1",
  "project_id": "project-10",
  "project_name": "Cedar Ridge Exterior",
  "project_address": "225 Confederation Drive, Toronto",
  "document_id": 25,
  "filename": "specifications.pdf",
  "source_type": "document",
  "chunk_index": 0,
  "chunk_type": "raw_text",
  "entities": [],
  "source_text_ref": "specifications.pdf",
  "text": "Chunk text..."
}
```

The retrieval layer filters by `user_id` and `project_id`, then combines semantic search, keyword scoring, and exact-reference boosts.

## Main Flows

### Upload

```text
POST /documents/upload
  user_id
  project_id
  project_name
  project_address
  divisions
  instructions
  files[]
  addendum[]
```

Flow:

```text
validate scope
-> validate file type
-> save file under uploads/{user_id}/{project_id}/{source_type}
-> save SQLite Document row with metadata
-> extract text
-> chunk text
-> embed chunks
-> upsert Qdrant vectors
-> update total_chunks
-> return per-file upload status
```

`file_size` is saved during upload. `page_count` is saved for PDFs when available.

### Chat

```text
POST /chat
```

Flow:

```text
normalize user_id/project_id
-> embed question
-> retrieve scoped vector matches
-> scan scoped payloads for keyword and exact-ID matches
-> rerank candidates
-> build source-labeled context
-> call OpenAI
-> extract cited sources
-> save chat_history row
-> return answer and sources
```

### Analysis

```text
GET /summary
GET /scope
GET /pricing
GET /risks
GET /clarifications
GET /assumptions
GET /exclusions
GET /addenda
GET /quote-draft
GET /resources
```

Analysis routes use only query params:

```text
user_id
project_id
```

The AI-backed analysis routes load saved `divisions` and `instructions` from uploaded `Document` rows for the same scope. They require at least one saved division value containing a numeric CSI code. This keeps the generated estimating outputs tied to the selected project scope.

General AI-backed route flow:

```text
validate user_id/project_id
-> load project documents
-> read saved divisions/instructions
-> validate saved divisions
-> build screen-specific retrieval queries
-> retrieve scoped Qdrant context
-> call OpenAI for JSON
-> normalize model payload
-> return Pydantic response model
```

`/resources` is the exception: it reads SQLite document metadata and returns the source document list. It does not call OpenAI or Qdrant.

## Response Responsibilities

```text
/summary
  Project dashboard numbers, highlights, and selected divisions.

/scope
  Scope items, quantities, specifications, and references.

/pricing
  Pricing impact items with descriptions, impacts, amounts, and references.

/risks
  Risks and coordination items with categories and references.

/clarifications
  Questions to send to the owner, architect, consultant, or GC.

/assumptions
  Supported bid assumptions.

/exclusions
  Explicit exclusions and by-others scope.

/addenda
  Addendum numbers, dates, affected divisions, scope changes, and pricing impact notes.

/quote-draft
  Suggested quote information for review.

/resources
  Uploaded source documents with filename, upload date, file size, page count, and type.
```

## Validation Rules

- `user_id` and `project_id` are required and normalized as non-empty strings.
- Upload accepts only `pdf`, `docx`, and `txt`.
- Upload instructions are limited to 5000 characters.
- AI-backed analysis routes require uploaded documents.
- AI-backed analysis routes require saved divisions.
- Saved divisions must include at least one numeric CSI-style code.
- Missing project documents return `404`.
- Invalid scope or invalid saved analysis inputs return `400`.

## Testing Strategy

The test suite uses FastAPI `TestClient`, a temporary SQLite database, and fake services for external dependencies. That verifies route registration, validation, upload behavior, database behavior, and response serialization without requiring live OpenAI or Qdrant calls.

Run:

```bash
.\venv\Scripts\python.exe -m pytest -v
```

Current expected result:

```text
10 passed
```
