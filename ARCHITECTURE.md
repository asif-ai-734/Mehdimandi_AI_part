# Document RAG Architecture

This backend is a simple FastAPI document RAG service. There is no auth layer and no project management layer. Callers provide string `user_id` and `project_id` values directly, and those two values scope uploaded documents, Qdrant retrieval, and SQLite chat history.

## Layers

1. **API routes**
   - `app/routes/documents.py`: upload, list, and delete documents
   - `app/routes/chat.py`: chat, chat history, and history clearing
   - `app/routes/summary.py`: GET-only AI analysis summary for the results screen

2. **Services**
   - `app/services/file_extractor.py`: extracts text from PDF, DOCX, and TXT files
   - `app/services/chunker.py`: splits text into overlapping chunks
   - `app/services/analysis_service.py`: retrieves project context and asks OpenAI for structured tender analysis JSON
   - `app/services/embeddings.py`: creates embeddings with Sentence Transformers
   - `app/services/qdrant_service.py`: stores vectors in Qdrant and supports scoped payload scans for exact-ID retrieval
   - `app/services/openai_service.py`: generates final assistant responses
   - `app/services/rag_service.py`: orchestrates document processing, retrieval, response generation, and SQLite chat-history persistence

3. **Data**
   - `app/database.py`: SQLAlchemy engine/session setup
   - `app/models.py`: `Document` and `ChatHistory`
   - `app/schemas/`: Pydantic request and response schemas

## Database

Default database:

```env
DATABASE_URL=sqlite:///./rag.db
```

Tables:

```text
documents
  id
  filename
  file_path
  file_type
  source_type
  user_id
  project_id
  project_name
  project_address
  divisions
  instructions
  total_chunks
  created_at

chat_history
  id
  user_id
  project_id
  user_message
  assistant_response
  sources
  created_at
```

## Qdrant Payload

Each stored chunk includes the same scope fields used by SQLite:

```json
{
  "user_id": "user-1",
  "project_id": "project-10",
  "project_name": "Cedar Ridge Exterior",
  "project_address": "225 Confederation Drive, Toronto",
  "document_id": 25,
  "filename": "document.pdf",
  "source_type": "document",
  "chunk_index": 0,
  "chunk_type": "raw_text",
  "entities": [],
  "source_text_ref": "document.pdf",
  "text": "Chunk text..."
}
```

## Main Flows

### Document Upload

```text
POST /documents/upload
multipart fields:
  user_id=user-1
  project_id=project-10
  project_name=Cedar Ridge Exterior
  project_address=225 Confederation Drive, Toronto
  divisions=["06", "08"]
  instructions=Focus on openings and wood scope.
  files=@one.pdf
  files=@two.txt
  addendum=@addendum-01.pdf
  addendum=@addendum-02.docx
```

The API saves each file under `uploads/{user_id}/{project_id}/{source_type}`. Project name/address, saved divisions/instructions, and `source_type` (`document` or `addendum`) are stored on each document. Project name/address and `source_type` are also embedded into Qdrant payload metadata. PDFs, DOCX files, and TXT files follow the same text extraction and chunking path:

```text
PDF/DOCX/TXT
-> text extraction
-> overlapping raw-text chunks
-> embeddings in Qdrant
```

Processing is synchronous, so the uploaded documents are ready for chat when the response returns.

### Chat

```text
POST /chat
body: {
  "user_id": "user-1",
  "project_id": "project-10",
  "message": "What do these files say?"
}
```

The API embeds the question, retrieves scoped vector candidates from Qdrant, scans scoped payloads for BM25 keyword matches and exact IDs like `W126`, `3.23`, and `AR-402/7`, reranks the combined set, asks OpenAI for the final answer with `[S1]`-style source labels, and writes the full exchange to SQLite `chat_history`.

History endpoints:

```text
GET /chat/history?user_id=user-1&project_id=project-10&limit=50&offset=0
DELETE /chat/history?user_id=user-1&project_id=project-10
```

### Summary

```text
GET /summary?user_id=user-1&project_id=project-10
```

The GET route accepts only `user_id` and `project_id`. It uses the latest
divisions and instructions saved during document upload for that user/project
scope.

```text
response sections:
  estimated_value
  duration_weeks
  labor_hours
  total_items
  key_highlights: title, description, type
  selected_divisions: code, name
```

The API builds an estimator-focused retrieval query from the saved CSI divisions, instructions, and addendum/change terms, retrieves existing project chunks from Qdrant, asks OpenAI for JSON, normalizes the response, and converts it into the summary response used by the screen. `Total Items` is the uploaded file count. The executive summary covers both original tender documents and addenda, and the summary includes an addenda changes highlight.
