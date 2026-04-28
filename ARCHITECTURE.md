# Document RAG Architecture

This backend is a simple FastAPI document RAG service. There is no auth layer and no project management layer. Callers provide `user_id` and `project_id` directly, and those two values scope uploaded documents, Qdrant retrieval, and SQLite chat history.

## Layers

1. **API routes**
   - `app/routes/documents.py`: upload, list, and delete documents
   - `app/routes/chat.py`: chat, chat history, and history clearing

2. **Services**
   - `app/services/file_extractor.py`: extracts text from PDF, DOCX, and TXT files
   - `app/services/chunker.py`: splits text into overlapping chunks
   - `app/services/drawing_processor.py`: renders PDF pages, asks OpenAI for sheet JSON, synthesizes document summaries, normalizes drawing entities, and builds semantic chunks
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
  user_id
  project_id
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
  "user_id": 1,
  "project_id": 10,
  "document_id": 25,
  "filename": "document.pdf",
  "chunk_index": 0,
  "chunk_type": "keynote",
  "sheet": "AR-601",
  "page": 9,
  "entities": ["3.23", "AR-601", "W126"],
  "source_image": "uploads/1/10/document_25_pages/page_0009.png",
  "source_text_ref": "document.pdf#page=9",
  "text": "Chunk text..."
}
```

## Main Flows

### Document Upload

```text
POST /documents/upload
multipart fields:
  user_id=1
  project_id=10
  files=@one.pdf
  files=@two.txt
```

The API saves each file under `uploads/{user_id}/{project_id}`. DOCX/TXT files follow the standard text extraction and chunking path.

PDFs use the drawing-aware path when `ENABLE_PDF_DRAWING_ANALYSIS=true`:

```text
PDF
-> rendered page PNGs + parser text
-> OpenAI structured JSON per page
-> document-level synthesis
-> entity normalization
-> semantic chunks with sheet/page/image metadata
-> embeddings in Qdrant
```

Processing is synchronous, so the uploaded documents are ready for chat when the response returns.

### Chat

```text
POST /chat
body: {
  "user_id": 1,
  "project_id": 10,
  "message": "What do these files say?"
}
```

The API embeds the question, retrieves scoped vector candidates from Qdrant, scans scoped payloads for BM25 keyword matches and exact IDs like `W126`, `3.23`, and `AR-402/7`, reranks the combined set, asks OpenAI for the final answer with `[S1]`-style source labels, and writes the full exchange to SQLite `chat_history`.

History endpoints:

```text
GET /chat/history?user_id=1&project_id=10&limit=50&offset=0
DELETE /chat/history?user_id=1&project_id=10
```
