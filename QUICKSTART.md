# Quick Start

This guide gets the FastAPI document RAG backend running locally, uploads a scoped project, and calls the main chat and analysis routes.

## Option 1: Docker Compose

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Set `OPENAI_API_KEY` in `.env`.

3. Start the API and Qdrant:

```bash
docker-compose up -d
```

4. Open Swagger UI:

```text
http://localhost:8000/docs
```

## Option 2: Local Development

1. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy environment defaults and set `OPENAI_API_KEY`:

```bash
cp .env.example .env
```

4. Start Qdrant:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

5. Run the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

6. Open Swagger UI:

```text
http://localhost:8000/docs
```

## Example Workflow

Pick any scope values. They are plain strings:

```bash
USER_ID=user-1
PROJECT_ID=project-10
```

Upload documents and addenda:

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "user_id=$USER_ID" \
  -F "project_id=$PROJECT_ID" \
  -F "project_name=Cedar Ridge Exterior" \
  -F "project_address=225 Confederation Drive, Toronto" \
  -F "divisions=[\"06\", \"08\"]" \
  -F "instructions=Focus on openings and wood scope." \
  -F "files=@document1.pdf" \
  -F "files=@document2.docx" \
  -F "files=@notes.txt" \
  -F "addendum=@addendum-01.pdf" \
  -F "addendum=@addendum-02.docx"
```

The route returns after files are processed and embeddings are stored in Qdrant. Files uploaded through `files` are saved with `source_type=document`; files uploaded through `addendum` are saved with `source_type=addendum`.

List uploaded documents:

```bash
curl "http://localhost:8000/documents?user_id=$USER_ID&project_id=$PROJECT_ID"
```

List source resources for the analysis screen:

```bash
curl "http://localhost:8000/resources?user_id=$USER_ID&project_id=$PROJECT_ID"
```

Run chat:

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-1",
    "project_id": "project-10",
    "message": "What are the main pricing risks?"
  }'
```

Read chat history:

```bash
curl "http://localhost:8000/chat/history?user_id=$USER_ID&project_id=$PROJECT_ID&limit=10&offset=0"
```

## Analysis Routes

All analysis routes are grouped under the `analysis` tag in Swagger. They use saved upload metadata, so the request only needs `user_id` and `project_id`.

```bash
curl "http://localhost:8000/summary?user_id=$USER_ID&project_id=$PROJECT_ID"
curl "http://localhost:8000/scope?user_id=$USER_ID&project_id=$PROJECT_ID"
curl "http://localhost:8000/pricing?user_id=$USER_ID&project_id=$PROJECT_ID"
curl "http://localhost:8000/risks?user_id=$USER_ID&project_id=$PROJECT_ID"
curl "http://localhost:8000/clarifications?user_id=$USER_ID&project_id=$PROJECT_ID"
curl "http://localhost:8000/assumptions?user_id=$USER_ID&project_id=$PROJECT_ID"
curl "http://localhost:8000/exclusions?user_id=$USER_ID&project_id=$PROJECT_ID"
curl "http://localhost:8000/addenda?user_id=$USER_ID&project_id=$PROJECT_ID"
curl "http://localhost:8000/quote-draft?user_id=$USER_ID&project_id=$PROJECT_ID"
curl "http://localhost:8000/resources?user_id=$USER_ID&project_id=$PROJECT_ID"
```

The AI-backed analysis routes require uploaded documents and saved divisions with at least one numeric CSI code, such as `"06"` or `"08"`.

## Tests

Run the full suite:

```bash
.\venv\Scripts\python.exe -m pytest -v
```

Expected local status after the latest updates:

```text
10 passed
```

## Database

SQLite is the default:

```env
DATABASE_URL=sqlite:///./rag.db
```

Useful inspection query:

```bash
sqlite3 rag.db
.tables
SELECT id, user_id, project_id, filename, source_type, divisions, instructions, file_size, page_count, total_chunks FROM documents;
SELECT id, user_id, project_id, created_at FROM chat_history;
.quit
```

## Cleanup

For Docker Compose:

```bash
docker-compose down
```

For local generated data:

```bash
rm rag.db
rm -rf uploads/
```
