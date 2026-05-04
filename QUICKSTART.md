# Quick Start Guide

## Docker Compose

1. Copy environment defaults:

```bash
cp .env.example .env
```

2. Add `OPENAI_API_KEY` to `.env`.

3. Start the API and Qdrant:

```bash
docker-compose up -d
```

4. Open the API docs:

```text
http://localhost:8000/docs
```

## Local Development

1. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Qdrant:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

4. Run the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Example Usage

Pick any scope values your client wants to use:

```bash
USER_ID=1
PROJECT_ID=10
```

Upload multiple documents. This chunks and embeds them before returning:

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "user_id=$USER_ID" \
  -F "project_id=$PROJECT_ID" \
  -F "divisions=[\"06\", \"08\"]" \
  -F "instructions=Focus on openings and wood scope." \
  -F "files=@document1.pdf" \
  -F "files=@document2.docx" \
  -F "files=@notes.txt" \
  -F "addendum=@addendum-01.pdf" \
  -F "addendum=@addendum-02.docx"
```

Normal files are saved with `source_type=document`; files posted under the
`addendum` field are saved with `source_type=addendum` and included in chat and
analysis summaries.

Run the AI analysis summary with GET query params:

```bash
curl "http://localhost:8000/summary?user_id=$USER_ID&project_id=$PROJECT_ID"
```

The summary request only needs `user_id` and `project_id`. Divisions and
instructions are read from the uploaded documents for that project.

Chat against that scoped knowledge base:

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "project_id": 10,
    "message": "What are the main topics in these documents?"
  }'
```

Read chat history:

```bash
curl "http://localhost:8000/chat/history?user_id=$USER_ID&project_id=$PROJECT_ID&limit=10"
```

List uploaded documents:

```bash
curl "http://localhost:8000/documents?user_id=$USER_ID&project_id=$PROJECT_ID"
```

## Cleanup

```bash
docker-compose down
rm rag.db
rm -rf uploads/
```

## Database

SQLite is the default:

```env
DATABASE_URL=sqlite:///./rag.db
```

Inspect tables:

```bash
sqlite3 rag.db
.tables
SELECT id, user_id, project_id, filename, source_type, divisions, instructions, total_chunks FROM documents;
SELECT id, user_id, project_id, created_at FROM chat_history;
.quit
```
