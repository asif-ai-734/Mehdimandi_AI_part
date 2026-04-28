"""
Test cases for the RAG API.
Run with: pytest tests/test_api.py -v
"""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.main import app
from app.models import Base, ChatHistory, Document
from app.routes import chat as chat_routes
from app.routes import documents as document_routes


SQLALCHEMY_DATABASE_URL = "sqlite://"
TEST_USER_ID = 123
TEST_PROJECT_ID = 456

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


class FakeRagService:
    """Small test double for document processing and chat persistence."""

    def process_document(
        self,
        file_path,
        file_type,
        document_id,
        user_id,
        project_id,
        filename,
        db,
    ):
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.total_chunks = 2
            db.commit()
        return 2, ""

    def generate_chat_response(self, query, user_id, project_id, db):
        return f"Answer for: {query}", ["source.txt"]

    def save_chat_history(
        self,
        user_id,
        project_id,
        user_message,
        assistant_response,
        sources,
        db,
    ):
        chat = ChatHistory(
            user_id=user_id,
            project_id=project_id,
            user_message=user_message,
            assistant_response=assistant_response,
            sources=json.dumps(sources) if sources else None,
        )
        db.add(chat)
        db.commit()
        db.refresh(chat)
        return chat


@pytest.fixture(scope="function")
def setup_db():
    """Create test database before each test."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(setup_db):
    """Create test client without running external startup services."""
    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    yield test_client
    test_client.close()
    app.dependency_overrides.clear()


@pytest.fixture
def fake_rag_service(monkeypatch, tmp_path):
    """Patch route services to avoid OpenAI/Qdrant during API tests."""
    fake_service = FakeRagService()
    monkeypatch.setattr(chat_routes, "get_rag_service", lambda: fake_service)
    monkeypatch.setattr(document_routes, "get_rag_service", lambda: fake_service)
    monkeypatch.setattr(document_routes.settings, "upload_dir", str(tmp_path))


class TestHealth:
    """Health check tests."""

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestDocuments:
    """Document upload tests for the simplified user/project scope."""

    def test_upload_multiple_documents_processes_synchronously(self, client, fake_rag_service):
        """Test upload accepts user_id, project_id, and multiple files."""
        response = client.post(
            "/documents/upload",
            data={"user_id": str(TEST_USER_ID), "project_id": str(TEST_PROJECT_ID)},
            files=[
                ("files", ("first.txt", b"first document", "text/plain")),
                ("files", ("second.txt", b"second document", "text/plain")),
            ],
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert {item["filename"] for item in data} == {"first.txt", "second.txt"}
        assert all(item["status"] == "success" for item in data)
        assert all(item["total_chunks"] == 2 for item in data)

        list_response = client.get(
            "/documents",
            params={"user_id": TEST_USER_ID, "project_id": TEST_PROJECT_ID},
        )
        assert list_response.status_code == 200
        documents = list_response.json()
        assert len(documents) == 2
        assert all(doc["user_id"] == TEST_USER_ID for doc in documents)
        assert all(doc["project_id"] == TEST_PROJECT_ID for doc in documents)


class TestChat:
    """Chat history tests with explicit user_id and project_id."""

    def test_chat_saves_history_to_sqlite(self, client, fake_rag_service):
        """Test chat uses user/project scope and persists to chat_history."""
        chat_response = client.post(
            "/chat",
            json={
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
                "message": "What is in this project?",
            },
        )

        assert chat_response.status_code == 200
        data = chat_response.json()
        assert data["user_id"] == TEST_USER_ID
        assert data["project_id"] == TEST_PROJECT_ID
        assert data["user_message"] == "What is in this project?"
        assert data["assistant_response"] == "Answer for: What is in this project?"
        assert data["sources"] == ["source.txt"]

        history_response = client.get(
            "/chat/history",
            params={"user_id": TEST_USER_ID, "project_id": TEST_PROJECT_ID},
        )
        assert history_response.status_code == 200
        history = history_response.json()
        assert history["total_messages"] == 1
        assert history["messages"][0]["id"] == data["id"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
