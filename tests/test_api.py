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
from app.routes import summary as summary_routes


SQLALCHEMY_DATABASE_URL = "sqlite://"
TEST_USER_ID = "user-123"
TEST_PROJECT_ID = "project-456"
TEST_PROJECT_NAME = "Cedar Ridge Exterior"
TEST_PROJECT_ADDRESS = "225 Confederation Drive, Toronto"

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
        project_name,
        project_address,
        filename,
        source_type,
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


class FakeAnalysisService:
    """Small test double for tender analysis."""

    def analyze_tender(self, user_id, project_id, divisions, instructions):
        return {
            "user_id": user_id,
            "project_id": project_id,
            "status": "preview",
            "instructions": instructions,
            "selected_divisions": [
                {
                    "code": "06",
                    "label": "Wood, Plastics & Composites",
                    "allocation_percent": 50,
                },
                {
                    "code": "08",
                    "label": "Openings",
                    "allocation_percent": 50,
                },
            ],
            "metrics": {
                "estimated_value": "$485,000",
                "duration": "14 weeks",
                "labor_hours": "2,840 hours",
                "complexity": "Medium-High",
                "risk_score": 68,
            },
            "analysis_preview": {
                "executive_summary": {
                    "title": "Executive Summary",
                    "content": "Tender analysis preview.",
                    "badge": "3 key points identified",
                },
                "scope_of_work": {
                    "title": "Scope of Work",
                    "items": ["Wooden doors", "Aluminum windows"],
                    "badge": "2 items with quantities",
                },
                "risk_assessment": {
                    "title": "Risk Assessment",
                    "items": [
                        {"label": "Limited site access", "severity": "High"},
                    ],
                    "badge": "1 risk identified",
                },
                "pricing_impacts": {
                    "title": "Pricing Impacts",
                    "items": [
                        {"label": "Performance Bond", "value": "$24,250"},
                    ],
                    "badge": "1 cost factor analyzed",
                },
                "addenda_summary": {
                    "title": "Addenda Summary",
                    "content": "Addendum 01 changes the window specification.",
                    "badge": "1 addendum change identified",
                },
            },
            "sources": ["spec.pdf"],
        }


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


@pytest.fixture
def fake_analysis_service(monkeypatch):
    """Patch analysis service to avoid OpenAI/Qdrant during API tests."""
    fake_service = FakeAnalysisService()
    monkeypatch.setattr(summary_routes, "get_analysis_service", lambda: fake_service)


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
            data={
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
                "project_name": TEST_PROJECT_NAME,
                "project_address": TEST_PROJECT_ADDRESS,
                "divisions": '["06", "08"]',
                "instructions": "Focus on openings and wood scope.",
            },
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
        assert all(item["source_type"] == "document" for item in data)
        assert all(item["divisions"] == ["06", "08"] for item in data)
        assert all(item["instructions"] == "Focus on openings and wood scope." for item in data)
        assert all(item["total_chunks"] == 2 for item in data)
        assert all(item["project_name"] == TEST_PROJECT_NAME for item in data)
        assert all(item["project_address"] == TEST_PROJECT_ADDRESS for item in data)

        list_response = client.get(
            "/documents",
            params={"user_id": TEST_USER_ID, "project_id": TEST_PROJECT_ID},
        )
        assert list_response.status_code == 200
        documents = list_response.json()
        assert len(documents) == 2
        assert all(doc["user_id"] == TEST_USER_ID for doc in documents)
        assert all(doc["project_id"] == TEST_PROJECT_ID for doc in documents)
        assert all(doc["source_type"] == "document" for doc in documents)
        assert all(doc["divisions"] == ["06", "08"] for doc in documents)
        assert all(doc["instructions"] == "Focus on openings and wood scope." for doc in documents)
        assert all(doc["project_name"] == TEST_PROJECT_NAME for doc in documents)
        assert all(doc["project_address"] == TEST_PROJECT_ADDRESS for doc in documents)

    def test_upload_accepts_multiple_addendum_files(self, client, fake_rag_service):
        """Test upload accepts addendum as a separate multiple-file field."""
        response = client.post(
            "/documents/upload",
            data={
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
                "project_name": TEST_PROJECT_NAME,
                "project_address": TEST_PROJECT_ADDRESS,
                "divisions": "06,08",
                "instructions": "Review addenda.",
            },
            files=[
                ("files", ("spec.txt", b"base specification", "text/plain")),
                ("addendum", ("addendum-01.txt", b"window changes", "text/plain")),
                ("addendum", ("addendum-02.txt", b"schedule changes", "text/plain")),
            ],
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert [item["source_type"] for item in data] == [
            "document",
            "addendum",
            "addendum",
        ]
        assert all(item["status"] == "success" for item in data)
        assert all(item["divisions"] == ["06", "08"] for item in data)
        assert all(item["instructions"] == "Review addenda." for item in data)

        list_response = client.get(
            "/documents",
            params={"user_id": TEST_USER_ID, "project_id": TEST_PROJECT_ID},
        )
        assert list_response.status_code == 200
        documents = list_response.json()
        assert {doc["filename"]: doc["source_type"] for doc in documents} == {
            "spec.txt": "document",
            "addendum-01.txt": "addendum",
            "addendum-02.txt": "addendum",
        }
        assert all(doc["divisions"] == ["06", "08"] for doc in documents)
        assert all(doc["instructions"] == "Review addenda." for doc in documents)


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


class TestSummary:
    """Project summary tests."""

    def test_summary_get_route_returns_dashboard_shape(
        self,
        client,
        fake_rag_service,
        fake_analysis_service,
    ):
        """Test summary uses saved upload inputs and returns UI-shaped output."""
        upload_response = client.post(
            "/documents/upload",
            data={
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
                "project_name": TEST_PROJECT_NAME,
                "project_address": TEST_PROJECT_ADDRESS,
                "divisions": '["06", "08"]',
                "instructions": "Focus on openings and wood scope.",
            },
            files=[
                ("files", ("spec.txt", b"base specification", "text/plain")),
                ("addendum", ("addendum-01.txt", b"window changes", "text/plain")),
                ("addendum", ("addendum-02.txt", b"schedule changes", "text/plain")),
            ],
        )
        assert upload_response.status_code == 200

        response = client.get(
            "/summary",
            params={
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert set(data.keys()) == {
            "estimated_value",
            "duration_weeks",
            "labor_hours",
            "total_items",
            "key_highlights",
            "selected_divisions",
        }
        assert data["estimated_value"] == 485000
        assert data["duration_weeks"] == 14
        assert data["labor_hours"] == 2840
        assert data["total_items"] == 3
        assert data["key_highlights"][0]["title"] == "Scope Summary"
        assert data["key_highlights"][0]["type"] == "scope"
        assert data["key_highlights"][1]["title"] == "Pricing Impacts"
        assert data["key_highlights"][1]["type"] == "pricing"
        assert data["key_highlights"][2]["title"] == "Risks & Coordination"
        assert data["key_highlights"][2]["type"] == "risk"
        assert data["key_highlights"][3]["title"] == "Addenda Changes"
        assert data["key_highlights"][3]["description"] == (
            "Addendum 01 changes the window specification."
        )
        assert data["key_highlights"][3]["type"] == "addenda"
        assert data["selected_divisions"][0] == {
            "code": "06",
            "name": "Wood & Plastics",
        }

    def test_summary_get_route_accepts_only_user_and_project_id(
        self,
        client,
        fake_rag_service,
        fake_analysis_service,
    ):
        """Test GET summary input is only user_id and project_id."""
        upload_response = client.post(
            "/documents/upload",
            data={
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
                "project_name": TEST_PROJECT_NAME,
                "project_address": TEST_PROJECT_ADDRESS,
                "divisions": '["06", "08"]',
                "instructions": "Saved upload instructions.",
            },
            files=[
                ("files", ("spec.txt", b"base specification", "text/plain")),
            ],
        )
        assert upload_response.status_code == 200

        response = client.get(
            "/summary",
            params={
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["selected_divisions"][0]["code"] == "06"

    def test_summary_requires_uploaded_documents(self, client, fake_analysis_service):
        """Test summary rejects projects with no uploaded files."""
        response = client.get(
            "/summary",
            params={
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
            },
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "No uploaded documents found for this project"

    def test_summary_requires_saved_divisions(
        self,
        client,
        fake_rag_service,
        fake_analysis_service,
    ):
        """Test summary rejects uploads without saved divisions."""
        upload_response = client.post(
            "/documents/upload",
            data={
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
            },
            files=[
                ("files", ("spec.txt", b"base specification", "text/plain")),
            ],
        )
        assert upload_response.status_code == 200

        response = client.get(
            "/summary",
            params={
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "No saved divisions found for this project"

    def test_summary_requires_valid_saved_division_code(
        self,
        client,
        fake_rag_service,
        fake_analysis_service,
    ):
        """Test summary rejects saved division values without a CSI code."""
        upload_response = client.post(
            "/documents/upload",
            data={
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
                "divisions": "openings",
            },
            files=[
                ("files", ("spec.txt", b"base specification", "text/plain")),
            ],
        )
        assert upload_response.status_code == 200

        response = client.get(
            "/summary",
            params={
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "At least one valid division code must be selected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
