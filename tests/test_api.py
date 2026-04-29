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
from app.routes import analysis as analysis_routes
from app.routes import chat as chat_routes
from app.routes import documents as document_routes


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
    monkeypatch.setattr(analysis_routes, "get_analysis_service", lambda: fake_service)


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
        assert all(doc["project_name"] == TEST_PROJECT_NAME for doc in documents)
        assert all(doc["project_address"] == TEST_PROJECT_ADDRESS for doc in documents)


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


class TestAnalysis:
    """Tender analysis tests."""

    def test_tender_analysis_returns_preview_shape(self, client, fake_analysis_service):
        """Test tender analysis accepts selected divisions and instructions."""
        response = client.post(
            "/analysis/tender",
            json={
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
                "divisions": ["06", "08"],
                "instructions": "Focus on openings and wood scope.",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "preview"
        assert data["metrics"]["risk_score"] == 68
        assert data["selected_divisions"][0]["code"] == "06"
        assert data["analysis_preview"]["scope_of_work"]["items"] == [
            "Wooden doors",
            "Aluminum windows",
        ]
        assert data["sources"] == ["spec.pdf"]

    def test_tender_analysis_accepts_single_division_alias(self, client, fake_analysis_service):
        """Test client can send division instead of divisions."""
        response = client.post(
            "/analysis/tender",
            json={
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
                "division": "Div 06",
                "instructions": "",
            },
        )

        assert response.status_code == 200
        assert response.json()["selected_divisions"][0]["code"] == "06"

    def test_tender_analysis_accepts_divion_typo_alias(self, client, fake_analysis_service):
        """Test client can send divion typo instead of divisions."""
        response = client.post(
            "/analysis/tender",
            json={
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
                "divion": "Div 06",
                "instructions": "",
            },
        )

        assert response.status_code == 200
        assert response.json()["selected_divisions"][0]["code"] == "06"

    def test_tender_analysis_requires_divisions(self, client, fake_analysis_service):
        """Test analysis rejects missing divisions."""
        response = client.post(
            "/analysis/tender",
            json={
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
                "divisions": [],
                "instructions": "Analyze tender.",
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "At least one division must be selected"

    def test_tender_analysis_requires_valid_division_code(self, client, fake_analysis_service):
        """Test analysis rejects division values without a CSI code."""
        response = client.post(
            "/analysis/tender",
            json={
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
                "divisions": ["openings"],
                "instructions": "Analyze tender.",
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "At least one valid division code must be selected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
