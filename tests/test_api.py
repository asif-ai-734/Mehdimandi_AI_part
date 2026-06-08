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
from app.models import AnalysisRules, Base, ChatHistory, Document
from app.routes import chat as chat_routes
from app.routes import documents as document_routes
from app.routes import pricing as pricing_routes
from app.routes import scope as scope_routes
from app.routes import section_reanalysis as section_reanalysis_routes
from app.routes import summary as summary_routes
from app.schemas.section_reanalysis import ProposedChanges
from app.services import rag_service as rag_service_module
from app.services.file_extractor import ExtractedTextSection
from app.services.rag_service import RAGService


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

    def __init__(self):
        self.last_page_chat_request = None

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

    def generate_page_chat_response(self, query, user_id, project_id, page, page_json):
        self.last_page_chat_request = {
            "query": query,
            "user_id": user_id,
            "project_id": project_id,
            "page": page,
            "page_json": page_json,
        }
        return f"{page} page answer for: {query}", [f"page_json:{page}"]

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


class FakeScopeService:
    """Small test double for scope re-analysis."""

    def __init__(self):
        self.instructions = []

    def extract_scope_of_work(self, user_id, project_id, divisions, instructions):
        self.instructions.append(instructions)
        items = [
            {
                "id": 1,
                "title": "Install doors",
                "division_code": "08",
                "division_label": "Openings",
                "quantity": {"value": 12, "unit": "ea"},
                "specifications": "Door installation scope.",
                "references": [{"code": "08 10 00", "title": "Doors", "page": 4}],
            }
        ]

        if "Exclude painting" not in instructions:
            items.append(
                {
                    "id": 2,
                    "title": "Paint doors",
                    "division_code": "09",
                    "division_label": "Finishes",
                    "quantity": {"value": 12, "unit": "ea"},
                    "specifications": "Painting scope.",
                    "references": [
                        {"code": "09 90 00", "title": "Painting", "page": 8}
                    ],
                }
            )

        return {"items": items}


class FakeSectionReanalysisService:
    """Small test double for proposed-change generation."""

    def build_proposed_changes(self, tab, ai_instructions, previous, updated):
        return ProposedChanges(
            changes_label="Scope Changes",
            changes=["Remove painting scope items"],
            pricing_impact="Review Division 09 pricing impact",
            affected_tabs=["Scope", "Pricing", "Exclusions"],
        )


class FakeQuery:
    """Small query object for tests that do not need a stored document row."""

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return None


class FakeDbSession:
    """Small DB object for RAG metadata tests."""

    def query(self, *args, **kwargs):
        return FakeQuery()

    def commit(self):
        pass


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
    return fake_service


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

    def test_chat_accepts_page_json_context(self, client, fake_rag_service):
        """Test chat can answer from the current page JSON."""
        page_json = {
            "title": "Scope of Work",
            "total_items": 1,
            "items": [
                {
                    "id": 1,
                    "title": "Install doors",
                    "division_code": "08",
                    "quantity": {"value": 12, "unit": "ea"},
                }
            ],
        }

        chat_response = client.post(
            "/chat",
            json={
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
                "page": "scope",
                "page_json": page_json,
                "message": "How many scope items are shown?",
            },
        )

        assert chat_response.status_code == 200
        data = chat_response.json()
        assert data["assistant_response"] == (
            "scope page answer for: How many scope items are shown?"
        )
        assert data["sources"] == ["page_json:scope"]
        assert fake_rag_service.last_page_chat_request == {
            "query": "How many scope items are shown?",
            "user_id": TEST_USER_ID,
            "project_id": TEST_PROJECT_ID,
            "page": "scope",
            "page_json": page_json,
        }

    def test_chat_requires_page_json_when_page_is_sent(self, client, fake_rag_service):
        """Test page-scoped chat requires the current page JSON."""
        chat_response = client.post(
            "/chat",
            json={
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
                "page": "pricing",
                "message": "What pricing impacts are shown?",
            },
        )

        assert chat_response.status_code == 400
        assert chat_response.json()["detail"] == (
            "page_json is required when page is provided"
        )


class TestRagMetadata:
    """Vector metadata tests."""

    def test_process_document_stores_pdf_page_numbers(self, monkeypatch):
        """Test PDF chunks include one-based page metadata before embedding."""
        captured = {}

        class FakeEmbeddingsService:
            def embed_texts(self, texts, normalize=True):
                captured["texts"] = texts
                return [[0.1, 0.2, 0.3] for _ in texts]

        class FakeQdrantService:
            def upsert_points(self, embeddings, metadatas):
                captured["metadatas"] = metadatas
                return ["point-1", "point-2"]

        service = RAGService.__new__(RAGService)
        service.embeddings_service = FakeEmbeddingsService()
        service.qdrant_service = FakeQdrantService()

        monkeypatch.setattr(
            rag_service_module,
            "extract_text_sections",
            lambda file_path, file_type: [
                ExtractedTextSection(text="First page scope item.", page_no=1),
                ExtractedTextSection(text="Second page pricing note.", page_no=2),
            ],
        )
        monkeypatch.setattr(
            rag_service_module,
            "split_text_into_chunks",
            lambda text, chunk_size, chunk_overlap: [text],
        )

        total_chunks, error = service.process_document(
            file_path="spec.pdf",
            file_type="pdf",
            document_id=1,
            user_id=TEST_USER_ID,
            project_id=TEST_PROJECT_ID,
            project_name=TEST_PROJECT_NAME,
            project_address=TEST_PROJECT_ADDRESS,
            filename="spec.pdf",
            db=FakeDbSession(),
        )

        assert total_chunks == 2
        assert error == ""
        assert captured["texts"] == [
            "First page scope item.",
            "Second page pricing note.",
        ]
        assert [metadata["page_no"] for metadata in captured["metadatas"]] == [1, 2]
        assert [metadata["page_number"] for metadata in captured["metadatas"]] == [1, 2]
        assert [metadata["page"] for metadata in captured["metadatas"]] == [1, 2]

    def test_format_source_prefers_page_no(self):
        """Test retrieval source labels include page metadata."""
        source = RAGService._format_source(
            {
                "filename": "spec.pdf",
                "source_type": "document",
                "page_no": 7,
                "chunk_type": "raw_text",
            },
            index=1,
        )

        assert source == "S1: spec.pdf | document | page 7 | raw_text"


class TestSectionReanalysis:
    """Temporary AI instruction re-analysis tests."""

    def test_reanalyze_scope_returns_previous_updated_and_proposed_changes(
        self,
        client,
        fake_rag_service,
        monkeypatch,
    ):
        """Test section re-analysis returns current output plus proposed changes."""
        fake_scope_service = FakeScopeService()
        monkeypatch.setattr(scope_routes, "get_scope_service", lambda: fake_scope_service)
        monkeypatch.setattr(
            section_reanalysis_routes,
            "get_section_reanalysis_service",
            lambda: FakeSectionReanalysisService(),
        )

        upload_response = client.post(
            "/documents/upload",
            data={
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
                "project_name": TEST_PROJECT_NAME,
                "project_address": TEST_PROJECT_ADDRESS,
                "divisions": '["08", "09"]',
                "instructions": "Focus on openings and finishes.",
            },
            files=[
                ("files", ("spec.txt", b"base specification", "text/plain")),
            ],
        )
        assert upload_response.status_code == 200

        response = client.post(
            "/analysis/reanalyze",
            json={
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
                "tab": "scope",
                "ai_instructions": "Exclude painting from scope",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == TEST_USER_ID
        assert data["project_id"] == TEST_PROJECT_ID
        assert data["tab"] == "scope"
        assert data["previous"]["total_items"] == 2
        assert data["updated"]["total_items"] == 1
        assert data["updated"]["items"][0]["scopeItem"] == "Install doors"
        assert data["proposed_changes"] == {
            "title": "Proposed Changes",
            "changes_label": "Scope Changes",
            "changes": ["Remove painting scope items"],
            "pricing_impact": "Review Division 09 pricing impact",
            "affected_tabs": ["Scope", "Pricing", "Exclusions"],
            "notes": None,
        }
        assert fake_scope_service.instructions == [
            "Focus on openings and finishes.",
            (
                "Focus on openings and finishes.\n\n"
                "Temporary AI instruction for this scope re-analysis: "
                "Exclude painting from scope"
            ),
        ]


class TestPricing:
    """Pricing response contract tests."""

    def test_build_pricing_response_returns_pricing_screen_shape(self):
        """Test pricing output matches the frontend pricing schema."""
        response = pricing_routes.build_pricing_response(
            {
                "comparison": {
                    "aiDraftEstimate": 485000,
                    "estimatorFinalPrice": None,
                    "variance": None,
                },
                "aiDraftEstimateBreakdown": [
                    {
                        "division": "01",
                        "name": "General Requirements",
                        "amount": 72750,
                        "editable": True,
                    }
                ],
                "additionalCostItems": [
                    {
                        "name": "Performance Bond",
                        "description": "5% of contract value",
                        "amount": 24250,
                        "editable": True,
                    },
                    {
                        "name": "Builder Risk Insurance",
                        "description": "Project-specific insurance coverage",
                        "amount": 5000,
                        "editable": True,
                    },
                    {
                        "name": "Coordination Costs",
                        "description": "Site meetings, scheduling, RFIs",
                        "amount": 15000,
                        "editable": True,
                    },
                    {
                        "name": "Contingency (5%)",
                        "description": "Risk buffer",
                        "amount": 24250,
                        "editable": True,
                    }
                ],
                "missingInformation": [
                    {
                        "title": "Supplier quotes missing",
                        "description": (
                            "Door hardware and window frame pricing not "
                            "confirmed with suppliers"
                        ),
                        "severity": "critical",
                    }
                ],
                "pricingBasisAndReasoning": [
                    {
                        "title": "Bonds",
                        "description": (
                            "Performance bond calculated as 5% of contract value"
                        ),
                    },
                    {
                        "title": "Insurance",
                        "description": "Builder risk insurance requirement was identified",
                    },
                    {
                        "title": "Coordination",
                        "description": "Site meetings, scheduling, and RFIs require coordination time",
                    },
                    {
                        "title": "Contingency",
                        "description": (
                            "5% applied due to limited site access and material "
                            "approval lead times"
                        ),
                    }
                ],
            }
        )

        assert response.model_dump() == {
            "comparison": {
                "aiDraftEstimate": 485000,
                "estimatorFinalPrice": None,
                "variance": None,
            },
            "aiDraftEstimateBreakdown": [
                {
                    "division": "01",
                    "name": "General Requirements",
                    "amount": 72750,
                    "editable": True,
                }
            ],
            "additionalCostItems": [
                {
                    "name": "Bonds",
                    "description": "5% of contract value",
                    "amount": 24250,
                    "editable": True,
                },
                {
                    "name": "Insurance",
                    "description": "Project-specific insurance coverage",
                    "amount": 5000,
                    "editable": True,
                },
                {
                    "name": "Coordination",
                    "description": "Site meetings, scheduling, RFIs",
                    "amount": 15000,
                    "editable": True,
                },
                {
                    "name": "Contingency",
                    "description": "Risk buffer",
                    "amount": 24250,
                    "editable": True,
                }
            ],
            "missingInformation": [
                {
                    "title": "Supplier quotes missing",
                    "description": (
                        "Door hardware and window frame pricing not "
                        "confirmed with suppliers"
                    ),
                    "severity": "critical",
                }
            ],
            "pricingBasisAndReasoning": [
                {
                    "title": "Bonds",
                    "description": (
                        "Performance bond calculated as 5% of contract value"
                    ),
                },
                {
                    "title": "Insurance",
                    "description": "Builder risk insurance requirement was identified",
                },
                {
                    "title": "Coordination",
                    "description": "Site meetings, scheduling, and RFIs require coordination time",
                },
                {
                    "title": "Contingency",
                    "description": (
                        "5% applied due to limited site access and material "
                        "approval lead times"
                    ),
                }
            ],
        }

    def test_build_pricing_response_forces_fixed_additional_cost_categories(self):
        response = pricing_routes.build_pricing_response(
            {
                "comparison": {},
                "aiDraftEstimateBreakdown": [],
                "additionalCostItems": [
                    {
                        "name": "Night Work Premium",
                        "description": "18% labor cost increase",
                        "amount": 32400,
                    }
                ],
                "missingInformation": [],
                "pricingBasisAndReasoning": [
                    {
                        "title": "Division Pricing",
                        "description": "Based on extracted quantities.",
                    }
                ],
            }
        )

        data = response.model_dump()

        assert [item["name"] for item in data["additionalCostItems"]] == [
            "Bonds",
            "Insurance",
            "Coordination",
            "Contingency",
        ]
        assert all(
            item["amount"] is None
            for item in data["additionalCostItems"]
        )
        assert [item["title"] for item in data["pricingBasisAndReasoning"]] == [
            "Bonds",
            "Insurance",
            "Coordination",
            "Contingency",
        ]


class TestAnalysisRules:
    """User-level AI analysis rule tests."""

    def test_analysis_rules_route_updates_optional_rules(self, client):
        response = client.patch(
            "/analysis_rules",
            json={
                "user_id": TEST_USER_ID,
                "general_instructions": "Use conservative estimating language.",
                "scope_analysis_instructions": "Prioritize base bid scope.",
                "assumptions_instructions": "Flag owner-provided items.",
                "exclusions_instructions": "Exclude unsupported alternates.",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == TEST_USER_ID
        assert data["pricing_specific_instructions"] == ""
        assert data["scope_analysis_instructions"] == "Prioritize base bid scope."
        assert data["saved_scope_instructions"] == (
            "General analysis rules:\n"
            "Use conservative estimating language.\n\n"
            "Scope analysis rules:\n"
            "Prioritize base bid scope."
        )
        assert data["saved_assumptions_instructions"] == (
            "General analysis rules:\n"
            "Use conservative estimating language.\n\n"
            "Assumptions rules:\n"
            "Flag owner-provided items."
        )
        assert data["saved_exclusions_instructions"] == (
            "General analysis rules:\n"
            "Use conservative estimating language.\n\n"
            "Exclusions rules:\n"
            "Exclude unsupported alternates."
        )

    def test_saved_scope_inputs_include_analysis_rules(self, setup_db):
        db = TestingSessionLocal()
        try:
            db.add(
                Document(
                    filename="spec.txt",
                    file_path="spec.txt",
                    file_type="txt",
                    project_id=TEST_PROJECT_ID,
                    user_id=TEST_USER_ID,
                    divisions='["06"]',
                    instructions="Uploaded project instruction.",
                )
            )
            db.add(
                AnalysisRules(
                    user_id=TEST_USER_ID,
                    general_instructions="Apply user-level rules.",
                    scope_analysis_instructions="Use scope-specific rule.",
                )
            )
            db.commit()

            divisions, instructions = scope_routes.get_saved_scope_inputs(
                db=db,
                user_id=TEST_USER_ID,
                project_id=TEST_PROJECT_ID,
            )
        finally:
            db.close()

        assert divisions == ["06"]
        assert instructions == (
            "Uploaded project instruction.\n\n"
            "General analysis rules:\n"
            "Apply user-level rules.\n\n"
            "Scope analysis rules:\n"
            "Use scope-specific rule."
        )


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


class TestAdmin:
    """Admin route tests."""

    def test_admin_users_returns_user_project_rows_without_input(self, client):
        db = TestingSessionLocal()
        try:
            db.add_all(
                [
                    Document(
                        filename="spec.txt",
                        file_path="spec.txt",
                        file_type="txt",
                        project_id=TEST_PROJECT_ID,
                        user_id=TEST_USER_ID,
                        project_name=TEST_PROJECT_NAME,
                        total_chunks=2,
                    ),
                    Document(
                        filename="office.txt",
                        file_path="office.txt",
                        file_type="txt",
                        project_id="project-789",
                        user_id="user-999",
                        project_name="Office Fitout",
                        total_chunks=1,
                    ),
                ]
            )
            db.commit()
        finally:
            db.close()

        response = client.get("/admin/users")

        assert response.status_code == 200
        assert response.json() == [
            {
                "user_id": TEST_USER_ID,
                "project_id": TEST_PROJECT_ID,
                "project_name": TEST_PROJECT_NAME,
            },
            {
                "user_id": "user-999",
                "project_id": "project-789",
                "project_name": "Office Fitout",
            },
        ]

    def test_admin_users_coerces_numeric_ids_to_strings(self, client):
        db = TestingSessionLocal()
        try:
            db.add(
                Document(
                    filename="numeric-ids.txt",
                    file_path="numeric-ids.txt",
                    file_type="txt",
                    project_id=1,
                    user_id=1,
                    project_name=100,
                    total_chunks=1,
                )
            )
            db.commit()
        finally:
            db.close()

        response = client.get("/admin/users")

        assert response.status_code == 200
        assert response.json() == [
            {
                "user_id": "1",
                "project_id": "1",
                "project_name": "100",
            }
        ]

        logs_response = client.get("/admin/ai-logs")

        assert logs_response.status_code == 200
        log = logs_response.json()[0]
        assert log["project_name"] == "100"
        assert log["details"]["project"] == {
            "user_id": "1",
            "user_owner": "1",
            "project_id": "1",
            "project_name": "100",
        }

    def test_admin_ai_logs_returns_table_and_nested_detail_shape(self, client):
        db = TestingSessionLocal()
        try:
            db.add(
                Document(
                    filename="Door Schedule Addendum-4582.pdf",
                    file_path="door-schedule.pdf",
                    file_type="pdf",
                    project_id=TEST_PROJECT_ID,
                    user_id=TEST_USER_ID,
                    project_name=TEST_PROJECT_NAME,
                    divisions='["08", "09"]',
                    instructions="Extract all door specifications and pricing information",
                    file_size=2048,
                    page_count=12,
                    total_chunks=3,
                )
            )
            db.commit()
        finally:
            db.close()

        response = client.get("/admin/ai-logs")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1

        log = data[0]
        assert set(log.keys()) == {
            "file_name",
            "project_name",
            "stage",
            "status",
            "error_message",
            "timestamp",
            "duration",
            "actions",
            "details",
        }
        assert log["file_name"] == "Door Schedule Addendum-4582.pdf"
        assert log["project_name"] == TEST_PROJECT_NAME
        assert log["stage"] == "Extraction"
        assert log["status"] == "Success"
        assert log["error_message"] is None
        assert log["duration"] == "3s"
        assert log["actions"] == {
            "view": True,
            "retry": False,
            "review_and_fix": False,
        }
        assert log["details"]["document"] == {
            "file_name": "Door Schedule Addendum-4582.pdf",
            "document_id": "DOC-1",
            "file_type": "pdf",
            "source_type": "document",
        }
        assert log["details"]["project"] == {
            "user_id": TEST_USER_ID,
            "user_owner": TEST_USER_ID,
            "project_id": TEST_PROJECT_ID,
            "project_name": TEST_PROJECT_NAME,
        }
        assert log["details"]["processing"] == {
            "stage": "Extraction",
            "status": "Success",
            "error_message": None,
            "divisions_selected": ["08", "09"],
            "instructions_given": "Extract all door specifications and pricing information",
        }
        assert log["details"]["output"] == {
            "output_version": "v2",
            "total_chunks": 3,
            "page_count": 12,
            "file_size": 2048,
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
