"""
Database connection and session management.
"""

from sqlalchemy import create_engine
from sqlalchemy import inspect, text
from sqlalchemy.orm import sessionmaker, Session
from app.config import settings

# Create engine
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    echo=False
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_database_schema() -> None:
    """Apply small SQLite-compatible schema updates for existing local DBs."""
    inspector = inspect(engine)
    if "documents" not in inspector.get_table_names():
        return

    existing_columns = {
        column["name"]
        for column in inspector.get_columns("documents")
    }
    statements = []
    if "project_name" not in existing_columns:
        statements.append("ALTER TABLE documents ADD COLUMN project_name VARCHAR(255) NOT NULL DEFAULT ''")
    if "project_address" not in existing_columns:
        statements.append("ALTER TABLE documents ADD COLUMN project_address VARCHAR(500) NOT NULL DEFAULT ''")
    if "source_type" not in existing_columns:
        statements.append("ALTER TABLE documents ADD COLUMN source_type VARCHAR(20) NOT NULL DEFAULT 'document'")
    if "divisions" not in existing_columns:
        statements.append("ALTER TABLE documents ADD COLUMN divisions TEXT NOT NULL DEFAULT '[]'")
    if "instructions" not in existing_columns:
        statements.append("ALTER TABLE documents ADD COLUMN instructions TEXT NOT NULL DEFAULT ''")
    if "file_size" not in existing_columns:
        statements.append("ALTER TABLE documents ADD COLUMN file_size INTEGER NOT NULL DEFAULT 0")
    if "page_count" not in existing_columns:
        statements.append("ALTER TABLE documents ADD COLUMN page_count INTEGER")

    if not statements:
        return

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
