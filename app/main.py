"""
FastAPI application for Document RAG System.
User/project-scoped document retrieval and chat system.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
from app.config import settings
from app.database import engine, ensure_database_schema
from app.models import Base
from app.routes import documents, chat, summary
from app.routes.scope import router as scope_router
from app.routes.pricing import router as pricing_router
from app.routes.risks import router as risks_router
from app.routes.clarifications import router as clarifications_router
from app.routes.assumptions import router as assumptions_router
from app.routes.exclusions import router as exclusions_router
from app.routes.addenda import router as addenda_router
from app.routes.quote_draft import router as quote_draft_router
from app.routes.resources import router as resources_router
from app.routes.system import router as system_router
from app.services.embeddings import get_embeddings_service
from app.services.qdrant_service import get_qdrant_service
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle app startup and shutdown."""
    logger.info("Starting up RAG API...")

    Base.metadata.create_all(bind=engine)
    ensure_database_schema()
    logger.info("Database tables initialized")

    try:
        embeddings_service = get_embeddings_service()
        logger.info(f"Embeddings model loaded: {embeddings_service.model_name}")
        logger.info(f"Embedding dimension: {embeddings_service.get_embedding_dimension()}")
    except Exception as e:
        logger.error(f"Error initializing embeddings service: {str(e)}")
        raise

    try:
        qdrant_service = get_qdrant_service(
            vector_size=embeddings_service.get_embedding_dimension()
        )
        qdrant_service.create_collection()
        logger.info(f"Qdrant collection initialized: {qdrant_service.collection_name}")
    except Exception as e:
        logger.error(f"Error initializing Qdrant: {str(e)}")
        raise

    yield

    logger.info("Shutting down RAG API...")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Simple document RAG system with explicit user_id and project_id inputs",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(system_router)
app.include_router(documents.router)
# app.include_router(chat.router)
app.include_router(summary.router)
app.include_router(scope_router)
app.include_router(pricing_router)
app.include_router(risks_router)
app.include_router(clarifications_router)
app.include_router(assumptions_router)
app.include_router(exclusions_router)
app.include_router(addenda_router)
app.include_router(quote_draft_router)
app.include_router(resources_router)



def custom_openapi():
    """
    Override OpenAPI schema to fix Swagger UI file upload rendering.
    Forces 'files' field to render as a proper file picker (format: binary)
    instead of array<string> text input.
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    for path in openapi_schema.get("paths", {}).values():
        for operation in path.values():
            request_body = operation.get("requestBody", {})
            content = request_body.get("content", {})
            form_data = content.get("multipart/form-data", {})
            schema_ref = form_data.get("schema", {}).get("$ref", "")

            if schema_ref:
                schema_name = schema_ref.split("/")[-1]
                schema = openapi_schema["components"]["schemas"].get(schema_name, {})
                properties = schema.get("properties", {})

                for upload_field in ("files", "addendum"):
                    if upload_field not in properties:
                        continue
                    properties[upload_field] = {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "format": "binary"
                        }
                    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/")
async def read_root() -> dict:
    """API health check."""
    return {
        "message": "Document RAG API is running",
        "version": settings.api_version,
        "status": "operational"
    }


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
