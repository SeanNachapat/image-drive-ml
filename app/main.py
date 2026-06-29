import io
import uuid
import json
import logging
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from PIL import Image

from app.config import settings
from app.schemas import SearchResponse, IndexResponse, SearchResultItem
from app.services.embedding import embedding_service
from app.services.vector_db import vector_db_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Image Drive ML Backend...")
    # Pre-load embedding model to memory
    logger.info(f"Loading embedding model: {settings.MODEL_NAME}...")
    _ = embedding_service.model
    _ = embedding_service.processor
    logger.info("Embedding model loaded successfully.")

    # Initialize Qdrant collection
    logger.info("Initializing Qdrant collection...")
    try:
        vector_db_service.init_collection()
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant collection during startup: {e}")
    
    yield
    logger.info("Shutting down Image Drive ML Backend...")

app = FastAPI(
    title="Image Drive ML Backend",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"}

@app.post("/api/v1/index", response_model=IndexResponse)
async def index_image(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image."
        )

    # Parse metadata if provided
    meta_payload = {}
    if metadata:
        try:
            meta_payload = json.loads(metadata)
            if not isinstance(meta_payload, dict):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Metadata must be a JSON object."
                )
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Metadata is not valid JSON."
            )

    try:
        # Read image content
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Generate UUID and file paths
        image_id = str(uuid.uuid4())
        # Use file extension or default to .jpg
        ext = file.filename.split(".")[-1] if file.filename and "." in file.filename else "jpg"
        filename = f"{image_id}.{ext}"
        
        # Save image locally
        local_path = settings.upload_path / filename
        with open(local_path, "wb") as f:
            f.write(contents)

        # Extract CLIP embedding
        embedding = embedding_service.extract_embedding(image)

        # Build Qdrant payload
        # Standard schema requirements: file_path and added_at
        # Make the stored file_path look clean (e.g. data/uploads/uuid.jpg)
        relative_file_path = str(Path(settings.UPLOAD_DIR) / filename).replace("\\", "/")
        payload = {
            "file_path": relative_file_path,
            "added_at": datetime.utcnow().isoformat() + "Z",
            **meta_payload
        }

        # Index vector in Qdrant
        vector_db_service.upsert_vector(
            point_id=image_id,
            vector=embedding,
            payload=payload
        )

        return IndexResponse(
            status="success",
            message="Image indexed successfully",
            id=image_id
        )

    except Exception as e:
        logger.error(f"Error indexing image: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while indexing the image: {str(e)}"
        )

@app.post("/api/v1/search", response_model=SearchResponse)
async def search_similar_images(
    file: UploadFile = File(...),
    limit: int = Form(10)
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image."
        )

    try:
        # Read image content
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Extract embedding
        embedding = embedding_service.extract_embedding(image)

        # Query Qdrant
        search_results = vector_db_service.search_similar(embedding, limit=limit)

        # Map results to schemas
        results = []
        for res in search_results:
            results.append(
                SearchResultItem(
                    id=str(res.id),
                    score=res.score,
                    payload=res.payload
                )
            )

        return SearchResponse(
            status="success",
            results=results
        )

    except Exception as e:
        logger.error(f"Error searching similar images: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during search: {str(e)}"
        )
