import os
import argparse
import uuid
import shutil
import logging
from datetime import datetime
from pathlib import Path
from PIL import Image

# Setup sys.path to allow imports from app
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.services.embedding import embedding_service
from app.services.vector_db import vector_db_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ingest")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def ingest_directory(input_dir: str):
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory '{input_dir}' does not exist.")
        return

    logger.info("Initializing services...")
    # Initialize Qdrant collection
    vector_db_service.init_collection()
    
    # Warm up embedding model
    _ = embedding_service.model

    logger.info(f"Scanning '{input_dir}' for images...")
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))

    # Convert to set to deduplicate in case of duplicate matches
    image_files = list(set(image_files))

    if not image_files:
        logger.warning("No images found in the specified directory.")
        return

    logger.info(f"Found {len(image_files)} images to ingest.")
    success_count = 0

    for img_path in image_files:
        try:
            logger.info(f"Processing {img_path.name}...")
            
            # Read image
            image = Image.open(img_path)
            
            # Generate UUID and new filename
            image_id = str(uuid.uuid4())
            ext = img_path.suffix.lower()
            new_filename = f"{image_id}{ext}"
            
            # Save a copy in UPLOAD_DIR
            dest_path = settings.upload_path / new_filename
            shutil.copy2(img_path, dest_path)
            
            # Extract embedding
            embedding = embedding_service.extract_embedding(image)
            
            # Prepare payload
            relative_file_path = str(Path(settings.UPLOAD_DIR) / new_filename).replace("\\", "/")
            payload = {
                "file_path": relative_file_path,
                "added_at": datetime.utcnow().isoformat() + "Z",
                "original_filename": img_path.name
            }
            
            # Upsert into Qdrant
            vector_db_service.upsert_vector(
                point_id=image_id,
                vector=embedding,
                payload=payload
            )
            
            success_count += 1
            logger.info(f"Indexed successfully: {img_path.name} -> ID: {image_id}")
        except Exception as e:
            logger.error(f"Failed to process {img_path.name}: {e}", exc_info=True)

    logger.info(f"Ingestion completed. Successfully indexed {success_count}/{len(image_files)} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch ingest images into the vector database.")
    parser.add_argument("--dir", required=True, help="Path to the directory containing images.")
    args = parser.parse_args()
    
    ingest_directory(args.dir)
