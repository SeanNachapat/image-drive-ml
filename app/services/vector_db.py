import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import settings

logger = logging.getLogger(__name__)

class VectorDBService:
    def __init__(self):
        self._client = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        return self._client

    def init_collection(self):
        """
        Initializes the target collection in Qdrant if it doesn't exist.
        """
        collection_name = settings.QDRANT_COLLECTION
        try:
            # Check if collection exists
            exists = self.client.collection_exists(collection_name)
            if not exists:
                logger.info(f"Collection '{collection_name}' not found. Creating it...")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
                )
                logger.info(f"Collection '{collection_name}' created successfully.")
            else:
                logger.info(f"Collection '{collection_name}' already exists.")
        except Exception as e:
            logger.error(f"Error checking/creating Qdrant collection: {e}")
            raise e

    def upsert_vector(self, point_id: str, vector: list[float], payload: dict):
        """
        Upserts a vector embedding and its associated payload into the collection.
        """
        self.client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )

    def search_similar(self, vector: list[float], limit: int = 10):
        """
        Searches for the nearest vectors using cosine similarity.
        """
        search_result = self.client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=vector,
            limit=limit
        )
        return search_result

# Singleton instance
vector_db_service = VectorDBService()
