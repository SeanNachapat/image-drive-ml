import io
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Import the app while mocking lifecycle events to prevent downloading the large CLIP model
with patch("app.services.embedding.embedding_service.model", new=MagicMock()), \
     patch("app.services.embedding.embedding_service.processor", new=MagicMock()), \
     patch("app.services.vector_db.vector_db_service.init_collection", new=MagicMock()):
    from app.main import app

client = TestClient(app)

# Setup mock embedding and vector DB services for tests
@pytest.fixture
def mock_services():
    with patch("app.services.embedding.embedding_service.extract_embedding") as mock_extract, \
         patch("app.services.vector_db.vector_db_service.upsert_vector") as mock_upsert, \
         patch("app.services.vector_db.vector_db_service.search_similar") as mock_search, \
         patch("app.services.vector_db.vector_db_service.init_collection") as mock_init:
        
        mock_extract.return_value = [0.1] * 512
        yield mock_extract, mock_upsert, mock_search, mock_init

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "ok"

def test_index_image_success(mock_services, tmp_path):
    mock_extract, mock_upsert, mock_search, mock_init = mock_services
    
    # Override settings UPLOAD_DIR to a temporary directory for isolated testing
    from app.config import settings
    original_upload_dir = settings.UPLOAD_DIR
    settings.UPLOAD_DIR = str(tmp_path)
    # Re-verify/re-create upload path
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create a small dummy image in memory
        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        # Upload files and metadata
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        data = {"metadata": '{"category": "test-category"}'}

        response = client.post("/api/v1/index", files=files, data=data)
        
        assert response.status_code == 200
        json_data = response.json()
        assert json_data["status"] == "success"
        assert json_data["message"] == "Image indexed successfully"
        assert "id" in json_data
        
        # Verify services were called
        mock_extract.assert_called_once()
        mock_upsert.assert_called_once()
        
        # Verify upsert payload contains standard schema fields
        call_args = mock_upsert.call_args[1]
        payload = call_args["payload"]
        assert "file_path" in payload
        assert "added_at" in payload
        assert payload["category"] == "test-category"
        
    finally:
        settings.UPLOAD_DIR = original_upload_dir

def test_index_image_invalid_file_type(mock_services):
    files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    response = client.post("/api/v1/index", files=files)
    assert response.status_code == 400
    assert "detail" in response.json()

def test_index_image_invalid_metadata(mock_services):
    img = Image.new("RGB", (10, 10))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    
    files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
    data = {"metadata": "invalid-json"}
    
    response = client.post("/api/v1/index", files=files, data=data)
    assert response.status_code == 400
    assert "Metadata is not valid JSON" in response.json()["detail"]

def test_search_images_success(mock_services):
    mock_extract, mock_upsert, mock_search, mock_init = mock_services
    
    # Mock return value of search_similar
    mock_item = MagicMock()
    mock_item.id = "mock-uuid"
    mock_item.score = 0.95
    mock_item.payload = {"file_path": "data/uploads/mock-uuid.jpg"}
    mock_search.return_value = [mock_item]

    # Create dummy image
    img = Image.new("RGB", (100, 100), color="blue")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    files = {"file": ("search.jpg", img_bytes, "image/jpeg")}
    data = {"limit": "5"}

    response = client.post("/api/v1/search", files=files, data=data)
    
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "success"
    assert len(json_data["results"]) == 1
    
    result = json_data["results"][0]
    assert result["id"] == "mock-uuid"
    assert result["score"] == 0.95
    assert result["payload"]["file_path"] == "data/uploads/mock-uuid.jpg"
    
    mock_extract.assert_called_once()
    mock_search.assert_called_once_with(mock_extract.return_value, limit=5)
