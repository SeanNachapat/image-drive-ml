# Product Specification: Image Drive ML Backend

## 1. Overview
The `image-drive-ml` service is the core machine learning and vector retrieval backend for the Image Drive application. It enables Content-Based Image Retrieval (CBIR), allowing users to upload a reference image and instantly retrieve visually or conceptually similar images from a central database without relying on text tags or metadata.

## 2. System Architecture
The system follows a modular, microservice-based architecture to separate computational heavy lifting from fast similarity search.

* **API Layer:** FastAPI (Python 3.11+)
* **Machine Learning Engine:** PyTorch & Hugging Face Transformers
* **Vision Model:** `openai/clip-vit-base-patch32` (or equivalent SigLIP model)
* **Vector Database:** Qdrant (via Docker)

## 3. Core Workflows

### 3.1. Ingestion Pipeline (Batch Processing)
* **Objective:** Populate the vector database with reference images.
* **Process:**
    1.  Read raw images from local storage or cloud buckets.
    2.  Pre-process images (resize, crop, normalize).
    3.  Extract a 512-dimensional vector embedding using the CLIP model.
    4.  Upsert the vector and a payload (e.g., file path, metadata) into the Qdrant database.

### 3.2. Inference Pipeline (Real-time Search)
* **Objective:** Handle user queries and return similar images.
* **Process:**
    1.  Accept a binary image file via the API.
    2.  Extract the vector embedding using the same CLIP model.
    3.  Query Qdrant using Cosine Similarity to find the nearest neighbors.
    4.  Return the top *N* matches with their similarity scores.

## 4. API Specification (Draft)

### `POST /api/v1/search`
Retrieves similar images based on an uploaded reference image.
* **Request Type:** `multipart/form-data`
* **Parameters:**
    * `file` (File): The reference image.
    * `limit` (Integer, Optional): Number of results to return. Default: 10.
* **Response (JSON):**
    ```json
    {
      "status": "success",
      "results": [
        {
          "id": "uuid-1234",
          "score": 0.923,
          "payload": {
            "file_path": "images/city_night.jpg"
          }
        }
      ]
    }
    ```

### `POST /api/v1/index`
Indexes a single new image into the vector database.
* **Request Type:** `multipart/form-data`
* **Parameters:**
    * `file` (File): The image to index.
    * `metadata` (JSON): Optional metadata (e.g., origin URL).
* **Response (JSON):**
    ```json
    {
      "status": "success",
      "message": "Image indexed successfully",
      "id": "uuid-5678"
    }
    ```

## 5. Data Models & Vector Schema

* **Vector Dimension:** 512 (Standard for CLIP ViT-B/32)
* **Distance Metric:** Cosine Similarity
* **Payload Schema:**
    * `file_path` (String): URI or path to the original image.
    * `added_at` (Timestamp): Time of indexing.

## 6. Infrastructure & Deployment
* **Environment:** Dockerized deployments for both the API and Qdrant.
* **Resource Requirements (Minimum):**
    * RAM: 4GB+ (for model weights in memory)
    * CPU: Multi-core CPU recommended for inference if GPU is unavailable.
    * Storage: Depends on Qdrant payload size, SSD recommended for fast HNSW traversal.
