from pydantic import BaseModel
from typing import List, Dict, Any

class SearchResultItem(BaseModel):
    id: str
    score: float
    payload: Dict[str, Any]

class SearchResponse(BaseModel):
    status: str = "success"
    results: List[SearchResultItem]

class IndexResponse(BaseModel):
    status: str = "success"
    message: str = "Image indexed successfully"
    id: str
