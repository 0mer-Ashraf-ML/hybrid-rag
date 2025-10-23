from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None

class RetrievedDoc(BaseModel):
    id: str
    text: str
    score: float
    meta: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: Optional[str] = None
    retrieved: List[RetrievedDoc]
