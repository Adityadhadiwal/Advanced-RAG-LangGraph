from typing import List, TypedDict, Optional, Dict, Any

class GraphState(TypedDict):
    
    question: str
    solution: str
    online_search: bool
    documents: List[str]
    search_method: Optional[str]  # 'documents' or 'online'
    document_evaluations: Optional[List[Dict[str, Any]]]  # Store document evaluation results
    document_relevance_score: Optional[Dict[str, Any]]  # Store document relevance check
    question_relevance_score: Optional[Dict[str, Any]]  # Store question relevance check