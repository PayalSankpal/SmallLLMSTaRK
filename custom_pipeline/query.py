
from typing import List, Dict, Any
class Query:
    """
    A class to hold the attributes of a query and its processing state.
    """    
    def __init__(
        self, 
        id: Any, 
        query: str, 
        ground_truths: List[Any],
        status: str = "IN_PROGRESS",
        prompt: str = "",
        entities: Dict[str, Any] = None,
        symbol_candidates: Dict[str, Any] = None,
        relations : Dict[tuple, Any] = None ,
    ):
        self.id = id
        self.query = query
        self.ground_truths = ground_truths
        self.status = status
        self.entity_id_response = ""
        self.relations_id_response = ""
        self.relations = relations        
        # Use None as default for mutable types (dicts/lists)
        # and initialize to empty dict if None is passed.
        self.entities = entities if entities is not None else {}
        self.initial_symbol_candidates = symbol_candidates if symbol_candidates is not None else {}
        self.relations = entities if entities is not None else {}
        self.results = {}

    def __repr__(self) -> str:
        """Provides a clean string representation for the object."""
        return (f"Query(id={self.id!r}, query={self.query!r}, "
                f"status={self.status!r}, ground_truths={self.ground_truths!r})")