import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dotenv import load_dotenv
from openai import OpenAI
import os

# from vss import VSS


def load_emb_model(offline_mode: bool, model_name: str):
    if offline_mode:
        return None
    else:
        
        if model_name == "text-embedding-ada-002":
            load_dotenv()
            return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Invalid embedding model name: {model_name}.")

class VSSRetriever:
    """
    Vector Similarity Search Retriever for knowledge base entity retrieval.
    Encapsulates embedding loading and similarity-based node retrieval functionality.
    """
    
    def __init__(
        self,
        kb,
        emb_base_path: str,
        emb_model: str = "text-embedding-ada-002",
        qa_dataset: Optional[Any] = None,
        dataset_name: str = "test",
        use_vss: bool = True,
        use_embedding_cache: bool = True,  # Enable embedding cache
        use_gpu: bool = True  # NEW: Enable GPU if available
    ):
        """
        Initialize the VSSRetriever.
        
        Args:
            kb: Knowledge base object
            emb_base_path: Base path to embeddings directory (e.g., "./emb/prime")
            emb_model: Name of the embedding model (default: "text-embedding-ada-002")
            qa_dataset: Optional QA dataset
            dataset_name: Name of the dataset (default: "test")
            use_vss: Whether to initialize VSS object (default: False)
            use_embedding_cache: Whether to cache generated embeddings (default: True)
            use_gpu: Whether to use GPU if available (default: True)
        """
        self.kb = kb
        self.emb_base_path = Path(emb_base_path)
        self.emb_model = emb_model
        self.qa_dataset = qa_dataset
        self.dataset_name = dataset_name
        self.use_embedding_cache = use_embedding_cache
        offline_mode = False
        self.emb_client = load_emb_model(offline_mode, emb_model)
        
        # NEW: Setup device (GPU if available and requested, else CPU)
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"[VSSRetriever] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print(f"[VSSRetriever] Using CPU")
        
        # NEW: Embedding cache for runtime-generated embeddings
        self.embedding_cache = {} if use_embedding_cache else None
        
        # Load embeddings
        self._load_embeddings()
        
        # Get node IDs by type
        self.node_ids_by_type = self._load_node_ids_by_type()
        
        # Initialize VSS if needed
        self.vss = None

    
    def _load_embeddings(self) -> None:
        """Load all embedding dictionaries from disk and move to device."""
        emb_dir = self.emb_base_path / self.emb_model
        
        # Load entity embeddings
        entity_emb_path = emb_dir / "entities" / "entity_emb_dict.pt"
        if entity_emb_path.exists():
            self.entity_emb_dict = torch.load(entity_emb_path, map_location=self.device)
            # Ensure all tensors are on the correct device
            self.entity_emb_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                    for k, v in self.entity_emb_dict.items()}
        else:
            self.entity_emb_dict = {}
        
        # Load query embeddings
        query_emb_path = emb_dir / "query" / "query_emb_dict.pt"
        if query_emb_path.exists():
            self.query_emb_dict = torch.load(query_emb_path, map_location=self.device)
            self.query_emb_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                   for k, v in self.query_emb_dict.items()}
        else:
            self.query_emb_dict = {}
        
        # Load candidate embeddings
        self.candidate_emb_dict = {}
        
        # Load node embeddings by type
        self.node_emb_dict = self._load_node_embeddings()
    
    def _load_node_embeddings(self) -> Dict[str, torch.Tensor]:
        """Load node embeddings for all node types and move to device."""
        nodes_emb_dir = self.emb_base_path / self.emb_model / "nodes"
        node_emb_dict = {}
        
        for file in nodes_emb_dir.glob("*.pt"):
            node_type = file.stem
            type_name = "_".join(node_type.split("_")[:-1])
            
            # Handle special type names
            if type_name == "geneprotein":
                type_name = "gene/protein"
            elif type_name == "effectphenotype":
                type_name = "effect/phenotype"
            
            # Load and move to device
            embeddings = torch.load(file, map_location=self.device)
            node_emb_dict[type_name] = embeddings.to(self.device)
        
        return node_emb_dict
    
    def _load_node_ids_by_type(self) -> Dict[str, List[int]]:
        """Get node IDs organized by type from the knowledge base."""
        node_ids_by_type = {}
        for n_type in self.kb.node_type_lst():
            node_ids_by_type[n_type] = self.kb.get_node_ids_by_type(n_type)
        return node_ids_by_type
    
    def get_openai_embedding(self, query: str, model: str):
        """Get embedding using OpenAI client."""
        print(f"Getting embedding for query: {query} using model: {model}")
        emb = self.emb_client.embeddings.create(input=query, model=model, timeout=120)
        return torch.FloatTensor(emb.data[0].embedding)



    def get_query_emb(
        self, 
        query: str, 
        query_id: Optional[Any] = None,
        use_cache: bool = True  # Control cache usage per call
    ) -> Optional[torch.Tensor]:
        """
        Get embedding for a query string.
        
        Args:
            query: Query string
            query_id: Optional query ID to look up in query_emb_dict
            use_cache: Whether to use/store in cache (default: True)
            
        Returns:
            Query embedding tensor or None if not found (on correct device)
        """
        # Check pre-loaded embeddings first
        if query in self.entity_emb_dict:
            emb = self.entity_emb_dict[query]
            return emb.to(self.device) if isinstance(emb, torch.Tensor) else emb
        elif query_id is not None and query_id in self.query_emb_dict:
            emb = self.query_emb_dict[query_id]
            return emb.to(self.device) if isinstance(emb, torch.Tensor) else emb
        
        # Check cache if enabled
        if use_cache and self.use_embedding_cache and self.embedding_cache is not None:
            cache_key = f"{query}_{query_id}" if query_id else query
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]
        
        # Generate new embedding via VSS
        embedding = self.get_openai_embedding(query, model=self.emb_model)
        
        # Move to device if tensor
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.to(self.device)
        
        # Store in cache if enabled
        if use_cache and self.use_embedding_cache and self.embedding_cache is not None and embedding is not None:
            cache_key = f"{query}_{query_id}" if query_id else query
            self.embedding_cache[cache_key] = embedding
        
        return embedding
            
    def cache_embedding(self, query: str, embedding: torch.Tensor, query_id: Optional[Any] = None):
        """
        Manually cache an embedding.
        
        Args:
            query: Query string
            embedding: Embedding tensor to cache
            query_id: Optional query ID
        """
        if self.use_embedding_cache and self.embedding_cache is not None:
            cache_key = f"{query}_{query_id}" if query_id else query
            # Ensure embedding is on correct device before caching
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.to(self.device)
            self.embedding_cache[cache_key] = embedding
    
    def get_cached_embedding(self, query: str, query_id: Optional[Any] = None) -> Optional[torch.Tensor]:
        """
        Retrieve a cached embedding without generating a new one.
        
        Args:
            query: Query string
            query_id: Optional query ID
            
        Returns:
            Cached embedding tensor or None if not found
        """
        if not self.use_embedding_cache or self.embedding_cache is None:
            return None
        
        cache_key = f"{query}_{query_id}" if query_id else query
        return self.embedding_cache.get(cache_key)
    
    def clear_cache(self):
        """Clear the embedding cache."""
        if self.embedding_cache is not None:
            self.embedding_cache.clear()
    
    def get_cache_size(self) -> int:
        """Get the number of cached embeddings."""
        return len(self.embedding_cache) if self.embedding_cache is not None else 0
    
    def compute_similarities(
        self,
        query_emb: torch.Tensor,
        node_type: str,
        node_id_mask: Optional[List[int]] = None,
        node_ids_to_exclude: List[int] = []
    ) -> Dict[int, float]:
        """
        Compute similarity scores between query embedding and nodes of a specific type.
        Uses GPU acceleration if available.
        
        Args:
            query_emb: Query embedding tensor
            node_type: Type of nodes to compute similarity with
            node_id_mask: Optional list of node IDs to restrict search to
            node_ids_to_exclude: List of node IDs to exclude from results
            
        Returns:
            Dictionary mapping node IDs to similarity scores
        """
        # Ensure query embedding is on correct device
        if isinstance(query_emb, torch.Tensor):
            query_emb = query_emb.to(self.device)
        
        # Compute similarity on GPU/CPU
        similarity = torch.matmul(self.node_emb_dict[node_type], query_emb.T)
        
        node_ids = self.kb.get_node_ids_by_type(node_type)
        
        # Move results to CPU for dictionary creation (more efficient)
        similarity_cpu = similarity.cpu()
        score_dict = {node_ids[i]: similarity_cpu[i].item() for i in range(len(similarity_cpu))}
        
        # Apply node ID mask filter
        if node_id_mask is not None:
            filtered_score_dict = {}
            for node_id in node_id_mask:
                if node_id in score_dict:
                    filtered_score_dict[node_id] = score_dict[node_id]
            score_dict = filtered_score_dict
        
        # Exclude specified node IDs
        if len(node_ids_to_exclude) > 0:
            for node_id in node_ids_to_exclude:
                if node_id in score_dict:
                    score_dict.pop(node_id)
        
        return score_dict
    
    def get_top_k_nodes(
        self,
        search_str: str,
        k: int,
        node_type: Optional[str] = None,
        logger: Optional[Any] = None,
        node_id_mask: Optional[List[int]] = None,
        complement_with_non_masked_ids: bool = False,
        query_id: Optional[Any] = None,
        node_ids_to_exclude: List[int] = [],
        node_types_to_consider: List[str] = [],
        cutoff: float = 0.0
    ) -> Tuple[List[int], List[float]]:
        """
        Retrieve top-k most similar nodes for a given search string.
        Uses GPU acceleration if available.
        
        Args:
            search_str: Search query string
            k: Number of top results to return
            node_type: Specific node type to search (if None, searches node_types_to_consider)
            logger: Optional logger object
            node_id_mask: Optional list of node IDs to restrict search to
            complement_with_non_masked_ids: Whether to add non-masked nodes if k not reached
            query_id: Optional query ID for embedding lookup
            node_ids_to_exclude: List of node IDs to exclude from results
            node_types_to_consider: List of node types to search if node_type is None
            cutoff: Minimum similarity score threshold (default: 0.0)
            
        Returns:
            Tuple of (top_k_node_ids, vss_scores)
        """
        if cutoff is None:
            cutoff = 0.0
        
        # Get query embedding (will use cache if available)
        query_emb = self.get_query_emb(search_str, query_id, use_cache=True)
        if query_emb is None:
            msg = f"VSS: No embedding found for query '{search_str}'. Returning empty list."
            if logger is not None:
                logger.log(msg)
            else:
                print(msg)
            return [], []
        
        # Compute similarities
        if node_type is None:
            score_dict = {}
            for n_type in node_types_to_consider:
                score_dict.update(
                    self.compute_similarities(
                        query_emb=query_emb,
                        node_type=n_type,
                        node_id_mask=node_id_mask,
                        node_ids_to_exclude=node_ids_to_exclude
                    )
                )
        else:
            score_dict = self.compute_similarities(
                query_emb=query_emb,
                node_type=node_type,
                node_id_mask=node_id_mask,
                node_ids_to_exclude=node_ids_to_exclude
            )
        
        # Apply cutoff filter
        if cutoff > 0.0:
            filtered_dict = {}
            for key in score_dict:
                if score_dict[key] >= cutoff:
                    filtered_dict[key] = score_dict[key]
            score_dict = filtered_dict
        
        # Get top k node IDs based on similarity
        node_scores = list(score_dict.values())
        
        # Use GPU for topk if beneficial (move to device, compute, move back)
        scores_tensor = torch.FloatTensor(node_scores).to(self.device)
        top_k_idx = torch.topk(
            scores_tensor,
            min(k, len(node_scores)),
            dim=-1,
            largest=True,
            sorted=True
        ).indices.cpu().tolist()  # Move back to CPU for list conversion
        
        # Get scores and node IDs
        vss_scores = torch.tensor(node_scores)[top_k_idx].tolist()
        keys_tensor = torch.tensor(list(score_dict.keys()), dtype=torch.long)
        top_k_node_ids = keys_tensor[top_k_idx].tolist()
        
        # Complement with non-masked IDs if needed
        if complement_with_non_masked_ids and node_id_mask is not None and len(node_id_mask) < k:
            additional_node_ids, additional_vss_scores = self.get_top_k_nodes(
                search_str,
                k - len(node_id_mask),
                node_type,
                logger=logger,
                complement_with_non_masked_ids=False,
                query_id=query_id,
                node_ids_to_exclude=top_k_node_ids,
                cutoff=cutoff,
                node_types_to_consider=node_types_to_consider
            )
            
            top_k_node_ids += additional_node_ids
            vss_scores += additional_vss_scores
            
            msg = f"VSS: Added further answers to candidate list. New list (top10): {top_k_node_ids[:10]}"
            if logger is not None:
                logger.log(msg)
            else:
                print(msg)
        
        return top_k_node_ids, vss_scores
    
    def get_available_node_types(self) -> List[str]:
        """Get list of available node types in embeddings."""
        return list(self.node_emb_dict.keys())
    
    def get_node_count_by_type(self, node_type: str) -> int:
        """Get count of nodes for a specific type."""
        return len(self.node_ids_by_type.get(node_type, []))
    
    def get_device_info(self) -> str:
        """Get information about the device being used."""
        if self.device.type == 'cuda':
            return f"GPU: {torch.cuda.get_device_name(0)} (Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)"
        else:
            return "CPU"

