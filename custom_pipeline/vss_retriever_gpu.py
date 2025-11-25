import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import time
import os

# Try importing OpenAI, handle if not present
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class VSSRetriever:
    """
    Vector Similarity Search Retriever for knowledge base entity retrieval.
    Optimized to prevent double-loading of embeddings.
    """
    
    def __init__(
        self,
        kb,
        emb_base_path: str,
        emb_model: str = "text-embedding-ada-002",
        qa_dataset: Optional[Any] = None,
        dataset_name: str = "test",
        use_vss: bool = True, # Kept for signature compatibility, but ignored
        use_embedding_cache: bool = True,
        use_gpu: bool = True
    ):
        self.kb = kb
        self.emb_base_path = Path(emb_base_path)
        self.emb_model = emb_model
        self.qa_dataset = qa_dataset
        self.dataset_name = dataset_name
        self.use_embedding_cache = use_embedding_cache
        
        # Setup device
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"[VSSRetriever] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print(f"[VSSRetriever] Using CPU")
        
        self.embedding_cache = {} if use_embedding_cache else None
        
        # Load embeddings (Optimized)
        t0 = time.time()
        self._load_embeddings()
        print(f"[VSSRetriever] Embeddings loaded in {time.time() - t0:.2f}s")
        
        self.node_ids_by_type = self._load_node_ids_by_type()
        
        # REMOVED: self.vss = VSS(...) 
        # This was causing the double-loading freeze.
    
    def _load_embeddings(self) -> None:
        """Load all embedding dictionaries from disk and move to device."""
        emb_dir = self.emb_base_path / self.emb_model
        
        # 1. Load entity embeddings
        entity_emb_path = emb_dir / "entities" / "entity_emb_dict.pt"
        if entity_emb_path.exists():
            # print(f"[VSSRetriever] Loading entities from {entity_emb_path.name}...")
            self.entity_emb_dict = torch.load(entity_emb_path, map_location=self.device)
        else:
            self.entity_emb_dict = {}
        
        # 2. Load query embeddings
        query_emb_path = emb_dir / "query" / "query_emb_dict.pt"
        if query_emb_path.exists():
            # print(f"[VSSRetriever] Loading queries from {query_emb_path.name}...")
            self.query_emb_dict = torch.load(query_emb_path, map_location=self.device)
        else:
            self.query_emb_dict = {}
        
        # 3. Load candidate embeddings
        candidate_emb_path = emb_dir / "doc" / "candidate_emb_dict.pt"
        if candidate_emb_path.exists():
            # print(f"[VSSRetriever] Loading candidates from {candidate_emb_path.name}...")
            self.candidate_emb_dict = torch.load(candidate_emb_path, map_location=self.device)
        else:
            self.candidate_emb_dict = {}
        
        # 4. Load node embeddings by type
        self.node_emb_dict = self._load_node_embeddings()
    
    def _load_node_embeddings(self) -> Dict[str, torch.Tensor]:
        """Load node embeddings for all node types and move to device."""
        nodes_emb_dir = self.emb_base_path / self.emb_model / "nodes"
        node_emb_dict = {}
        
        files = list(nodes_emb_dir.glob("*.pt"))
        for file in files:
            node_type = file.stem
            type_name = "_".join(node_type.split("_")[:-1])
            
            if type_name == "geneprotein":
                type_name = "gene/protein"
            elif type_name == "effectphenotype":
                type_name = "effect/phenotype"
            
            # Optimization: Load directly to device
            embeddings = torch.load(file, map_location=self.device)
            node_emb_dict[type_name] = embeddings
        
        return node_emb_dict
    
    def _load_node_ids_by_type(self) -> Dict[str, List[int]]:
        node_ids_by_type = {}
        for n_type in self.kb.node_type_lst():
            node_ids_by_type[n_type] = self.kb.get_node_ids_by_type(n_type)
        return node_ids_by_type

    def _fetch_openai_embedding(self, text: str) -> Optional[torch.Tensor]:
        """Fetch embedding directly from OpenAI API to avoid loading the VSS class."""
        if not OPENAI_AVAILABLE:
            print("[Error] OpenAI library not installed. Cannot generate new embeddings.")
            return None
            
        text = text.replace("\n", " ")
        try:
            # Try new OpenAI client (v1.0+)
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.embeddings.create(input=[text], model=self.emb_model)
            embedding = response.data[0].embedding
            return torch.tensor(embedding, device=self.device)
        except Exception as e:
            print(f"[VSSRetriever] Error generating embedding: {e}")
            return None

    def get_query_emb(
        self, 
        query: str, 
        query_id: Optional[Any] = None,
        use_cache: bool = True
    ) -> Optional[torch.Tensor]:
        
        # Check pre-loaded embeddings first
        if query in self.entity_emb_dict:
            return self.entity_emb_dict[query]
        elif query_id is not None and query_id in self.query_emb_dict:
            return self.query_emb_dict[query_id]
        
        # Check cache
        if use_cache and self.use_embedding_cache and self.embedding_cache is not None:
            cache_key = f"{query}_{query_id}" if query_id else query
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]
        
        # Generate new embedding using INTERNAL method (No VSS class)
        embedding = self._fetch_openai_embedding(query)
            
        if use_cache and self.use_embedding_cache and self.embedding_cache is not None and embedding is not None:
            cache_key = f"{query}_{query_id}" if query_id else query
            self.embedding_cache[cache_key] = embedding
            
        return embedding

    # ... (Rest of the methods: cache_embedding, get_cached_embedding, clear_cache, etc. remain UNCHANGED) ...

    def cache_embedding(self, query: str, embedding: torch.Tensor, query_id: Optional[Any] = None):
        if self.use_embedding_cache and self.embedding_cache is not None:
            cache_key = f"{query}_{query_id}" if query_id else query
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.to(self.device)
            self.embedding_cache[cache_key] = embedding
    
    def get_cached_embedding(self, query: str, query_id: Optional[Any] = None) -> Optional[torch.Tensor]:
        if not self.use_embedding_cache or self.embedding_cache is None:
            return None
        cache_key = f"{query}_{query_id}" if query_id else query
        return self.embedding_cache.get(cache_key)
    
    def clear_cache(self):
        if self.embedding_cache is not None:
            self.embedding_cache.clear()
    
    def get_cache_size(self) -> int:
        return len(self.embedding_cache) if self.embedding_cache is not None else 0
    
    def compute_similarities(
        self,
        query_emb: torch.Tensor,
        node_type: str,
        node_id_mask: Optional[List[int]] = None,
        node_ids_to_exclude: List[int] = []
    ) -> Dict[int, float]:
        
        if isinstance(query_emb, torch.Tensor):
            query_emb = query_emb.to(self.device)
        
        similarity = torch.matmul(self.node_emb_dict[node_type], query_emb.T)
        
        node_ids = self.kb.get_node_ids_by_type(node_type)
        
        similarity_cpu = similarity.cpu()
        score_dict = {node_ids[i]: similarity_cpu[i].item() for i in range(len(similarity_cpu))}
        
        if node_id_mask is not None:
            filtered_score_dict = {}
            for node_id in node_id_mask:
                if node_id in score_dict:
                    filtered_score_dict[node_id] = score_dict[node_id]
            score_dict = filtered_score_dict
        
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
        
        if cutoff is None:
            cutoff = 0.0
        
        query_emb = self.get_query_emb(search_str, query_id, use_cache=True)
        if query_emb is None:
            msg = f"VSS: No embedding found for query '{search_str}'. Returning empty list."
            if logger is not None: logger.log(msg)
            else: print(msg)
            return [], []
        
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
        
        if cutoff > 0.0:
            filtered_dict = {}
            for key in score_dict:
                if score_dict[key] >= cutoff:
                    filtered_dict[key] = score_dict[key]
            score_dict = filtered_dict
        
        node_scores = list(score_dict.values())
        if not node_scores:
             return [], []

        scores_tensor = torch.FloatTensor(node_scores).to(self.device)
        top_k_idx = torch.topk(
            scores_tensor,
            min(k, len(node_scores)),
            dim=-1,
            largest=True,
            sorted=True
        ).indices.cpu().tolist()
        
        vss_scores = torch.tensor(node_scores)[top_k_idx].tolist()
        keys_tensor = torch.tensor(list(score_dict.keys()), dtype=torch.long)
        top_k_node_ids = keys_tensor[top_k_idx].tolist()
        
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
        
        return top_k_node_ids, vss_scores
    
    def get_available_node_types(self) -> List[str]:
        return list(self.node_emb_dict.keys())
    
    def get_node_count_by_type(self, node_type: str) -> int:
        return len(self.node_ids_by_type.get(node_type, []))
    
    def get_device_info(self) -> str:
        if self.device.type == 'cuda':
            return f"GPU: {torch.cuda.get_device_name(0)} (Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)"
        else:
            return "CPU"