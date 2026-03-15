import heapq
from typing import Dict, Set, List, Tuple, TYPE_CHECKING, Optional
from collections import defaultdict
import math

# Import CandidateContext
if TYPE_CHECKING:
    from custom_pipeline.candidate_context import CandidateContext
else:
    try:
        from .candidate_context import CandidateContext
    except ImportError:
        from custom_pipeline.candidate_context import CandidateContext

class PriorityQueueGrounding:
    """
    Optimized priority queue-based grounding.
    """
    
    def __init__(
        self,
        query_obj,
        kb,
        vss_retriever,
        max_candidates_per_symbol: int = 100, # Defaulted to 100 per request
        max_answer_candidates: int = 100,
        top_k_neighbors: int = 10,
        score_decay: float = 0.9,
        support_boost: float = 0.15,
        verbose: bool = True
    ):
        self.query_obj = query_obj
        self.kb = kb
        self.vss_retriever = vss_retriever
        self.max_candidates_per_symbol = max_candidates_per_symbol
        self.max_answer_candidates = max_answer_candidates
        self.top_k_neighbors = top_k_neighbors
        self.support_boost = support_boost
        self.score_decay = score_decay
        self.verbose = verbose
        
        self.candidate_queues = {}
        self.processed_nodes = defaultdict(set)
        self.all_candidates: Dict[str, Dict[int, CandidateContext]] = defaultdict(dict)
        self.candidate_count = defaultdict(int)
        
        self.entity_query_embeddings = {}
        
        # OPTIMIZATION: Nested dict is faster than Tuple key creation
        # Structure: {target_entity: {node_id: score}}
        self.vss_score_cache = defaultdict(dict) 
        
    def _build_entity_query_string(self, entity: str) -> str:
        entity_data = self.query_obj.entities.get(entity, {})
        parts = []
        lexical = entity_data.get('lexical', {})
        if 'name' in lexical:
            parts.append(lexical['name']) 
        semantic = entity_data.get('semantic', [])
        if semantic:
            parts.extend(semantic)
        return " ".join(parts)
    
    def _initialize_queues(self):
        if self.verbose:
            print("\n[INIT] Initializing priority queues")
        
        # Pre-compute embeddings
        for entity in self.query_obj.entities.keys():
            query_str = self._build_entity_query_string(entity)
            if query_str:
                self.entity_query_embeddings[entity] = self.vss_retriever.get_query_emb(query_str)
            else:
                self.entity_query_embeddings[entity] = None
        
        # Initialize candidates
        for entity, cand_list in self.query_obj.initial_symbol_candidates.items():
            # OPTIMIZATION: Build list then heapify (O(N)) instead of pushing one by one (O(N log N))
            raw_heap = []
            entity_candidates = self.all_candidates[entity]
            
            for cand in cand_list:
                entity_candidates[cand.node_id] = cand
                # Push to list: (-score, node_id)
                raw_heap.append((-cand.score, cand.node_id))
            
            heapq.heapify(raw_heap)
            self.candidate_queues[entity] = raw_heap
            self.candidate_count[entity] += len(cand_list)
            
            if self.verbose:
                print(f"  Initialized '{entity}' with {len(cand_list)} candidates")
        
        if 'ANSWER' not in self.candidate_queues:
            self.candidate_queues['ANSWER'] = []
    
    def _get_top_candidate(self, entity: str) -> Optional[Tuple[float, int, 'CandidateContext']]:
        queue = self.candidate_queues.get(entity)
        if not queue:
            return None
        
        # Local variable lookup speedup
        all_cands = self.all_candidates[entity]
        
        while queue:
            neg_score, node_id = heapq.heappop(queue)
            
            candidate = all_cands.get(node_id)
            
            # Lazy Deletion Check:
            # Using epsilon 1e-9 for float comparison stability
            if candidate and abs(candidate.score - (-neg_score)) < 1e-9:
                return (-neg_score, node_id, candidate)
                
        return None
    
    def _add_candidate(
        self,
        entity: str,
        node_id: int,
        score: float,
        source_entity: str,
        source_node: int,
        source_score: float,
        relation: str,
        target_entity: str
    ):
        max_limit = self.max_answer_candidates if entity == 'ANSWER' else self.max_candidates_per_symbol
        
        # O(1) Lookup
        existing = self.all_candidates[entity].get(node_id)
        
        triplet = (source_entity, relation, target_entity)
        
        if existing:
            # Update existing candidate
            if triplet not in existing.triplets:
                existing.add_triplet(triplet)
                existing.add_symbol_candidates(source_entity, source_node, source_score)
                
                old_score = existing.score
                existing.score = max(existing.score, source_score)
                
                # Apply support boost
                support_multiplier = 1 + (self.support_boost * math.log(existing.support + 1))
                new_score = existing.score * support_multiplier
                existing.score = new_score
                
                # Push new score if improved (Lazy Update)
                if new_score > old_score:
                    heapq.heappush(self.candidate_queues[entity], (-new_score, node_id))
            return True
            
        else:
            # Check limits only for NEW candidates
            if self.candidate_count[entity] >= max_limit:
                return False

            new_candidate = CandidateContext(
                node_id=node_id,
                entity=entity,
                score=score
            )
            new_candidate.add_triplet(triplet)
            new_candidate.add_symbol_candidates(source_entity, source_node, source_score)
            
            self.all_candidates[entity][node_id] = new_candidate
            self.candidate_count[entity] += 1
            
            heapq.heappush(self.candidate_queues[entity], (-score, node_id))
            return True

    def _score_neighbors_with_vss(
        self,
        neighbor_nodes: List[int],
        target_entity: str
    ) -> Dict[int, float]:
        if not neighbor_nodes:
            return {}
            
        nodes_to_compute = []
        cached_scores = {}
        
        # Optimization: Use nested dict lookup (faster than tuple key)
        # Target entity cache
        entity_cache = self.vss_score_cache[target_entity]
        
        for nid in neighbor_nodes:
            if nid in entity_cache:
                cached_scores[nid] = entity_cache[nid]
            else:
                nodes_to_compute.append(nid)
        
        if not nodes_to_compute:
            return cached_scores

        query_emb = self.entity_query_embeddings.get(target_entity)
        if query_emb is None:
            # Fallback
            for nid in nodes_to_compute:
                score = 0.5
                cached_scores[nid] = score
                entity_cache[nid] = score
            return cached_scores
        
        entity_data = self.query_obj.entities.get(target_entity, {})
        node_types = entity_data.get('type', [])
        
        computed_scores = {}
        if node_types:
            for node_type in node_types:
                if node_type in self.vss_retriever.node_emb_dict:
                    type_scores = self.vss_retriever.compute_similarities(
                        query_emb=query_emb,
                        node_type=node_type,
                        node_id_mask=nodes_to_compute
                    )
                    computed_scores.update(type_scores)
        
        # Update cache and result
        for nid in nodes_to_compute:
            score = computed_scores.get(nid, 0.5) 
            cached_scores[nid] = score
            entity_cache[nid] = score
            
        return cached_scores

    def _process_candidate(
        self,
        entity: str,
        candidate: 'CandidateContext',
        relations_to_process: List[Tuple[str, List[str]]]
    ) -> int:
        curr_node = candidate.node_id
        curr_score = candidate.score
        
        # Optimization: Access processed set directly via reference
        processed_set = self.processed_nodes[entity]
        if curr_node in processed_set:
            return 0
            
        processed_set.add(curr_node)
        
        added_count = 0
        
        # Pre-calc decay to avoid doing it inside the loop
        decayed_source = curr_score * self.score_decay
        
        for target_entity, relation_types in relations_to_process:
            # Optimization: Access processed set for target to filter early
            target_processed_set = self.processed_nodes[target_entity]
            
            for relation in relation_types:
                neighbors = self.kb.get_neighbor_nodes(curr_node, relation)
                if not neighbors:
                    continue
                
                # Filter neighbors before scoring to save VSS computation
                # Only score neighbors we haven't processed yet
                # Note: This is optional depending on if we allow re-visiting for score updates.
                # Keeping it standard for now.

                neighbor_scores = self._score_neighbors_with_vss(neighbors, target_entity)
                
                # Get Top-K
                sorted_neighbors = heapq.nlargest(
                    self.top_k_neighbors, 
                    neighbor_scores.items(), 
                    key=lambda x: x[1]
                )
                
                for neigh_node, vss_score in sorted_neighbors:
                    if neigh_node in target_processed_set:
                        continue
                    
                    # Optimized Score Propagation (Harmonic Mean)
                    if vss_score > 1e-6 and decayed_source > 1e-6:
                        # (2 * a * b) / (a + b)
                        propagated_score = (2 * vss_score * decayed_source) / (vss_score + decayed_source)
                    else:
                        propagated_score = 0.1

                    success = self._add_candidate(
                        entity=target_entity,
                        node_id=neigh_node,
                        score=propagated_score,
                        source_entity=entity,
                        source_node=curr_node,
                        source_score=curr_score,
                        relation=relation,
                        target_entity=target_entity 
                    )
                    
                    if success:
                        added_count += 1
                        
        return added_count
    
    def ground(self) -> Dict[str, List['CandidateContext']]:
        self._initialize_queues()
        
        entity_relations = defaultdict(list)
        if self.query_obj.relations:
            for (src, tgt), rel_types in self.query_obj.relations.items():
                entity_relations[src].append((tgt, rel_types))
                entity_relations[tgt].append((src, rel_types))
        
        entities_to_process = [e for e in self.query_obj.entities.keys() if e != 'ANSWER']
        
        iteration = 0
        while True:
            iteration += 1
            round_processed = 0
            
            # Optimization: Iterate over a copy so we can remove finished entities
            # This prevents looping over entities that are already full/done
            active_entities = []
            
            for entity in entities_to_process:
                # Limit Check
                if self.candidate_count[entity] >= self.max_candidates_per_symbol:
                    continue # Skip this entity, don't add to active
                
                top = self._get_top_candidate(entity)
                if not top:
                    continue # Empty queue, don't add to active
                    
                active_entities.append(entity)
                
                neg_score, node_id, candidate = top
                
                # If popped node was already processed, just loop again immediately to find next
                # simple 'continue' here wastes a round slot
                if node_id in self.processed_nodes[entity]:
                    continue
                
                added = self._process_candidate(
                    entity=entity,
                    candidate=candidate,
                    relations_to_process=entity_relations.get(entity, [])
                )
                round_processed += 1
            
            # Update the list to only include entities that still have work
            entities_to_process = active_entities

            if self.candidate_count['ANSWER'] >= self.max_answer_candidates:
                if self.verbose: print("[STOP] ANSWER max reached")
                break
            
            if round_processed == 0 and not entities_to_process:
                break
                
            if round_processed == 0:
                # Safety break if we have entities but made no progress
                break
                
        return {k: list(v.values()) for k, v in self.all_candidates.items()}

def run_priority_queue_grounding(
    query_obj, kb, vss_retriever,
    max_candidates_per_symbol: int = 100, # UPDATED DEFAULT
    max_answer_candidates: int = 100,
    top_k_neighbors: int = 10,
    support_boost: float = 0.15,
    score_decay: float = 0.9,
    verbose: bool = True
):
    grounder = PriorityQueueGrounding(
        query_obj, kb, vss_retriever,
        max_candidates_per_symbol, max_answer_candidates,
        top_k_neighbors, score_decay, support_boost, verbose
    )
    return grounder.ground()
