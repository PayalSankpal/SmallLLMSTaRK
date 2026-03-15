import heapq
import math
from typing import Dict, Set, List, Tuple, TYPE_CHECKING
from collections import defaultdict

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
    Hybrid Grounding:
    - Structure: Uses global_candidates map and (-support, -score, node_id) heaps.
    - Logic: Uses Harmonic Mean scoring, VSS expansion, and Logarithmic Support Boost.
    """
    
    def __init__(
        self,
        query_obj,
        kb,
        vss_retriever,
        max_candidates_per_symbol: int = 1000,
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
        
        # STRUCTURE CHANGE: 
        # 1. Global Map for Singleton Contexts: {entity: {node_id: CandidateContext}}
        self.global_candidates = defaultdict(dict)
        
        # 2. Priority Queues now store: (-support, -score, node_id)
        self.priority_queues = defaultdict(list)
        
        # Tracking processed nodes to avoid re-expanding
        self.processed_nodes = defaultdict(set)
        
        # Entity query strings for VSS scoring
        self.entity_query_strings = {}
        self.entity_query_embeddings = {} 
        
    def _build_entity_query_string(self, entity: str) -> str:
        """Build query string from entity's lexical and semantic constraints."""
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
        """Initialize data structures using the global candidate map pattern."""
        if self.verbose:
            print("\n[INIT] Initializing priority queues...")
        
        # Build entity query strings and cache embeddings
        for entity in self.query_obj.entities.keys():
            query_str = self._build_entity_query_string(entity)
            self.entity_query_strings[entity] = query_str
            
            if query_str:
                self.entity_query_embeddings[entity] = self.vss_retriever.get_query_emb(query_str)
            else:
                self.entity_query_embeddings[entity] = None
            
            if self.verbose:
                print(f"  Entity '{entity}' query: '{query_str}'")
        
        # Initialize queues with initial candidates
        for entity, cand_list in self.query_obj.initial_symbol_candidates.items():
            for cand in cand_list:
                # 1. Store in Global Map (Singleton)
                self.global_candidates[entity][cand.node_id] = cand
                
                # 2. Push to Heap with new Tuple Format: (-support, -score, node_id)
                current_support = getattr(cand, 'support', 0)
                heapq.heappush(
                    self.priority_queues[entity],
                    (-current_support, -cand.score, cand.node_id)
                )
            
            if self.verbose:
                print(f"  Initialized '{entity}' with {len(cand_list)} candidates")
        
        # --- FIX STARTS HERE ---
        # Explicitly ensure 'ANSWER' exists in both structures.
        # This ensures the final return dictionary always has the 'ANSWER' key,
        # even if it is an empty list.
        if 'ANSWER' not in self.priority_queues:
            self.priority_queues['ANSWER'] = []
        
        # Accessing the key in a defaultdict(dict) automatically creates it
        # preventing the KeyError downstream when the query fails.
        if 'ANSWER' not in self.global_candidates:
            _ = self.global_candidates['ANSWER']
        # --- FIX ENDS HERE ---

    def _update_candidate_score(
        self,
        existing_candidate,
        source_score: float,
        support_boost: float
    ):
        """Update candidate score logic (Retained from Script 1)."""
        # Keep track of best path quality
        existing_candidate.score = max(existing_candidate.score, source_score)
        
        # Apply logarithmic support multiplier
        support_multiplier = 1 + (support_boost * math.log(existing_candidate.support + 1))
        updated_score = existing_candidate.score * support_multiplier
        
        existing_candidate.score = updated_score
        return updated_score

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
        """
        Add or Update candidate using Global Map and Heap Pushing.
        """
        # Check limits on NEW candidates only
        if node_id not in self.global_candidates[entity]:
            max_limit = self.max_answer_candidates if entity == 'ANSWER' else self.max_candidates_per_symbol
            if len(self.global_candidates[entity]) >= max_limit:
                if self.verbose:
                    print(f"    [LIMIT] '{entity}' max candidates reached")
                return False

        # Access Singleton from Map
        existing = self.global_candidates[entity].get(node_id)
        triplet = (source_entity, relation, target_entity)
        
        if existing:
            # --- UPDATE EXISTING ---
            if triplet not in existing.triplets:
                existing.add_triplet(triplet)
                existing.add_symbol_candidates(source_entity, source_node, source_score)
                
                # Recalculate Score (Script 1 Logic)
                updated_score = self._update_candidate_score(existing, source_score, self.support_boost)
                
                # Push NEW state to heap (Lazy Deletion approach)
                # The old state remains in the heap but will be ignored when popped
                heapq.heappush(
                    self.priority_queues[entity],
                    (-existing.support, -updated_score, node_id)
                )
                
                if self.verbose:
                    print(f"    ↳ [UPDATE] Node {node_id} for '{entity}' | new_support={existing.support} | new_score={updated_score:.4f}")
            return True
        else:
            # --- CREATE NEW ---
            new_candidate = CandidateContext(
                node_id=node_id,
                entity=entity,
                score=score
            )
            new_candidate.add_triplet(triplet)
            new_candidate.add_symbol_candidates(source_entity, source_node, source_score)
            
            # Store in Map
            self.global_candidates[entity][node_id] = new_candidate
            
            # Push to Heap
            heapq.heappush(
                self.priority_queues[entity],
                (-new_candidate.support, -score, node_id)
            )
            
            if self.verbose:
                print(f"    [+] Added node {node_id} for '{entity}' | score={score:.4f}")
            
            return True

    def _score_neighbors_with_vss(self, neighbor_nodes: List[int], target_entity: str) -> Dict[int, float]:
        """Score neighbor nodes using VSS (Retained from Script 1)."""
        if not neighbor_nodes: return {}
        
        query_emb = self.entity_query_embeddings.get(target_entity)
        if query_emb is None:
            return {node_id: 0.5 for node_id in neighbor_nodes}
        
        entity_data = self.query_obj.entities.get(target_entity, {})
        node_types = entity_data.get('type', [])
        
        if not node_types:
            return {node_id: 0.5 for node_id in neighbor_nodes}
        
        score_dict = {}
        for node_type in node_types:
            if node_type in self.vss_retriever.node_emb_dict:
                type_scores = self.vss_retriever.compute_similarities(
                    query_emb=query_emb,
                    node_type=node_type,
                    node_id_mask=neighbor_nodes
                )
                score_dict.update(type_scores)
        return score_dict

    def _process_candidate(
        self,
        entity: str,
        candidate: 'CandidateContext',
        relations_to_process: List[Tuple[str, List[str]]]
    ) -> int:
        """
        Expand candidate to neighbors (Retained Logic from Script 1).
        """
        curr_node = candidate.node_id
        curr_score = candidate.score
        added_count = 0
        
        # Double check processed (though ground loop checks this too)
        if curr_node in self.processed_nodes[entity]:
            return 0
        
        if self.verbose:
            print(f"\n  [PROCESS] Entity '{entity}', Node {curr_node}, Score {curr_score:.4f}, Support {candidate.support}")
        
        self.processed_nodes[entity].add(curr_node)
        
        for target_entity, relation_types in relations_to_process:
            for relation in relation_types:
                neighbors = self.kb.get_neighbor_nodes(curr_node, relation)
                if not neighbors: continue
                
                # VSS Scoring
                neighbor_scores = self._score_neighbors_with_vss(neighbors, target_entity)
                
                sorted_neighbors = sorted(neighbor_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k_neighbors]
                
                for neigh_node, vss_score in sorted_neighbors:
                    if neigh_node in self.processed_nodes[target_entity]:
                        continue
                    
                    # Harmonic Mean Logic (Script 1)
                    decayed_source = curr_score * self.score_decay
                    epsilon = 1e-9

                    if vss_score > epsilon and decayed_source > epsilon:
                        propagated_score = 2 / (1/vss_score + 1/decayed_source)
                    elif vss_score > epsilon:
                        propagated_score = vss_score * 0.9
                    elif decayed_source > epsilon:
                        propagated_score = decayed_source * 0.9
                    else:
                        propagated_score = 0.1

                    triplet_target = target_entity
                    
                    success = self._add_candidate(
                        entity=target_entity,
                        node_id=neigh_node,
                        score=propagated_score,
                        source_entity=entity,
                        source_node=curr_node,
                        source_score=curr_score,
                        relation=relation,
                        target_entity=triplet_target
                    )
                    if success: added_count += 1
        return added_count
    
    def ground(self) -> Dict[str, List['CandidateContext']]:
        """
        Main Loop with Stale Entry Checking (Lazy Deletion).
        """
        self._initialize_queues()
        
        relations = self.query_obj.relations if self.query_obj.relations else {}
        entity_relations = defaultdict(list)
        for (source_entity, target_entity), relation_types in relations.items():
            entity_relations[source_entity].append((target_entity, relation_types))
            entity_relations[target_entity].append((source_entity, relation_types))
        
        entities_to_process = [e for e in self.query_obj.entities.keys() if e != 'ANSWER']
        iteration = 0
        
        while True:
            iteration += 1
            if self.verbose: 
                print(f"\n{'='*70}")
                print(f"ROUND {iteration}")
                print(f"{'='*70}")
            
            round_processed = 0
            
            for entity in entities_to_process:
                # Limit check based on map size
                if len(self.global_candidates[entity]) >= self.max_candidates_per_symbol:
                    # If we have processed most candidates, skip
                    if len(self.processed_nodes[entity]) >= len(self.global_candidates[entity]):
                         continue

                # Process 1 valid item from the heap
                while self.priority_queues[entity]:
                    # Pop tuple: (-support, -score, node_id)
                    neg_support, neg_score, node_id = heapq.heappop(self.priority_queues[entity])
                    
                    # 1. Processed Check
                    if node_id in self.processed_nodes[entity]:
                        continue
                    
                    # 2. STALE ENTRY CHECK (Lazy Deletion)
                    # Get the 'live' candidate object
                    current_candidate = self.global_candidates[entity][node_id]
                    
                    # Compare popped values (negated) with current object values
                    # Using small epsilon for float comparison on score
                    if (-neg_support != current_candidate.support) or \
                       (abs((-neg_score) - current_candidate.score) > 1e-9):
                        # This tuple represents an old state of the candidate. Skip it.
                        continue
                        
                    # If we match, this is the valid/latest entry. Process it.
                    added = self._process_candidate(
                        entity, 
                        current_candidate, 
                        entity_relations.get(entity, [])
                    )
                    round_processed += 1
                    break # Move to next entity after processing one node
            
            # Stop Conditions
            if len(self.global_candidates.get('ANSWER', [])) >= self.max_answer_candidates:
                break
            
            if round_processed == 0:
                break
                
        # Return dict of lists to match expected output format
        return {e: list(self.global_candidates[e].values()) for e in self.global_candidates}

def run_priority_queue_grounding(query_obj, kb, vss_retriever, **kwargs):
    grounder = PriorityQueueGrounding(query_obj, kb, vss_retriever, **kwargs)
    return grounder.ground()