import heapq
import math
from typing import Dict, Set, List, Tuple, TYPE_CHECKING, Optional
from collections import defaultdict
from copy import deepcopy
# Import CandidateContext
if TYPE_CHECKING:
    from custom_pipeline.new_candidate_context import CandidateContext
else:
    try:
        from .new_candidate_context import CandidateContext
    except ImportError:
        from custom_pipeline.new_candidate_context import CandidateContext

class PriorityQueueGrounding:
    """
    Priority queue-based grounding with separated path finding and scoring.
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
        nodes_per_round: int = 1,
        max_paths_per_candidate: int = 50,
        verbose: bool = True
    ):
        self.query_obj = query_obj
        self.kb = kb
        self.vss_retriever = vss_retriever
        self.max_candidates_per_symbol = max_candidates_per_symbol
        self.max_answer_candidates = max_answer_candidates
        self.top_k_neighbors = top_k_neighbors
        self.score_decay = score_decay
        self.nodes_per_round = nodes_per_round
        self.max_paths_per_candidate = max_paths_per_candidate
        self.verbose = verbose
        
        # Global candidate map: {entity: {node_id: CandidateContext}}
        self.global_candidates = defaultdict(dict)
        
        # Priority queues: {entity: [(-support, -score, node_id)]}
        self.priority_queues = defaultdict(list)
        
        # Processed nodes: {entity: {node_id, ...}}
        self.processed_nodes = defaultdict(set)
        
        # Entity query strings for VSS scoring
        self.entity_query_strings = {}
        self.entity_query_embeddings = {}
        
        # Tracking
        self.current_round = 0
    
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
    
    def _initialize(self):
        """Initialize data structures with initial candidates."""
        if self.verbose:
            print("\n[INIT] Initializing grounding system")
        
        # Build entity query strings and cache embeddings
        for entity in self.query_obj.entities.keys():
            query_str = self._build_entity_query_string(entity)
            self.entity_query_strings[entity] = query_str
            
            if query_str:
                query_emb = self.vss_retriever.get_query_emb(query_str)
                self.entity_query_embeddings[entity] = query_emb
            else:
                self.entity_query_embeddings[entity] = None
            
            if self.verbose:
                print(f"  Entity '{entity}' query: '{query_str}'")
        
        # Initialize with initial candidates
        for entity, cand_list in self.query_obj.initial_symbol_candidates.items():
            for cand in cand_list:
                # Check if it's already a new CandidateContext with paths
                if hasattr(cand, 'paths') and len(cand.paths) > 0:
                    # Already properly initialized from get_initial_candidates_for_entity
                    self.global_candidates[entity][cand.node_id] = cand
                    
                    # Add to priority queue
                    heapq.heappush(
                        self.priority_queues[entity],
                        (-cand.support, -cand.current_score, cand.node_id)
                    )
                else:
                    # Old style candidate, need to convert
                    # Handle both .score and .initial_score
                    score = getattr(cand, 'initial_score', None) or getattr(cand, 'score', 0.5)
                    
                    # Create new candidate context
                    context = CandidateContext(
                        node_id=cand.node_id,
                        entity=entity,
                        initial_score=score
                    )
                    
                    # Add initial path (just this node)
                    initial_path = {
                        'entities': {entity: cand.node_id},
                        'scores': {entity: score},
                        'relations': []
                    }
                    context.add_path(initial_path)
                    
                    # Add to global map
                    self.global_candidates[entity][cand.node_id] = context
                    
                    # Add to priority queue
                    heapq.heappush(
                        self.priority_queues[entity],
                        (-context.support, -context.current_score, cand.node_id)
                    )
            
            if self.verbose:
                print(f"  Initialized '{entity}' with {len(cand_list)} candidates")
    
        # Initialize ANSWER structures if not present
        if 'ANSWER' not in self.priority_queues:
            self.priority_queues['ANSWER'] = []
            if self.verbose:
                print(f"  Initialized 'ANSWER' with 0 candidates")

    def _score_neighbors_with_vss(
        self,
        neighbor_nodes: List[int],
        target_entity: str
    ) -> Dict[int, float]:
        """Score neighbor nodes using VSS similarity to target entity query."""
        if not neighbor_nodes:
            return {}
        
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
    
    def _expand_candidate(
        self,
        source_entity: str,
        source_node_id: int,
        source_context: CandidateContext,
        relations_to_process: List[Tuple[str, List[str]]],
        temp_queues: Dict[str, List[Tuple[int, int, float]]]
    ):
        """Expand a single candidate to its neighbors."""
        if self.verbose:
            print(f"\n  [EXPAND] Entity '{source_entity}', Node {source_node_id}, Score {source_context.current_score:.4f}, Support {source_context.support}")
        
        # Process each relation
        for target_entity, relation_types in relations_to_process:
            for relation in relation_types:
                neighbors = self.kb.get_neighbor_nodes(source_node_id, relation)
                
                if not neighbors:
                    continue
                
                if self.verbose:
                    print(f"    [REL] {source_entity} --{relation}--> {target_entity}: {len(neighbors)} neighbors")
                
                # Score neighbors
                neighbor_scores = self._score_neighbors_with_vss(neighbors, target_entity)
                
                # Sort and take top-k
                sorted_neighbors = sorted(
                    neighbor_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:self.top_k_neighbors]
                
                if self.verbose and len(neighbor_scores) > len(sorted_neighbors):
                    print(f"    [FILTER] Taking top {self.top_k_neighbors} of {len(neighbor_scores)} neighbors")
                
                # Process top-k neighbors
                for neigh_node, vss_score in sorted_neighbors:
                    # Skip if already processed
                    if neigh_node in self.processed_nodes[target_entity]:
                        continue
                    
                    # Check limit
                    max_limit = self.max_answer_candidates if target_entity == 'ANSWER' else self.max_candidates_per_symbol
                    if len(self.global_candidates[target_entity]) >= max_limit:
                        continue
                    
                    # Get or create target context
                    if neigh_node not in self.global_candidates[target_entity]:
                        target_context = CandidateContext(
                            node_id=neigh_node,
                            entity=target_entity,
                            initial_score=vss_score
                        )
                        target_context.current_score = 0.0  # Will be updated
                        self.global_candidates[target_entity][neigh_node] = target_context
                    else:
                        target_context = self.global_candidates[target_entity][neigh_node]
                    
                    # Extend all paths from source to target
                    for source_path in source_context.paths:
                        new_path = deepcopy(source_path)
                        new_path['entities'][target_entity] = neigh_node
                        new_path['scores'][target_entity] = vss_score
                        new_path['relations'].append((source_entity, relation, target_entity))
                        
                        target_context.add_path(new_path)
                    
                    # Prune paths if needed
                    target_context.prune_paths(self.max_paths_per_candidate)
                    
                    # Update score: max of current and decayed source
                    decayed_score = source_context.current_score * self.score_decay
                    target_context.current_score = max(target_context.current_score, decayed_score)
                    
                    # Add to temp queue
                    temp_queues[target_entity].append((
                        neigh_node,
                        target_context.support,
                        target_context.current_score
                    ))
                    
                    if self.verbose:
                        print(f"    [+] Added node {neigh_node} for '{target_entity}' | support={target_context.support} | score={target_context.current_score:.4f}")
    
    def ground(self) -> Dict:
        """
        Main grounding loop.
        
        Returns:
            {
                'candidates': Dict[entity, Dict[node_id, CandidateContext]],
            }
        """
        self._initialize()
        
        relations = self.query_obj.relations if self.query_obj.relations else {}
        
        if self.verbose:
            print(f"\n[INFO] Relations to process: {relations}")
        
        # Build relation lookup
        entity_relations = defaultdict(list)
        for (source_entity, target_entity), relation_types in relations.items():
            entity_relations[source_entity].append((target_entity, relation_types))
        
        if self.verbose:
            print(f"\n[INFO] Entity relations lookup:")
            for entity, rels in entity_relations.items():
                print(f"  {entity}: {rels}")
        
        # Entities to process (exclude ANSWER)
        entities_to_process = [e for e in self.query_obj.entities.keys() if e != 'ANSWER']
        
        if self.verbose:
            print(f"\n[INFO] Starting grounding")
            print(f"[INFO] Processing order: {entities_to_process}")
            print(f"[INFO] Nodes per round: {self.nodes_per_round}")
        
        # Main grounding loop
        while True:
            self.current_round += 1
            
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"ROUND {self.current_round}")
                print(f"{'='*70}")
            
            temp_queues = defaultdict(list)
            nodes_processed_this_round = 0
            
            # Process top c nodes from each entity
            for entity in entities_to_process:
                processed_count = 0
                
                while processed_count < self.nodes_per_round:
                    if not self.priority_queues[entity]:
                        break
                    
                    # Pop from queue
                    neg_support, neg_score, node_id = heapq.heappop(self.priority_queues[entity])
                    
                    # Skip if already processed (stale entry)
                    if node_id in self.processed_nodes[entity]:
                        continue
                    
                    # Mark as processed
                    self.processed_nodes[entity].add(node_id)
                    processed_count += 1
                    nodes_processed_this_round += 1
                    
                    # Get context
                    context = self.global_candidates[entity][node_id]
                    
                    # Expand to neighbors
                    self._expand_candidate(
                        source_entity=entity,
                        source_node_id=node_id,
                        source_context=context,
                        relations_to_process=entity_relations.get(entity, []),
                        temp_queues=temp_queues
                    )
            
            # Merge temp queues into main queues
            for entity, updates in temp_queues.items():
                for node_id, support, score in updates:
                    heapq.heappush(
                        self.priority_queues[entity],
                        (-support, -score, node_id)
                    )
            
            if self.verbose:
                print(f"\n[ROUND {self.current_round}] Processed {nodes_processed_this_round} nodes")
                print(f"[STATUS] Candidate counts: {dict((k, len(v)) for k, v in self.global_candidates.items())}")
            
            # Check stopping conditions
            if len(self.global_candidates['ANSWER']) >= self.max_answer_candidates:
                if self.verbose:
                    print(f"\n[STOP] ANSWER has reached max candidates")
                break
            
            if nodes_processed_this_round == 0:
                if self.verbose:
                    print(f"\n[STOP] No nodes processed in this round")
                break
        
        if self.verbose:
            print("\n" + "="*70)
            print("✅ GROUNDING COMPLETED")
            print("="*70)
            print("\nFinal candidate counts:")
            for entity, candidates in self.global_candidates.items():
                print(f"  {entity}: {len(candidates)} candidates")
        
        return {
            'candidates': dict(self.global_candidates)
        }


def run_priority_queue_grounding_v2(
    query_obj,
    kb,
    vss_retriever,
    max_candidates_per_symbol: int = 1000,
    max_answer_candidates: int = 100,
    top_k_neighbors: int = 10,
    score_decay: float = 0.9,
    nodes_per_round: int = 1,
    max_paths_per_candidate: int = 50,
    verbose: bool = True
) -> Dict:
    """
    Convenience function to run priority queue-based grounding V2.
    """
    grounder = PriorityQueueGrounding(
        query_obj=query_obj,
        kb=kb,
        vss_retriever=vss_retriever,
        max_candidates_per_symbol=max_candidates_per_symbol,
        max_answer_candidates=max_answer_candidates,
        top_k_neighbors=top_k_neighbors,
        score_decay=score_decay,
        nodes_per_round=nodes_per_round,
        max_paths_per_candidate=max_paths_per_candidate,
        verbose=verbose
    )
    
    return grounder.ground()