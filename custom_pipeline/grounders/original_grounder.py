import heapq
import math
from typing import Dict, Set, List, Tuple, TYPE_CHECKING, Optional
from collections import defaultdict
import pandas as pd

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
    Priority queue-based grounding with comprehensive statistics tracking.
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
        
        # Priority queues: max heap (negate scores for heapq)
        self.candidate_queues = {}
        
        # Tracking
        self.processed_nodes = defaultdict(set)
        self.all_candidates = defaultdict(list)
        self.candidate_count = defaultdict(int)
        
        # Entity query strings for VSS scoring
        self.entity_query_strings = {}
        self.entity_query_embeddings = {}
        
        # Statistics tracking
        self.ground_truths = None
        self.statistics = None
        self.current_round = 0
        self.processing_order = defaultdict(int)  # {entity: count of processed nodes}
        
    def _initialize_statistics(self, ground_truths: Optional[List[int]]):
        """Initialize statistics tracking structures."""
        if ground_truths is None:
            self.statistics = None
            return
        
        self.ground_truths = set(ground_truths)
        
        self.statistics = {
            'per_symbol': {},
            'ground_truth_tracking': {},
            'global_stats': {
                'total_rounds': 0,
                'total_nodes_processed': 0,
                'total_candidates_added': 0,
                'ground_truths_found': 0,
                'ground_truths_missed': len(ground_truths),
                'vss_neighbors_filtered': 0,
                'rejected_candidates': 0,
                'queue_sizes_by_round': {},
                'ground_truths_found_by_round': {},
                'avg_scores_by_entity_by_round': {},
            }
        }
        
        # Initialize per-symbol stats
        for entity in self.query_obj.entities.keys():
            self.statistics['per_symbol'][entity] = {
                'total_processed': 0,
                'total_candidates_added': 0,
                'avg_candidates_per_node': 0.0,
            }
        
        # Initialize ground truth tracking
        for gt in ground_truths:
            self.statistics['ground_truth_tracking'][gt] = {
                'found': False,
                'added_in_round': None,
                'added_by_source_entity': None,
                'added_by_source_node': None,
                'queue_position_when_processed': None,
                'initial_score': None,
                'final_score': None,
                'support_boosts': 0,
                'score_progression': [],
                'contributing_entities': [],
                'contributing_paths': [],
                'final_rank': None,
                'why_not_found': None,
            }
    
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
        """Initialize priority queues with initial candidates."""
        if self.verbose:
            print("\n[INIT] Initializing priority queues with initial candidates")
        
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
        
        # Initialize queues with initial candidates
        for entity, cand_list in self.query_obj.initial_symbol_candidates.items():
            self.candidate_queues[entity] = []
            
            for cand in cand_list:
                heapq.heappush(
                    self.candidate_queues[entity],
                    (-cand.score, cand.node_id, cand)
                )
                self.all_candidates[entity].append(cand)
                self.candidate_count[entity] += 1
            
            if self.verbose:
                print(f"  Initialized '{entity}' with {len(cand_list)} candidates")
        
        # Initialize ANSWER queue if not present
        if 'ANSWER' not in self.candidate_queues:
            self.candidate_queues['ANSWER'] = []
            if self.verbose:
                print(f"  Initialized 'ANSWER' with 0 candidates")
    
    def _get_top_candidate(self, entity: str) -> Optional[Tuple[float, int, 'CandidateContext']]:
        """Pop the top candidate from an entity's queue."""
        if entity in self.candidate_queues and self.candidate_queues[entity]:
            return heapq.heappop(self.candidate_queues[entity])
        return None
    
    def _track_ground_truth_added(
        self,
        node_id: int,
        score: float,
        source_entity: str,
        source_node: int,
        source_score: float
    ):
        """Track when a ground truth node is added."""
        if self.statistics is None or node_id not in self.ground_truths:
            return
        
        gt_stats = self.statistics['ground_truth_tracking'][node_id]
        
        if not gt_stats['found']:
            # First time this ground truth is added
            gt_stats['found'] = True
            gt_stats['added_in_round'] = self.current_round
            gt_stats['added_by_source_entity'] = source_entity
            gt_stats['added_by_source_node'] = source_node
            gt_stats['queue_position_when_processed'] = self.processing_order[source_entity]
            gt_stats['initial_score'] = score
            gt_stats['score_progression'].append(score)
            
            self.statistics['global_stats']['ground_truths_found'] += 1
            self.statistics['global_stats']['ground_truths_missed'] -= 1
            
            # Track in per-round stats
            if self.current_round not in self.statistics['global_stats']['ground_truths_found_by_round']:
                self.statistics['global_stats']['ground_truths_found_by_round'][self.current_round] = 0
            self.statistics['global_stats']['ground_truths_found_by_round'][self.current_round] += 1
        
        # Track contributing path
        if source_entity not in gt_stats['contributing_entities']:
            gt_stats['contributing_entities'].append(source_entity)
        gt_stats['contributing_paths'].append((source_entity, source_node, source_score))
    
    def _track_ground_truth_boost(self, node_id: int, new_score: float):
        """Track when a ground truth gets support boost."""
        if self.statistics is None or node_id not in self.ground_truths:
            return
        
        gt_stats = self.statistics['ground_truth_tracking'][node_id]
        if gt_stats['found']:
            gt_stats['support_boosts'] += 1
            gt_stats['score_progression'].append(new_score)
    
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
        """Add a new candidate to an entity's queue."""
        max_limit = self.max_answer_candidates if entity == 'ANSWER' else self.max_candidates_per_symbol
        
        if self.candidate_count[entity] >= max_limit:
            if self.verbose:
                print(f"    [LIMIT] '{entity}' has reached max candidates ({max_limit})")
            return False
        
        existing = next(
            (c for c in self.all_candidates[entity] if c.node_id == node_id),
            None
        )
        
        triplet = (source_entity, relation, target_entity)
        
        if existing:
            if triplet not in existing.triplets:
                old_score = existing.score
                
                existing.add_triplet(triplet)
                existing.add_symbol_candidates(source_entity, source_node, source_score)
                
                updated_score = self._update_candidate_score(existing, source_score, self.support_boost)
                
                # Remove old entry and re-push
                self.candidate_queues[entity] = [
                    (s, nid, c) for (s, nid, c) in self.candidate_queues[entity] if nid != node_id
                ]
                heapq.heapify(self.candidate_queues[entity])
                heapq.heappush(self.candidate_queues[entity], (-updated_score, node_id, existing))
                
                # Track ground truth boost
                self._track_ground_truth_boost(node_id, updated_score)
                
                if self.verbose:
                    print(f"    ↳ [UPDATE] Node {node_id} for '{entity}' | support={existing.support} | score={updated_score:.4f}")
            return True
        else:
            # Create new candidate
            new_candidate = CandidateContext(
                node_id=node_id,
                entity=entity,
                score=score
            )
            new_candidate.add_triplet(triplet)
            new_candidate.add_symbol_candidates(source_entity, source_node, source_score)
            
            heapq.heappush(
                self.candidate_queues[entity],
                (-score, node_id, new_candidate)
            )
            self.all_candidates[entity].append(new_candidate)
            self.candidate_count[entity] += 1
            
            # Update statistics
            if self.statistics:
                self.statistics['per_symbol'][source_entity]['total_candidates_added'] += 1
                self.statistics['global_stats']['total_candidates_added'] += 1
                        
            # Track ground truth
            self._track_ground_truth_added(node_id, score, source_entity, source_node, source_score)
            
            if self.verbose:
                print(f"    [+] Added node {node_id} for '{entity}' | score={score:.4f} | total={self.candidate_count[entity]}")
            
            return True
    
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
    
    def _update_candidate_score(self, existing_candidate, source_score, support_boost=0.15):
        # Validate source_score
        if source_score is None or source_score > 10:  # Cap at reasonable max
            source_score = existing_candidate.score
        
        # Track base score separately (better fix)
        if not hasattr(existing_candidate, 'base_score'):
            existing_candidate.base_score = existing_candidate.score
        
        existing_candidate.base_score = max(existing_candidate.base_score, source_score)
        
        support_multiplier = 1 + (support_boost * math.log(existing_candidate.support + 1))
        updated_score = existing_candidate.base_score * support_multiplier
        
        existing_candidate.score = updated_score
        return updated_score
    
    def _process_candidate(
        self,
        entity: str,
        candidate: 'CandidateContext',
        relations_to_process: List[Tuple[str, List[str]]]
    ) -> int:
        """Process a single candidate by expanding to neighbors."""
        curr_node = candidate.node_id
        curr_score = candidate.score
        added_count = 0
        
        if curr_node in self.processed_nodes[entity]:
            return 0
        
        # Increment processing order for this entity
        self.processing_order[entity] += 1
        
        if self.verbose:
            print(f"\n  [PROCESS] Entity '{entity}', Node {curr_node}, Score {curr_score:.4f}")
        
        self.processed_nodes[entity].add(curr_node)
        
        # Update statistics
        if self.statistics:
            self.statistics['per_symbol'][entity]['total_processed'] += 1
            self.statistics['global_stats']['total_nodes_processed'] += 1
        
        # Process each relation
        for target_entity, relation_types in relations_to_process:
            for relation in relation_types:
                neighbors = self.kb.get_neighbor_nodes(curr_node, relation)
                
                if not neighbors:
                    continue
                
                if self.verbose:
                    print(f"    [REL] {entity} --{relation}--> {target_entity}: {len(neighbors)} neighbors")
                
                # Score neighbors
                neighbor_scores = self._score_neighbors_with_vss(neighbors, target_entity)
                
                # Sort and take top-k
                sorted_neighbors = sorted(
                    neighbor_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Track VSS filtering
                if self.statistics and len(sorted_neighbors) > self.top_k_neighbors:
                    filtered_count = len(sorted_neighbors) - self.top_k_neighbors
                    self.statistics['global_stats']['vss_neighbors_filtered'] += filtered_count
                
                sorted_neighbors = sorted_neighbors[:self.top_k_neighbors]
                
                if self.verbose and len(neighbor_scores) > len(sorted_neighbors):
                    print(f"    [FILTER] Taking top {self.top_k_neighbors} of {len(neighbor_scores)} neighbors")
                
                # Add top-k neighbors
                for neigh_node, vss_score in sorted_neighbors:
                    if neigh_node in self.processed_nodes[target_entity]:
                        continue
                    
                    # Compute propagated score using harmonic mean
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
    
    def _finalize_statistics(self):
        """Finalize statistics after grounding completes."""
        if self.statistics is None:
            return
        
        # Calculate averages
        for entity, stats in self.statistics['per_symbol'].items():
            if stats['total_processed'] > 0:
                stats['avg_candidates_per_node'] = stats['total_candidates_added'] / stats['total_processed']
        
        # Finalize ground truth tracking
        for gt, gt_stats in self.statistics['ground_truth_tracking'].items():
            if gt_stats['found']:
                # Find final rank
                answer_candidates = self.all_candidates.get('ANSWER', [])
                sorted_candidates = sorted(answer_candidates, key=lambda c: c.score, reverse=True)
                
                for rank, cand in enumerate(sorted_candidates, 1):
                    if cand.node_id == gt:
                        gt_stats['final_rank'] = rank
                        gt_stats['final_score'] = cand.score
                        break
            else:
                # Track why not found
                gt_stats['why_not_found'] = "No path found or filtered out"
    
    def ground(self, ground_truths: Optional[List[int]] = None) -> Dict:
        """
        Main grounding loop with comprehensive statistics tracking.
        
        Args:
            ground_truths: Optional list of ground truth node IDs to track
        
        Returns:
            {
                'candidates': Dict[entity, List[CandidateContext]],
                'statistics': Dict (or None if no ground_truths provided)
            }
        """
        # Initialize statistics tracking
        self._initialize_statistics(ground_truths)
        
        self._initialize_queues()
        
        relations = self.query_obj.relations if self.query_obj.relations else {}
        
        if self.verbose:
            print(f"\n[INFO] Relations to process: {relations}")
        
        # Build relation lookup
        entity_relations = defaultdict(list)
        for (source_entity, target_entity), relation_types in relations.items():
            entity_relations[source_entity].append((target_entity, relation_types))
            entity_relations[target_entity].append((source_entity, relation_types))
        
        if self.verbose:
            print(f"\n[INFO] Entity relations lookup:")
            for entity, rels in entity_relations.items():
                print(f"  {entity}: {rels}")
        
        # Round-robin processing
        entities_to_process = [e for e in self.query_obj.entities.keys() if e != 'ANSWER']
        
        if self.verbose:
            print(f"\n[INFO] Starting round-robin grounding")
            print(f"[INFO] Processing order (per round): {entities_to_process}")
        
        while True:
            self.current_round += 1
            
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"ROUND {self.current_round}")
                print(f"{'='*70}")
            
            round_processed = 0
            
            # Track queue sizes for this round
            if self.statistics:
                self.statistics['global_stats']['queue_sizes_by_round'][self.current_round] = {
                    entity: len(queue) for entity, queue in self.candidate_queues.items()
                }
            
            # Process one candidate from each entity
            for entity in entities_to_process:
                if self.candidate_count[entity] >= self.max_candidates_per_symbol:
                    if self.verbose:
                        print(f"\n[STOP] '{entity}' has reached max candidates")
                    continue
                
                top = self._get_top_candidate(entity)
                if top is None:
                    if self.verbose:
                        print(f"\n[EMPTY] No more candidates for '{entity}'")
                    continue
                
                neg_score, node_id, candidate = top
                
                if node_id in self.processed_nodes[entity]:
                    continue
                
                added = self._process_candidate(
                    entity=entity,
                    candidate=candidate,
                    relations_to_process=entity_relations.get(entity, [])
                )
                
                round_processed += 1
            
            # Check stopping conditions
            if self.candidate_count['ANSWER'] >= self.max_answer_candidates:
                if self.verbose:
                    print(f"\n[STOP] ANSWER has reached max candidates")
                break
            
            if round_processed == 0:
                if self.verbose:
                    print(f"\n[STOP] No candidates processed in this round")
                break
            
            if self.verbose:
                print(f"\n[ROUND {self.current_round}] Processed {round_processed} candidates")
                print(f"[STATUS] Candidate counts: {dict(self.candidate_count)}")
        
        # Finalize statistics
        if self.statistics:
            self.statistics['global_stats']['total_rounds'] = self.current_round
            self._finalize_statistics()
        
        if self.verbose:
            print("\n" + "="*70)
            print("✅ GROUNDING COMPLETED")
            print("="*70)
            print("\nFinal candidate counts:")
            for entity, count in self.candidate_count.items():
                print(f"  {entity}: {count} candidates (processed: {len(self.processed_nodes[entity])})")
        
        return {
            'candidates': dict(self.all_candidates),
            'statistics': self.statistics
        }


def run_priority_queue_grounding(
    query_obj,
    kb,
    vss_retriever,
    ground_truths: Optional[List[int]] = None,
    max_candidates_per_symbol: int = 1000,
    max_answer_candidates: int = 100,
    top_k_neighbors: int = 10,
    support_boost: float = 0.15,
    score_decay: float = 0.9,
    verbose: bool = True
) -> Dict:
    """
    Convenience function to run priority queue-based grounding with statistics.
    
    Args:
        query_obj: Query class instance
        kb: Knowledge base object
        vss_retriever: VSSRetriever instance
        ground_truths: Optional list of ground truth node IDs to track
        max_candidates_per_symbol: Max candidates per entity (default 1000)
        max_answer_candidates: Max candidates for ANSWER (default 100)
        top_k_neighbors: Top neighbors to add per expansion (default 10)
        support_boost: Support boost factor (default 0.15)
        score_decay: Score decay factor (default 0.9)
        verbose: Print detailed logs (default True)
    
    Returns:
        {
            'candidates': Dict[entity, List[CandidateContext]],
            'statistics': Dict (or None if no ground_truths)
        }
    """
    grounder = PriorityQueueGrounding(
        query_obj=query_obj,
        kb=kb,
        vss_retriever=vss_retriever,
        max_candidates_per_symbol=max_candidates_per_symbol,
        max_answer_candidates=max_answer_candidates,
        top_k_neighbors=top_k_neighbors,
        support_boost=support_boost,
        score_decay=score_decay,
        verbose=verbose
    )
    
    return grounder.ground(ground_truths=ground_truths)


    """
    Convert statistics to pandas DataFrames for easy analysis and plotting.
    
    Args:
        statistics: Statistics dict from grounding
    
    Returns:
        {
            'per_symbol_df': Per-symbol statistics,
            'ground_truth_df': Ground truth tracking,
            'rounds_df': Per-round statistics,
            'score_progression_df': Score progression for ground truths (long format)
        }
    """
    if statistics is None:
        return None
    
    # Per-symbol DataFrame
    per_symbol_rows = []
    for entity, stats in statistics['per_symbol'].items():
        row = {
            'entity': entity,
            'total_processed': stats['total_processed'],
            'total_candidates_added': stats['total_candidates_added'],
            'avg_candidates_per_node': stats['avg_candidates_per_node'],
            'rejected_candidates': stats['rejected_candidates'],
        }
        per_symbol_rows.append(row)
    per_symbol_df = pd.DataFrame(per_symbol_rows)
    
    # Ground truth DataFrame
    gt_rows = []
    for gt_id, gt_stats in statistics['ground_truth_tracking'].items():
        row = {
            'ground_truth_id': gt_id,
            'found': gt_stats['found'],
            'added_in_round': gt_stats['added_in_round'],
            'added_by_source_entity': gt_stats['added_by_source_entity'],
            'queue_position': gt_stats['queue_position_when_processed'],
            'initial_score': gt_stats['initial_score'],
            'final_score': gt_stats['final_score'],
            'support_boosts': gt_stats['support_boosts'],
            'num_contributing_entities': len(gt_stats['contributing_entities']),
            'final_rank': gt_stats['final_rank'],
        }
        gt_rows.append(row)
    ground_truth_df = pd.DataFrame(gt_rows)
    
    # Rounds DataFrame
    rounds_rows = []
    for round_num, queue_sizes in statistics['global_stats']['queue_sizes_by_round'].items():
        row = {'round': round_num}
        row.update(queue_sizes)
        row['gts_found'] = statistics['global_stats']['ground_truths_found_by_round'].get(round_num, 0)
        rounds_rows.append(row)
    rounds_df = pd.DataFrame(rounds_rows)
    
    # Score progression DataFrame (long format for plotting)
    score_prog_rows = []
    for gt_id, gt_stats in statistics['ground_truth_tracking'].items():
        if gt_stats['found']:
            for step, score in enumerate(gt_stats['score_progression']):
                score_prog_rows.append({
                    'ground_truth_id': gt_id,
                    'step': step,
                    'score': score
                })
    score_progression_df = pd.DataFrame(score_prog_rows)
    
    return {
        'per_symbol_df': per_symbol_df,
        'ground_truth_df': ground_truth_df,
        'rounds_df': rounds_df,
        'score_progression_df': score_progression_df,
    }