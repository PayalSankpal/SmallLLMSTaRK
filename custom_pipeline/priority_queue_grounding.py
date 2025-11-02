import heapq
from typing import Dict, Set, List, Tuple, TYPE_CHECKING
from collections import defaultdict
import math  # ADD THIS

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
    Priority queue-based grounding that limits candidates per symbol
    and uses VSS to rank neighbor relevance.
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
        support_boost: float = 0.15,  # ADD THIS
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
        self.processed_nodes = defaultdict(set)  # {entity: set of processed node_ids}
        self.all_candidates = defaultdict(list)  # {entity: [CandidateContext objects]}
        self.candidate_count = defaultdict(int)  # {entity: total candidates added}
        
        # Entity query strings for VSS scoring
        self.entity_query_strings = {}
        self.entity_query_embeddings = {}  # Cache embeddings here
        
    def _build_entity_query_string(self, entity: str) -> str:
        """Build query string from entity's lexical and semantic constraints."""
        entity_data = self.query_obj.entities.get(entity, {})
        
        parts = []
        
        # Add lexical constraints
        lexical = entity_data.get('lexical', {})
        if 'name' in lexical:
            parts.append(lexical['name'])
        
        # Add semantic constraints
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
            
            # Cache the embedding
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
                # Max heap: negate score
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
    
    def _get_top_candidate(self, entity: str) -> Tuple[float, int, 'CandidateContext']:
        """Pop the top candidate from an entity's queue."""
        if entity in self.candidate_queues and self.candidate_queues[entity]:
            return heapq.heappop(self.candidate_queues[entity])
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
        """Add a new candidate to an entity's queue."""
        # Check if we've hit the limit
        max_limit = self.max_answer_candidates if entity == 'ANSWER' else self.max_candidates_per_symbol
        
        if self.candidate_count[entity] >= max_limit:
            if self.verbose:
                print(f"    [LIMIT] '{entity}' has reached max candidates ({max_limit})")
            return False
        
        # Check if node already exists in candidates
        existing = next(
            (c for c in self.all_candidates[entity] if c.node_id == node_id),
            None
        )
        
        triplet = (source_entity, relation, target_entity)
        
        if existing:
            # Update existing candidate
            if triplet not in existing.triplets:
                # CAPTURE OLD STATE
                old_score = existing.score
                old_support = existing.support - 1  # Before add_triplet increments it
                
                # Find old position in queue (approximate)
                queue_list = [(abs(s), nid) for s, nid, _ in self.candidate_queues[entity]]
                queue_list_sorted = sorted(queue_list, reverse=True)
                old_position = next((i for i, (_, nid) in enumerate(queue_list_sorted) if nid == node_id), -1)
                
                # UPDATE
                existing.add_triplet(triplet)
                existing.add_symbol_candidates(source_entity, source_node, source_score)
                
                updated_score = self._update_candidate_score(existing, source_score, self.support_boost)
                self.candidate_queues[entity] = [
                    (s, nid, c) for (s, nid, c) in self.candidate_queues[entity] if nid != node_id
                ]
                heapq.heapify(self.candidate_queues[entity])
                heapq.heappush(self.candidate_queues[entity], (-updated_score, node_id, existing))

                # # RE-PUSH TO HEAP WITH NEW SCORE (lazy deletion approach)
                # heapq.heappush(
                #     self.candidate_queues[entity],
                #     (-updated_score, node_id, existing)
                # )
                
                # CAPTURE NEW STATE
                new_score = updated_score
                new_support = existing.support
                
                # Find new position (after re-push)
                queue_list_new = [(abs(s), nid) for s, nid, _ in self.candidate_queues[entity]]
                queue_list_new_sorted = sorted(queue_list_new, reverse=True)
                new_position = next((i for i, (_, nid) in enumerate(queue_list_new_sorted) if nid == node_id), -1)
                
                # CALCULATE CHANGES
                score_change = new_score - old_score
                position_change = old_position - new_position  # Positive = moved up
                
                # DEBUG PRINT
                print(f"\n{'='*80}")
                print(f"🔄 SUPPORT BOOST UPDATE for Entity '{entity}', Node {node_id}")
                print(f"{'='*80}")
                print(f"📊 SCORE CHANGES:")
                print(f"   Old Score:      {old_score:.6f}")
                print(f"   New Score:      {new_score:.6f}")
                print(f"   Score Change:   {score_change:+.6f} ({(score_change/old_score)*100:+.2f}%)")
                print(f"\n📈 SUPPORT CHANGES:")
                print(f"   Old Support:    {old_support}")
                print(f"   New Support:    {new_support}")
                print(f"   Support Boost:  {self.support_boost}")
                print(f"   Multiplier:     {1 + (self.support_boost * math.log(new_support + 1)):.4f}")  # CHANGE THIS LINE
                print(f"\n🎯 QUEUE POSITION CHANGES:")
                print(f"   Queue Size:     {len(self.candidate_queues[entity])}")
                print(f"   Old Position:   #{old_position + 1}" if old_position >= 0 else "   Old Position:   Not found")
                print(f"   New Position:   #{new_position + 1}" if new_position >= 0 else "   New Position:   Not found")
                if old_position >= 0 and new_position >= 0:
                    print(f"   Position Change: {position_change:+d} {'⬆️ (moved up)' if position_change > 0 else ('⬇️ (moved down)' if position_change < 0 else '➡️ (same)')}")
                print(f"\n🔗 CONTRIBUTING PATH:")
                print(f"   Source Entity:  {source_entity}")
                print(f"   Source Node:    {source_node}")
                print(f"   Source Score:   {source_score:.6f}")
                print(f"   Relation:       {relation}")
                print(f"   Triplet:        {triplet}")
                print(f"\n📦 FULL CANDIDATE CONTEXT:")
                print(f"   {existing}")
                print(f"{'='*80}\n")
                
                if self.verbose:
                    print(f"    ↳ [UPDATE] Node {node_id} for '{entity}' | support={existing.support} | score={new_score:.4f}")
            return True
        else:
            # Create new candidate (no support boost here)
            new_candidate = CandidateContext(
                node_id=node_id,
                entity=entity,
                score=score
            )
            new_candidate.add_triplet(triplet)
            new_candidate.add_symbol_candidates(source_entity, source_node, source_score)
            
            # Add to queue and tracking
            heapq.heappush(
                self.candidate_queues[entity],
                (-score, node_id, new_candidate)
            )
            self.all_candidates[entity].append(new_candidate)
            self.candidate_count[entity] += 1
            
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
        
        # Use cached embedding
        query_emb = self.entity_query_embeddings.get(target_entity)
        if query_emb is None:
            # No query embedding available, return uniform scores
            return {node_id: 0.5 for node_id in neighbor_nodes}
        
        # Get target entity type(s)
        entity_data = self.query_obj.entities.get(target_entity, {})
        node_types = entity_data.get('type', [])
        
        if not node_types:
            return {node_id: 0.5 for node_id in neighbor_nodes}
        
        # Compute similarities for all neighbor nodes
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
    
    def _update_candidate_score(
        self,
        existing_candidate,
        source_score: float,
        support_boost: float = 0.15  # Adjusted for logarithmic scaling
    ):
        """Update candidate score when support increases using logarithmic diminishing returns."""
        # Keep track of best path quality
        existing_candidate.score = max(existing_candidate.score, source_score)
        
        # Apply logarithmic support multiplier for diminishing returns
        support_multiplier = 1 + (support_boost * math.log(existing_candidate.support + 1))
        updated_score = existing_candidate.score * support_multiplier
        
        # UPDATE THE CANDIDATE'S STORED SCORE
        existing_candidate.score = updated_score
        
        return updated_score
        
    def _process_candidate(
        self,
        entity: str,
        candidate: 'CandidateContext',
        relations_to_process: List[Tuple[str, List[str]]]
    ) -> int:
        """
        Process a single candidate by expanding to neighbors.
        Returns number of new candidates added.
        """
        curr_node = candidate.node_id
        curr_score = candidate.score
        added_count = 0
        # SKIP IF ALREADY PROCESSED (handles duplicate heap entries from score updates)
        if curr_node in self.processed_nodes[entity]:
            return 0  # ADD THIS CHECK
        
        if self.verbose:
            print(f"\n  [PROCESS] Entity '{entity}', Node {curr_node}, Score {curr_score:.4f}")
        
        # Mark as processed
        self.processed_nodes[entity].add(curr_node)
        
        # Process each relation
        for target_entity, relation_types in relations_to_process:
            for relation in relation_types:
                # Get all neighbors via this relation
                neighbors = self.kb.get_neighbor_nodes(curr_node, relation)
                
                if not neighbors:
                    continue
                
                if self.verbose:
                    print(f"    [REL] {entity} --{relation}--> {target_entity}: {len(neighbors)} neighbors")
                
                # Score neighbors using VSS
                neighbor_scores = self._score_neighbors_with_vss(neighbors, target_entity)
                
                # Sort and take top-k
                sorted_neighbors = sorted(
                    neighbor_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:self.top_k_neighbors]
                
                if self.verbose and len(sorted_neighbors) < len(neighbors):
                    print(f"    [FILTER] Taking top {self.top_k_neighbors} of {len(neighbors)} neighbors")
                
                # Add top-k neighbors as candidates
                for neigh_node, vss_score in sorted_neighbors:
                    # Skip if already processed
                    if neigh_node in self.processed_nodes[target_entity]:
                        continue
                    
                    # Compute propagated score
                    # Compute propagated score using weighted harmonic mean
                    # Prevents any single factor from dominating
                    decayed_source = curr_score * self.score_decay

                    # Harmonic mean: 2 / (1/a + 1/b)
                    # Handle edge cases where scores might be 0 or very small
                    epsilon = 1e-9  # Small value to prevent division by zero

                    if vss_score > epsilon and decayed_source > epsilon:
                        # Both scores valid - use harmonic mean
                        propagated_score = 2 / (1/vss_score + 1/decayed_source)
                    elif vss_score > epsilon:
                        # Only VSS score valid - use it with slight penalty
                        propagated_score = vss_score * 0.9
                    elif decayed_source > epsilon:
                        # Only source score valid - use it with slight penalty
                        propagated_score = decayed_source * 0.9
                    else:
                        # Both scores invalid - use small fallback
                        propagated_score = 0.1


                    # Determine triplet direction based on which entity is source/target
                    if entity == target_entity:
                        # Self-loop or bidirectional
                        triplet_source = entity
                        triplet_target = target_entity
                    else:
                        # Find the canonical direction from relations
                        # For now, always use (entity, target_entity) order
                        triplet_source = entity
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
                    
                    if success:
                        added_count += 1
        
        return added_count
    
    def ground(self) -> Dict[str, List['CandidateContext']]:
        """
        Main grounding loop using round-robin processing.
        
        Returns:
            Dictionary mapping entity symbols to lists of CandidateContext objects
        """
        self._initialize_queues()
        
        # Get all relations for processing
        relations = self.query_obj.relations if self.query_obj.relations else {}
        
        if self.verbose:
            print(f"\n[INFO] Relations to process: {relations}")
        
        # Build relation lookup: which entities should we expand to from each entity
        entity_relations = defaultdict(list)  # {entity: [(target_entity, [relation_types])]}
        
        for (source_entity, target_entity), relation_types in relations.items():
            # Bidirectional: both directions use same relations
            entity_relations[source_entity].append((target_entity, relation_types))
            entity_relations[target_entity].append((source_entity, relation_types))
        
        if self.verbose:
            print(f"\n[INFO] Entity relations lookup:")
            for entity, rels in entity_relations.items():
                print(f"  {entity}: {rels}")
        
        # Round-robin processing
        iteration = 0
        entities_to_process = [e for e in self.query_obj.entities.keys() if e != 'ANSWER']
        
        if self.verbose:
            print(f"\n[INFO] Starting round-robin grounding")
            print(f"[INFO] Processing order (per round): {entities_to_process}")
        
        while True:
            iteration += 1
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"ROUND {iteration}")
                print(f"{'='*70}")
            
            round_processed = 0
            
            # Process one candidate from each entity (round-robin)
            for entity in entities_to_process:
                # Check stopping condition for this entity
                if self.candidate_count[entity] >= self.max_candidates_per_symbol:
                    if self.verbose:
                        print(f"\n[STOP] '{entity}' has reached max candidates ({self.max_candidates_per_symbol})")
                    continue
                
                # Get top candidate
                top = self._get_top_candidate(entity)
                if top is None:
                    if self.verbose:
                        print(f"\n[EMPTY] No more candidates for '{entity}'")
                    continue
                
                neg_score, node_id, candidate = top
                
                # Skip if already processed
                if node_id in self.processed_nodes[entity]:
                    continue
                
                # Process this candidate
                added = self._process_candidate(
                    entity=entity,
                    candidate=candidate,
                    relations_to_process=entity_relations.get(entity, [])
                )
                
                round_processed += 1
            
            # Check global stopping conditions
            if self.candidate_count['ANSWER'] >= self.max_answer_candidates:
                if self.verbose:
                    print(f"\n[STOP] ANSWER has reached max candidates ({self.max_answer_candidates})")
                break
            
            if round_processed == 0:
                if self.verbose:
                    print(f"\n[STOP] No candidates processed in this round, terminating")
                break
            
            if self.verbose:
                print(f"\n[ROUND {iteration}] Processed {round_processed} candidates")
                print(f"[STATUS] Candidate counts: {dict(self.candidate_count)}")
        
        if self.verbose:
            print("\n" + "="*70)
            print("✅ GROUNDING COMPLETED")
            print("="*70)
            print("\nFinal candidate counts:")
            for entity, count in self.candidate_count.items():
                print(f"  {entity}: {count} candidates (processed: {len(self.processed_nodes[entity])})")
        
        return self.all_candidates


def run_priority_queue_grounding(
    query_obj,
    kb,
    vss_retriever,
    max_candidates_per_symbol: int = 1000,
    max_answer_candidates: int = 100,
    top_k_neighbors: int = 10,
    support_boost: float = 0.15,  # ADD THIS

    score_decay: float = 0.9,
    verbose: bool = True
) -> Dict[str, List['CandidateContext']]:
    """
    Convenience function to run priority queue-based grounding.
    
    Args:
        query_obj: Query class instance with entities, relations, and initial_symbol_candidates
        kb: Knowledge base object
        vss_retriever: VSSRetriever instance for scoring neighbors
        max_candidates_per_symbol: Max candidates per entity (default 1000)
        max_answer_candidates: Max candidates for ANSWER entity (default 100)
        top_k_neighbors: Number of top neighbors to add per expansion (default 10)
        score_decay: Score decay factor for propagation (default 0.9)
        verbose: Print detailed logs (default True)
    
    Returns:
        Dictionary mapping entity symbols to lists of CandidateContext objects
    """
    grounder = PriorityQueueGrounding(
        query_obj=query_obj,
        kb=kb,
        vss_retriever=vss_retriever,
        max_candidates_per_symbol=max_candidates_per_symbol,
        max_answer_candidates=max_answer_candidates,
        top_k_neighbors=top_k_neighbors,
        support_boost=support_boost,  # ADD THIS
        score_decay=score_decay,
        verbose=verbose
    )
    
    return grounder.ground()