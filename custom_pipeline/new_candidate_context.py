class CandidateContext:
    def __init__(self, node_id, entity, initial_score):
        self.node_id = node_id
        self.entity = entity
        self.initial_score = initial_score  # VSS/BM25, never changes
        self.current_score = initial_score  # Max of incoming decayed scores
        self.support = 0
        
        # Store complete paths
        self.paths = []  # [{entities: {A: node_a, B: node_b}, scores: {A: score_a, B: score_b}, relations: [(A, rel, B)]}, ...]
        self.triplets = set()  # {(source_entity, relation, target_entity), ...}
    
    def add_path(self, path_dict):
        """Add a complete path to this candidate."""
        self.paths.append(path_dict)
        self.support += 1
        
        # Extract triplets
        for triplet in path_dict['relations']:
            self.triplets.add(triplet)
    
    def prune_paths(self, max_paths=50):
        """Keep only top max_paths by product of VSS scores."""
        if len(self.paths) <= max_paths:
            return
        
        # Score each path by product of all scores
        path_scores = []
        for i, path in enumerate(self.paths):
            score_product = 1.0
            for entity_score in path['scores'].values():
                score_product *= entity_score
            path_scores.append((score_product, i))
        
        # Sort by score descending and keep top max_paths
        path_scores.sort(reverse=True, key=lambda x: x[0])
        top_indices = {idx for _, idx in path_scores[:max_paths]}
        
        self.paths = [self.paths[i] for i in sorted(top_indices)]
        self.support = len(self.paths)
        
    def __str__(self):
        def __str__(self):
            init = f"{self.initial_score:.4f}" if isinstance(self.initial_score, (int, float)) else repr(self.initial_score)
            cur = f"{self.current_score:.4f}" if isinstance(self.current_score, (int, float)) else repr(self.current_score)
            return (
                f"CandidateContext(node_id={self.node_id}, entity={repr(self.entity)}, "
                f"initial_score={init}, current_score={cur}, support={self.support}, "
                f"paths={len(self.paths)}, triplets={len(self.triplets)})"
            )