class CandidateContext:
    def __init__(self, node_id=None, entity=None, score=None):
        self.node_id = node_id
        self.entity = entity
        self.score = score
        self.triplets = []
        self.support = 0  # CHANGE: Start at 0, will be incremented on first path
        self.symbol_candidates = {}  # CHANGE: Start empty, will be added on first path
        

    def add_triplet(self, triplet):
        """Add a triplet without incrementing support (support tracks paths, not triplets)."""
        if triplet not in self.triplets:  # Avoid duplicates
            self.triplets.append(triplet)

    def add_symbol_candidates(self, symbol, candidate, score):
        """Add a path from a symbol's candidate. Each call represents a new path."""
        self.support += 1  # Increment support for each unique path
        
        if symbol not in self.symbol_candidates:
            self.symbol_candidates[symbol] = [(candidate, score)]
        else:
            self.symbol_candidates[symbol].append((candidate, score))

    def __str__(self):
        return (
            f"CandidateContext(\n"
            f"  node_id={self.node_id},\n"
            f"  entity={self.entity},\n"
            f"  score={self.score},\n"
            f"  support={self.support},\n"
            f"  triplets={self.triplets},\n"
            f"  symbol_candidates={self.symbol_candidates}\n"
            f")"
        )

    __repr__ = __str__