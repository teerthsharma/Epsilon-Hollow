import numpy as np

class DifferentiableMemory:
    """Combines vector indices for episodic memory and knowledge graphs for semantic memory."""
    def __init__(self, dim: int):
        self.dim = dim
        self.episodic_store = []
        self.semantic_graph = {}

    def store_experience(self, vector: np.ndarray, metadata: dict):
        self.episodic_store.append({"embedding": vector, "meta": metadata})

    def retrieve(self, query_vector: np.ndarray, k=5) -> list:
        if not self.episodic_store: return []
        scores = [np.dot(query_vector, item["embedding"]) for item in self.episodic_store]
        top_k_indices = np.argsort(scores)[-k:]
        return [self.episodic_store[i] for i in top_k_indices]
