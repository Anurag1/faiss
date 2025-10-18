import numpy as np
import faiss
from faiss_symbolic import SymbolicFAISS

print("\n=== Symbolic FAISS CPU Demo ===")

# Create random dataset
vectors = np.random.random((1000, 64)).astype('float32')
queries = np.random.random((5, 64)).astype('float32')

# Build FAISS index
index = faiss.IndexFlatL2(64)
index.add(vectors)
print(f"Index built with {index.ntotal} vectors.")

# Initialize Symbolic Layer
sfaiss = SymbolicFAISS()

# Harmonize and quantize
centroids, q_vectors = sfaiss.harmonize_index(vectors)

# Run a symbolic-balanced search
D, I = sfaiss.run_search(index, queries, topk=5)

print("\nSearch Results (balanced):")
print(I)

# Simulate workload balancing
shards = [list(range(100)), list(range(80)), list(range(120))]
adjustments = sfaiss.balance_load(shards)
print("\nWorkload adjustments:", adjustments)
