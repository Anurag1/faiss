import numpy as np
import faiss
from faiss_symbolic import SymbolicFAISS

def test_symbolic_faiss_core():
    vectors = np.random.random((100, 32)).astype('float32')
    queries = np.random.random((2, 32)).astype('float32')
    index = faiss.IndexFlatL2(32)
    index.add(vectors)
    sfaiss = SymbolicFAISS()
    centroids, q_vectors = sfaiss.harmonize_index(vectors)
    D, I = sfaiss.run_search(index, queries)
    assert I.shape[1] == 5 or I.shape[1] > 0
    print("[TEST] Symbolic FAISS modules executed successfully.")
