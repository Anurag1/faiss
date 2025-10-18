from .geometry_harmonizer import GeometryHarmonizer
from .combinatorial_quantizer import CombinatorialQuantizer
from .harmony_search import HarmonySearch
from .coordination_bus import CoordinationBus

class SymbolicFAISS:
    def __init__(self):
        self.geo = GeometryHarmonizer()
        self.quant = CombinatorialQuantizer()
        self.search = HarmonySearch()
        self.bus = CoordinationBus()

    def harmonize_index(self, vectors):
        centroids = self.geo.generate_harmonic_centroids(vectors)
        q_vectors = self.quant.quantize(vectors, centroids)
        return centroids, q_vectors

    def run_search(self, index, query, topk=5):
        D, I = index.search(query, topk)
        return self.search.balance_results(D, I)

    def balance_load(self, shards):
        return self.bus.balance_workload(shards)
