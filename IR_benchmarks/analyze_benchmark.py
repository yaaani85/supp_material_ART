import pickle
import os
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple
import matplotlib.pyplot as plt

class BenchmarkAnalyzer:
    def __init__(self, benchmark_dir: str):
        """
        Initialize analyzer with benchmark data.
        
        Args:
            benchmark_dir: Directory containing benchmark.pkl, ent2idx.pkl, rel2idx.pkl
        """
        # Load benchmark and mappings
        self.benchmark, self.ent2idx, self.rel2idx = self._load_data(benchmark_dir)
        self.n_relations = len(self.rel2idx)
        
        # Create inverse mappings
        self.idx2rel = {idx: rel for rel, idx in self.rel2idx.items()}
        self.idx2ent = {idx: ent for ent, idx in self.ent2idx.items()}
        
        print(f"Loaded {self.n_relations} relations")
        
    def _load_data(self, directory: str) -> Tuple[Dict, Dict, Dict]:
        """Load benchmark and mappings from directory."""
        with open(os.path.join(directory, 'benchmark.pkl'), 'rb') as f:
            benchmark = pickle.load(f)
        with open(os.path.join(directory, 'ent2idx.pkl'), 'rb') as f:
            ent2idx = pickle.load(f)
        with open(os.path.join(directory, 'rel2idx.pkl'), 'rb') as f:
            rel2idx = pickle.load(f)
            
        return benchmark, ent2idx, rel2idx
    
    def analyze_relation_distribution(self):
        """Analyze distribution of forward vs inverse relations."""
        forward_count = 0
        inverse_count = 0
        
        for h, r in self.benchmark.keys():
            # print(h, r)
            if r < self.n_relations:
                forward_count += 1
            else:
                inverse_count += 1
                
        print("\nRelation Direction Statistics:")
        print(f"Forward triples (r < {self.n_relations}): {forward_count}")
        print(f"Inverse triples (r >= {self.n_relations}): {inverse_count}")
        print(f"Total triples: {len(self.benchmark)}")
        
    def analyze_candidate_stats(self):
        """Analyze statistics about number of candidates per triple."""
        n_candidates = [len(v) for v in self.benchmark.values()]
        
        print("\nCandidate Statistics:")
        print(f"Min candidates: {min(n_candidates)}")
        print(f"Max candidates: {max(n_candidates)}")
        print(f"Mean candidates: {np.mean(n_candidates):.2f}")
        print(f"Median candidates: {np.median(n_candidates):.2f}")
        
        # Optional: Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(n_candidates, bins=50)
        plt.title('Distribution of Number of Candidates per Triple')
        plt.xlabel('Number of Candidates')
        plt.ylabel('Frequency')
        plt.savefig('candidate_distribution.png')
        plt.close()
        
    def analyze_relation_stats(self):
        """Analyze statistics per relation type."""
        rel_stats = defaultdict(lambda: {'count': 0, 'avg_candidates': 0, 'avg_positives': 0})
        
        for (h, r), v in self.benchmark.items():
            r_orig = r if r < self.n_relations else r - self.n_relations
            rel_name = self.idx2rel[r_orig]
            
            rel_stats[rel_name]['count'] += 1
            rel_stats[rel_name]['avg_candidates'] += len(v)
            
        # Compute averages
        print("\nPer-Relation Statistics:")
        print("\nTop 10 most frequent relations:")
        sorted_rels = sorted(rel_stats.items(), 
                           key=lambda x: x[1]['count'], 
                           reverse=True)
        
        for rel, stats in sorted_rels[:10]:
            avg_cand = stats['avg_candidates'] / stats['count']
            avg_pos = stats['avg_positives'] / stats['count']
            print(f"\nRelation: {rel}")
            print(f"Count: {stats['count']}")
            print(f"Avg candidates: {avg_cand:.2f}")
            print(f"Avg positives: {avg_pos:.2f}")
    
    def run_all_analyses(self):
        """Run all analysis methods and print results."""
        print("=== Benchmark Analysis ===")
        print(f"Total number of triples: {len(self.benchmark)}")
        print(f"Number of entities: {len(self.ent2idx)}")
        print(f"Number of relations: {len(self.rel2idx)}")
        
        self.analyze_relation_distribution()
        self.analyze_candidate_stats()
        self.analyze_relation_stats()

if __name__ == "__main__":
    # Example usage
    analyzer = BenchmarkAnalyzer('benchmarks/FB15k-237/')
    analyzer.run_all_analyses()