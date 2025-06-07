import torch
import numpy as np
from collections import defaultdict
import pickle
import os
from tqdm import tqdm
from typing import Dict, Set, Tuple, List, DefaultDict, Union
import numpy.typing as npt
from preprocess_ogb import preprocess_ogbl_dataset
import json
from torchkge.data_structures import KnowledgeGraph

class KGEBenchmarkCreator:
    def __init__(self, train_path: str, valid_path: str, test_path: str):
        """Regular initialization for standard KG datasets"""
        # Get mappings first
        self.ent2idx, self.rel2idx, self.idx2ent, self.idx2rel = self._get_mappings(
            train_path, valid_path, test_path
        )

        self.n_entities = len(self.ent2idx)
        self.n_relations = len(self.rel2idx)

        # Load all splits but don't combine them yet
        self.train_triples = self._load_triples(train_path)
        self.valid_triples = self._load_triples(valid_path)
        self.test_triples = self._load_triples(test_path)

        self.has_candidates = False
        self.filtered_candidates_count = 0

        # Add this to your init to define the mappings

    def _init_structures(self, split: str) -> None:
        """Initialize structures for specific benchmark creation"""
        # Initialize data structures
        self.seen_tails = defaultdict(set)
        self.seen_heads = defaultdict(set)
        self.known_heads = defaultdict(set)
        self.known_tails = defaultdict(set)
        self.benchmark = defaultdict(lambda: np.array([], dtype=np.int64))
        self.labels = defaultdict(lambda: np.array([], dtype=np.int64))
        self.target_triples_dict = defaultdict(set)
        self.head_candidates = defaultdict(set)
        self.tail_candidates = defaultdict(set)

        # Set target triples and known triples based on split
        if split == "test":
            target_triples = self.test_triples
            known_triples = np.concatenate([self.train_triples, self.valid_triples], axis=0)
            if self.has_candidates:
                head_negatives = self._test_head_negatives
                tail_negatives = self._test_tail_negatives
        elif split == "validation":
            target_triples = self.valid_triples
            known_triples = np.concatenate([self.train_triples, self.test_triples], axis=0)
            if self.has_candidates:
                head_negatives = self._valid_head_negatives
                tail_negatives = self._valid_tail_negatives
        else:
            raise ValueError("Split must be either 'test' or 'validation'")

        # Fill known triples
        for h, r, t in known_triples:
            self.known_heads[(t, r)].add(h)
            self.known_tails[(h, r)].add(t)

        # Fill test triples dict and candidates for target split
        for i, (h, r, t) in enumerate(target_triples):
            self.target_triples_dict[(h, r)].add(t)
            self.target_triples_dict[(t, r + self.n_relations)].add(h)

        if self.has_candidates:
            for i, (h, r, t) in enumerate(target_triples):
                # Add candidates if we have OGB negatives
                self.tail_candidates[(h, r, i)].add(t)
                # add true triples in other direction
                self.tail_candidates[(h, r, i)].update(tail_negatives[i])
                self.tail_candidates[(h, r, i)].update(self.target_triples_dict[(h, r)])
                # add true triples in other direction
                self.head_candidates[(t, r, i)].update(self.target_triples_dict[(t, r+self.n_relations)])
                self.head_candidates[(t, r, i)].update(head_negatives[i])
                self.head_candidates[(t, r, i)].add(h)

        self._direction_maps = {
            "tail": {
                'candidates': self.tail_candidates,
                'seen': self.seen_tails,
                'known': self.known_tails
            },
            "head": {
                'candidates': self.head_candidates,
                'seen': self.seen_heads,
                'known': self.known_heads
            }
        }
        return target_triples

    def _get_mappings(self, train_path: str, valid_path: str, test_path: str):
        """Build entity and relation mappings using same logic as TorchKGE."""
        entities = set()
        relations = set()

        # Collect unique entities and relations
        for filepath in [train_path, valid_path, test_path]:
            with open(filepath, 'r', encoding='utf-8') as f:  # Explicitly specify encoding
                for line in f:
                    h, r, t = line.strip().split('\t')
                    entities.update([h, t])
                    relations.add(r)

        # Create sorted mappings
        ent2idx = {ent: i for i, ent in enumerate(sorted(entities))}
        rel2idx = {rel: i for i, rel in enumerate(sorted(relations))}

        # Create reverse mappings
        idx2ent = {i: ent for i, ent in enumerate(sorted(entities))}
        idx2rel = {i: rel for i, rel in enumerate(sorted(relations))}

        return ent2idx, rel2idx, idx2ent, idx2rel

    def _load_triples(self, file_path: str) -> List[Tuple[int, int, int]]:
        """Load triples from text file and convert to integer indices using existing mappings."""
        triples = []

        with open(file_path, 'r') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                # Convert to integer indices using existing mappings
                h_idx = self.ent2idx[h]
                r_idx = self.rel2idx[r]
                t_idx = self.ent2idx[t]
                triples.append((h_idx, r_idx, t_idx))

        return triples

    def _add_candidates_to_benchmark(self, e: int, r: int, qid: int, candidates: np.ndarray) -> None:
        self.benchmark[(e, r, qid)] = np.concatenate([self.benchmark[(e, r, qid)], candidates])
        self.labels[(e, r, qid)] = np.concatenate([self.labels[(e, r, qid)], self._get_labels_for_candidates(e, r, candidates)])

    def _get_labels_for_candidates(self, e: int, r: int, candidates: np.ndarray) -> np.ndarray:
        """Get binary labels for candidates based on test set."""
        true_targets = self.target_triples_dict[(e, r)]
        return np.array([1 if c in true_targets else 0 for c in set(candidates)], dtype=np.int64)

    def _get_unseen_candidates(self, e1: int, r: int, qid: int, direction: str) -> np.ndarray:
        """Get candidates that haven't been seen yet for either head or tail prediction."""
        maps = self._direction_maps[direction]

        candidates = set(maps['candidates'][(e1, r, qid)]) if self.has_candidates else set(np.arange(self.n_entities))
        seen = maps['seen'][(e1, r)]
        known = maps['known'][(e1, r)]

        self.filtered_candidates_count += len(seen) + len(known)
        return np.array(list(candidates - seen - known))

    def _add_to_seen(self, e1: int, r: int, candidates: np.ndarray, direction: str) -> None:
        """Add candidates to seen sets for either head or tail prediction."""
        maps = self._direction_maps[direction]
        opposite_direction = "head" if direction == "tail" else "tail"
        opposite_maps = self._direction_maps[opposite_direction]

        maps['seen'][(e1, r)].update(candidates)
        for c in candidates:
            opposite_maps['seen'][(c, r)].add(e1)

    def process_triple(self, h: int, r: int, t: int, triple_idx: int) -> None:
        """Process both forward and inverse triples."""
        r_inv = r + self.n_relations

        # Rest of code proceeds normally for both OGB and regular KGs
        # Tail prediction
        

        unseen_candidates = self._get_unseen_candidates(h, r, qid=triple_idx, direction="tail")
        if len(unseen_candidates) > 0:
            self._add_candidates_to_benchmark(h, r, qid=triple_idx, candidates=unseen_candidates)
            self._add_to_seen(h, r, unseen_candidates, "tail")

        unseen_candidates = self._get_unseen_candidates(t, r, qid=triple_idx, direction="head")
        if len(unseen_candidates) > 0:
            self._add_candidates_to_benchmark(t, r_inv, qid=triple_idx, candidates=unseen_candidates)
            self._add_to_seen(t, r, unseen_candidates, "head")
        # Head prediction (inverse)

    def _load_triples_from_kg(self, kg: KnowledgeGraph) -> np.ndarray:
        """Convert TorchKGE KnowledgeGraph to numpy array of triples."""
        return np.column_stack([
            kg.head_idx.numpy(),
            kg.relations.numpy(),
            kg.tail_idx.numpy()
        ])

    @classmethod
    def from_ogb(cls, dataset_name: str, dataset_path: str, processed_path: str) -> 'KGEBenchmarkCreator':
        """Initialize benchmark creator from OGB dataset."""
        instance = cls.__new__(cls)
        instance.processed_dir = os.path.join("../data", 'ogbl_biokg', 'processed')

        # Load KGs from pickle files
        print("Loading KGs from pickle files...")
        with open(os.path.join(instance.processed_dir, 'kg_train.pkl'), 'rb') as f:
            instance.kg_train = pickle.load(f)
        with open(os.path.join(instance.processed_dir, 'kg_val.pkl'), 'rb') as f:
            instance.kg_val, val_neg_samples = pickle.load(f)  # val_neg_samples is already torch.LongTensor
        with open(os.path.join(instance.processed_dir, 'kg_test.pkl'), 'rb') as f:
            instance.kg_test, test_neg_samples = pickle.load(f)  # test_neg_samples is already torch.LongTensor

        # Convert KGs to triples
        print("Converting KGs to triples...")
        instance.train_triples = instance._load_triples_from_kg(instance.kg_train)
        instance.valid_triples = instance._load_triples_from_kg(instance.kg_val)
        instance.test_triples = instance._load_triples_from_kg(instance.kg_test)
        instance.n_entities = instance.kg_train.n_ent
        instance.n_relations = instance.kg_train.n_rel
        instance.has_candidates = True

        del instance.kg_train
        del instance.kg_val
        del instance.kg_test
        print("Converted KGs to triples")
        # Initialize OGB structures with the negative samples (already torch.LongTensor)
        instance._init_ogb_structures(val_neg_samples, test_neg_samples)
        print("Initialized OGB structures")
        return instance

    def _init_ogb_structures(self, val_neg_samples: torch.Tensor, test_neg_samples: torch.Tensor):
        """Initialize structures specific to OGB datasets."""
        self.num_candidates = 501  # 500 negatives + 1 true

        # Convert to numpy arrays since rest of code uses numpy
        self._valid_head_negatives = val_neg_samples[:, :500].numpy()  # First 500 for head
        self._valid_tail_negatives = val_neg_samples[:, 500:].numpy()  # Last 500 for tail
        self._test_head_negatives = test_neg_samples[:, :500].numpy()
        self._test_tail_negatives = test_neg_samples[:, 500:].numpy()

    def create_benchmark(self, split: str) -> Dict:
        """Process all test triples and create benchmark dataset."""
        self.filtered_candidates_count = 0  # Reset counter

        self.target_triples = self._init_structures(split)

        for idx, (h, r, t) in enumerate(tqdm(self.target_triples, desc="Creating benchmark")):
            self.process_triple(h, r, t, idx)

        # Get statistics and perform health checks
        statistics = self.check_health_and_get_statistics()

        # Print key statistics
        print("\nBenchmark Statistics:")
        for key, value in statistics.items():
            print(f"{key}: {value}")

        return statistics

    def save_benchmark(self, output_dir: str, split: str) -> None:
        """Save benchmark, labels, mappings and original test queries to directory."""
        os.makedirs(output_dir, exist_ok=True)

        # First create benchmark to ensure all processing is done
        statistics = self.create_benchmark(split)

        # Save statistics
        import json
        with open(os.path.join(output_dir, f'{split}_statistics.json'), 'w') as f:
            json.dump(statistics, f, indent=4)
        print("saved statistics")

        # Save mappings
        # with open(os.path.join(output_dir, 'ent2idx.pkl'), 'wb') as f:
        #     pickle.dump(self.ent2idx, f)
        # with open(os.path.join(output_dir, 'rel2idx.pkl'), 'wb') as f:
        #     pickle.dump(self.rel2idx, f)
        # print("saved ids")

        # Create benchmark dictionary
        benchmark_dict = {}
        seen_keys = set()  # Track which (e,r) pairs we've already saved
        empty_data = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))

        for i, (h, r, t) in enumerate(self.target_triples):
            # Get tail prediction data
            if (h, r, i) in self.benchmark:
                tail_data = (self.benchmark[(h, r, i)], self.labels[(h, r, i)])
            else:
                tail_data = empty_data

            # Get head prediction data
            if (t, r + self.n_relations, i) in self.benchmark:
                head_data = (self.benchmark[(t, r + self.n_relations, i)], self.labels[(t, r + self.n_relations, i)])
            else:
                head_data = empty_data

            # Store data for this triple
            benchmark_dict[(h, r, t)] = (head_data, tail_data)

        # Save benchmark dictionary
        with open(os.path.join(output_dir, f'{split}_benchmark.pkl'), 'wb') as f:
            pickle.dump(benchmark_dict, f)
        print("saved benchmark")

    def check_health_and_get_statistics(self) -> Dict:
        """Perform health checks and collect statistics about the benchmark."""
        # Calculate basic statistics
        total_candidates = sum(len(candidates) for candidates in self.benchmark.values())
        total_labels = sum(len(labels) for labels in self.labels.values())
        total_true_labels = sum(np.sum(labels) for labels in self.labels.values())
        n_test_triples = len(self.target_triples)

        # Count forward vs inverse true labels
        forward_true_labels = 0
        inverse_true_labels = 0
        forward_queries = 0
        inverse_queries = 0

        for (e, r, qid), labels in self.labels.items():
            if r < self.n_relations:  # Forward/tail prediction
                forward_true_labels += np.sum(labels)
                forward_queries += 1
            else:  # Inverse/head prediction
                inverse_true_labels += np.sum(labels)
                inverse_queries += 1

        # For OGB datasets with candidates, expect double true labels
        expected_true_labels = n_test_triples
        # assert total_true_labels == expected_true_labels, \
        #     f"Mismatch between true labels ({total_true_labels}) and expected ({expected_true_labels})"
        # assert self.benchmark.keys() == self.labels.keys(), \
        #     "Mismatch between benchmark and labels keys"
        # assert all(len(self.benchmark[k]) == len(self.labels[k]) for k in self.benchmark.keys()), \
        #     "Mismatch between benchmark candidates and labels lengths"

        # Calculate and verify candidate counts
        # if self.has_candidates:
        #     total_possible_candidates = len(self.target_triples) * 2 * self.num_candidates
        # else:
        #     total_possible_candidates = len(self.target_triples) * 2 * self.n_entities

        # expected_candidates = total_possible_candidates - self.filtered_candidates_count
        # assert total_candidates == expected_candidates, \
        #     f"Mismatch in candidate count: got {total_candidates}, expected {expected_candidates}"

        # Collect all statistics in a dictionary
        statistics = {
            "total_true_labels": int(total_true_labels),
            "total_false_labels": int(total_labels - total_true_labels),
            "forward_true_labels": int(forward_true_labels),
            "inverse_true_labels": int(inverse_true_labels),
            "total_candidates": total_candidates,
            "num_queries": len(self.benchmark),
            "average_candidates_per_query": float(total_candidates/len(self.benchmark)),
            "forward_queries": forward_queries,
            "inverse_queries": inverse_queries
        }

        return statistics
