from torch.nn import Module
import torch
from torch import Tensor
import os
import json
from typing import Dict, Tuple
import pickle
from torchdrug import datasets
from src.dataset import KGEDataset, OgbKGEDataset
from src.globals import NBF_TYPE_COUNTS, THIS_TYPE_COUNTS
import numpy as np

class NBF(Module):
    def __init__(self, config) -> None:
        super(NBF, self).__init__()
        # Load model
        from external_models.nbfnet import util, dataset, model, layer, task
        from torchdrug import core
        vars = {'gpus': [0]}
        
        cfg = util.load_config(config['config'], context=vars)
        print(config)
        dataset = core.Configurable.load_config_dict(cfg.dataset)
        if torch.cuda.is_available():
            cfg['gpus'] = [0]
        else:
            cfg['gpus'] = None
        cfg['checkpoint'] = config['model_path']

        solver = util.build_solver(cfg, dataset)
        self.nbf_model = solver.model
        self.config = config
        self.dataset = self._load_dataset()
        
        # Set properties
        #self.nbf_model = solver.model
        self.number_of_entities = config['n_entities']
        self.n_ent = config['n_entities']
        self.n_relations = config['n_relations']
        self.dataset_type = config['dataset']['class']
        self.processed_dir = os.path.join(config['dataset']['path'], 'ogbl_biokg', 'processed')

        self.cache_file = os.path.join(self.processed_dir, f'nbf_scores_labels.pkl')
        # Load benchmark mappings
        benchmark_dir = os.path.join('./data/kbc_benchmarks', config['dataset']['class'].lower())

        # Get NBF's internal mappings
        self.nbf_entity_dict, self.nbf_relation_dict = self._load_or_create_mappings(config)
        
        if config['dataset']['class'].lower() == "ogblbiokg":
            # For OGB-biokg, we directly use the entity_dict from _load_or_create_mappings
            self.entity_dict = self.nbf_entity_dict
            self.relation_dict = self.nbf_relation_dict
        else:
            with open(os.path.join(benchmark_dir, 'ent2idx.pkl'), 'rb') as f:
                self.benchmark_ent2idx = pickle.load(f)
            with open(os.path.join(benchmark_dir, 'rel2idx.pkl'), 'rb') as f:
                self.benchmark_rel2idx = pickle.load(f)
            # For other datasets, create mapping from benchmark IDs to NBF IDs
            # We need to go: benchmark_id -> entity_name -> nbf_id
            self.entity_dict = {
                benchmark_id: self.nbf_entity_dict[entity_name] 
                for entity_name, benchmark_id in self.benchmark_ent2idx.items()
            }
            self.relation_dict = {
                benchmark_id: self.nbf_relation_dict[relation_name]
                for relation_name, benchmark_id in self.benchmark_rel2idx.items()
            }
        
        # Print mappings to debug
        print("\nFinal mappings (first 5 entries):")
        print("Entity mapping:", dict(list(self.entity_dict.items())[:5]))
        print("Relation mapping:", dict(list(self.relation_dict.items())[:5]))

    def _load_or_create_mappings(self, config) -> Tuple[Dict, Dict]:
        """Load or create NBF mappings for entity and relation IDs"""
        from torchdrug import datasets
        
        if config['dataset']['class'].lower() == "ogblbiokg":
            from src.globals import NBF_TYPE_COUNTS, THIS_TYPE_COUNTS
            
            # Create offset dictionaries for both mappings
            nbf_offsets = self._create_offset_dict(NBF_TYPE_COUNTS)
            this_offsets = self._create_offset_dict(THIS_TYPE_COUNTS)
            
            # Create mapping dictionary
            entity_dict = {}
            for entity_type, count in THIS_TYPE_COUNTS.items():
                for i in range(count):
                    original_idx = this_offsets[entity_type] + i
                    nbf_idx = nbf_offsets[entity_type] + i
                    entity_dict[original_idx] = nbf_idx
            
            # Simple 1:1 mapping for relations as they should align
            relation_dict = {i: i for i in range(config['n_relations'])}
            
            # Print first few entries to debug
            print("NBF entity mapping (first 5):", dict(list(entity_dict.items())[:5]))
            print("NBF relation mapping (first 5):", dict(list(relation_dict.items())[:5]))
            
        elif config['dataset']['class'].lower() == "fb15k237":
            dataset = datasets.FB15k237(config['dataset']['path'])
            entity_dict = {name: int(idx) for idx, name in enumerate(dataset.entity_vocab)}
            relation_dict = {name: int(idx) for idx, name in enumerate(dataset.relation_vocab)}
        
        elif config['dataset']['class'].lower() == "wn18rr":
            dataset = datasets.WN18RR(config['dataset']['path'])
            entity_dict = {name: idx for idx, name in enumerate(dataset.entity_vocab)}
            relation_dict = {name: idx for idx, name in enumerate(dataset.relation_vocab)}
        
        else:
            raise ValueError(f"Unknown dataset: {config['dataset']['class']}")
        
        return entity_dict, relation_dict

    def _convert_ids(self, ids: torch.Tensor, mapping: Dict) -> torch.Tensor:
        """Convert IDs using mapping dictionary"""
        return torch.tensor([mapping[id.item()] for id in ids], 
                          device=ids.device, 
                          dtype=ids.dtype)

    def _convert_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Convert scores back to original ID space"""
        # Since we're using same ID space now, no conversion needed
        return scores

    def inference_tail_prediction_ogb(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, candidates_mask: torch.Tensor = None) -> torch.Tensor:
        """OGB-specific implementation of tail prediction"""
        device = h.device
        batch_size = len(h)
        scores = torch.full((batch_size, self.n_ent), float('-inf'), device=device)
        
        # If no candidates_mask provided, use all entities as candidates
        if candidates_mask is None:
            candidates_mask = torch.ones((batch_size, self.n_ent), dtype=torch.bool, device=device)

        # Get number of candidates per query and find valid queries
        cands_per_query = candidates_mask.sum(dim=1)
        valid_queries = candidates_mask.any(dim=1).nonzero().squeeze(-1)
        if len(valid_queries) == 0:
            return scores
        max_cands = cands_per_query.max().item()
        
        # Create batched triples where each sub-batch has same head
        batched_triples = []
        for qid in valid_queries:
            valid_candidates = candidates_mask[qid].nonzero().squeeze(-1)
            n_cands = len(valid_candidates)
            
            # Create triples for this query
            h_repeat = h[qid].repeat(max_cands)  # Same head for all candidates
            r_repeat = r[qid].repeat(max_cands)
            
            # Pad candidates if needed
            if n_cands < max_cands:
                t_cand = torch.cat([valid_candidates, valid_candidates[-1].repeat(max_cands - n_cands)])
            else:
                t_cand = valid_candidates
            
            query_triples = torch.stack([h_repeat, t_cand, r_repeat], dim=1)
            batched_triples.append(query_triples)
        
        # Stack all batches
        valid_triples = torch.stack(batched_triples, dim=0)  # Shape: [n_valid_queries, max_cands, 3]
        
        with torch.no_grad():
            batch_scores = self.nbf_model.predict(valid_triples)
            
            # Assign scores back to the correct positions
            for batch_idx, original_idx in enumerate(valid_queries):
                valid_candidates = candidates_mask[original_idx].nonzero().squeeze(-1)
                scores[original_idx, valid_candidates] = batch_scores[batch_idx, :len(valid_candidates)]
        
        return scores

    def inference_head_prediction_ogb(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, candidates_mask: torch.Tensor = None) -> torch.Tensor:
        """OGB-specific implementation of head prediction"""
        device = t.device
        batch_size = len(t)
        scores = torch.full((batch_size, self.n_ent), float('-inf'), device=device)
        
        # If no candidates_mask provided, use all entities as candidates
        if candidates_mask is None:
            candidates_mask = torch.ones((batch_size, self.n_ent), dtype=torch.bool, device=device)

        # Get number of candidates per query and find valid queries
        cands_per_query = candidates_mask.sum(dim=1)
        valid_queries = candidates_mask.any(dim=1).nonzero().squeeze(-1)
        if len(valid_queries) == 0:
            return scores
        max_cands = cands_per_query.max().item()
        
        # Create batched triples where each sub-batch has same tail
        batched_triples = []
        for qid in valid_queries:
            valid_candidates = candidates_mask[qid].nonzero().squeeze(-1)
            n_cands = len(valid_candidates)
            
            # Create triples for this query
            t_repeat = t[qid].repeat(max_cands)  # Same tail for all candidates
            r_repeat = r[qid].repeat(max_cands)
            
            # Pad candidates if needed
            if n_cands < max_cands:
                h_cand = torch.cat([valid_candidates, valid_candidates[-1].repeat(max_cands - n_cands)])
            else:
                h_cand = valid_candidates
            
            query_triples = torch.stack([h_cand, t_repeat, r_repeat], dim=1)
            batched_triples.append(query_triples)
        
        # Stack all batches
        valid_triples = torch.stack(batched_triples, dim=0)  # Shape: [n_valid_queries, max_cands, 3]
        
        with torch.no_grad():
            batch_scores = self.nbf_model.predict(valid_triples)
            
            # Assign scores back to the correct positions
            for batch_idx, original_idx in enumerate(valid_queries):
                valid_candidates = candidates_mask[original_idx].nonzero().squeeze(-1)
                scores[original_idx, valid_candidates] = batch_scores[batch_idx, :len(valid_candidates)]
        
        return scores

    def inference_tail_prediction(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, candidates_mask=None) -> torch.Tensor:
        """Main interface for tail prediction"""
        if self.config['dataset']['class'].lower() == 'ogblbiokg':
            return self.inference_tail_prediction_ogb(h, r, t, candidates_mask)
        else:
            # Original code for other datasets
            batch = torch.cat((h.unsqueeze(1), t.unsqueeze(1), r.unsqueeze(1)), dim=1)
            with torch.no_grad():
                self.scores = self.nbf_model.predict(batch)
                scores = self.scores[:, 0, :]
                if candidates_mask is not None:
                    scores[~candidates_mask] = float('-inf')
                return scores

    def inference_head_prediction(self, h: Tensor, r: Tensor, t: Tensor, candidates_mask=None) -> Tensor:
        """Main interface for head prediction"""
        if self.config['dataset']['class'].lower() == 'ogblbiokg':
            return self.inference_head_prediction_ogb(h, r, t, candidates_mask)
        else:
            # Use the head scores from the previous forward pass
            scores = self.scores[:, 1, :]
            if candidates_mask is not None:
                scores[~candidates_mask] = float('-inf')
            return scores

   
    def _create_offset_dict(self, type_counts):
        """Create dictionary of offsets for each entity type"""
        offset_dict = {}
        current_offset = 0
        for entity_type, count in type_counts.items():
            offset_dict[entity_type] = current_offset
            current_offset += count
        return offset_dict

    def _load_dataset(self):
        """Load dataset for ARM models"""
        dataset_class = self.config['dataset']['class'].lower()
        
        if dataset_class == "fb15k237":
            return KGEDataset(self.config)
        elif dataset_class == "wn18rr":
            return KGEDataset(self.config)
        elif dataset_class == "ogblbiokg":
            return OgbKGEDataset(self.config)
        else:
            raise ValueError(f"Unknown dataset for ARM models: {dataset_class}")

    def convert_benchmark_dict(self, benchmark_dict):
        """Convert benchmark dictionary from benchmark IDs (sorted) to NBF IDs (vocab order)"""
        if self.config['dataset']['class'].lower() == "ogblbiokg":
            # Create offset dictionaries for both mappings
            nbf_offsets = self._create_offset_dict(NBF_TYPE_COUNTS)
            this_offsets = self._create_offset_dict(THIS_TYPE_COUNTS)
            
            def convert_id(idx):
                for entity_type, offset in this_offsets.items():
                    if idx >= offset and idx < offset + THIS_TYPE_COUNTS[entity_type]:
                        return idx - this_offsets[entity_type] + nbf_offsets[entity_type]
                raise ValueError(f"Index {idx} is out of range")
            
            converted_dict = {}
            for (h, r, t), (head_data, tail_data) in benchmark_dict.items():
                new_h = convert_id(h)
                new_r = r  # Relations are 1:1 mapped
                new_t = convert_id(t)
                
                head_candidates, head_labels = head_data
                tail_candidates, tail_labels = tail_data
                
                # Convert while maintaining numpy array format
                new_head_candidates = np.array([convert_id(c) for c in head_candidates], dtype=np.int64)
                new_tail_candidates = np.array([convert_id(c) for c in tail_candidates], dtype=np.int64)
                
                converted_dict[(new_h, new_r, new_t)] = (
                    (new_head_candidates, head_labels),
                    (new_tail_candidates, tail_labels)
                )
            
            return converted_dict
        
        else:
            # Original code for other datasets
            from torchdrug import datasets
            if self.dataset_type.lower() == "fb15k237":
                dataset = datasets.FB15k237(self.config['dataset']['path'])
            else:
                dataset = datasets.WN18RR(self.config['dataset']['path'])
            
            # Get both mappings
            # NBF mapping: name -> nbf_id (vocab order)
            nbf_entity_dict = {string: int(i) for i, string in enumerate(dataset.entity_vocab)}
            nbf_relation_dict = {string: int(i) for i, string in enumerate(dataset.relation_vocab)}
            
            # Benchmark mapping: name -> bench_id (sorted order)
            # We already have this in self.benchmark_ent2idx and self.benchmark_rel2idx
            
            # Create mapping from benchmark IDs to NBF IDs
            bench_to_nbf_ent = {}
            for name in dataset.entity_vocab:
                bench_id = self.benchmark_ent2idx[name]
                nbf_id = nbf_entity_dict[name]
                bench_to_nbf_ent[bench_id] = nbf_id
            
            bench_to_nbf_rel = {}
            for name in dataset.relation_vocab:
                bench_id = self.benchmark_rel2idx[name]
                nbf_id = nbf_relation_dict[name]
                bench_to_nbf_rel[bench_id] = nbf_id
            
            # Print first few mappings for debugging
            print("\nFirst few entity mappings (bench_id -> nbf_id):")
            print(dict(list(bench_to_nbf_ent.items())[:5]))
            print("\nFirst few relation mappings (bench_id -> nbf_id):")
            print(dict(list(bench_to_nbf_rel.items())[:5]))
            
            # Convert using these mappings
            converted_dict = {}
            for (h, r, t), (head_data, tail_data) in benchmark_dict.items():
                new_h = bench_to_nbf_ent[h]
                new_r = bench_to_nbf_rel[r]
                new_t = bench_to_nbf_ent[t]
                
                head_candidates, head_labels = head_data
                tail_candidates, tail_labels = tail_data
                
                new_head_candidates = [bench_to_nbf_ent[c] for c in head_candidates]
                new_tail_candidates = [bench_to_nbf_ent[c] for c in tail_candidates]
                
                converted_dict[(new_h, new_r, new_t)] = (
                    (new_head_candidates, head_labels),
                    (new_tail_candidates, tail_labels)
                )
            
            return converted_dict