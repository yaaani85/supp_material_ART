import numpy as np
from ogb.linkproppred import LinkPropPredDataset
def preprocess_ogbl_dataset(name: str, path: str, ds_out_path: str):
    print("Preprocessing OGBL dataset...")
    dataset = LinkPropPredDataset(name, root=path)
    split_edge = dataset.get_edge_split()
    train_triples, valid_triples, test_triples = split_edge["train"], split_edge["valid"], split_edge["test"]

    if name == 'ogbl-biokg':
        cur_idx, cur_type_idx, type_dict, entity_dict = 0, 0, {}, {}
        for key in dataset[0]['num_nodes_dict']:
            type_dict[key] = cur_type_idx
            cur_type_idx += 1
            entity_dict[key] = (cur_idx, cur_idx + dataset[0]['num_nodes_dict'][key])
            cur_idx += dataset[0]['num_nodes_dict'][key]



        def index_triples_across_type(triples, entity_dict, type_dict):
            triples['head_type_idx'] = np.zeros_like(triples['head'])
            triples['tail_type_idx'] = np.zeros_like(triples['tail'])
            for i in range(len(triples['head'])):
                h_type = triples['head_type'][i]
                triples['head_type_idx'][i] = type_dict[h_type]
                triples['head'][i] += entity_dict[h_type][0]
                if 'head_neg' in triples:
                    triples['head_neg'][i] += entity_dict[h_type][0]
                t_type = triples['tail_type'][i]
                triples['tail_type_idx'][i] = type_dict[t_type]
                triples['tail'][i] += entity_dict[t_type][0]
                if 'tail_neg' in triples:
                    triples['tail_neg'][i] += entity_dict[t_type][0]
            return triples

        print('Indexing triples across different entity types ...')
        train_triples = index_triples_across_type(train_triples, entity_dict, type_dict)
        valid_triples = index_triples_across_type(valid_triples, entity_dict, type_dict)
        test_triples = index_triples_across_type(test_triples, entity_dict, type_dict)
        other_data = {
            'train': np.concatenate([
                train_triples['head_type_idx'].reshape(-1, 1),
                train_triples['tail_type_idx'].reshape(-1, 1)
            ], axis=1),
            'valid': np.concatenate([
                valid_triples['head_neg'],
                valid_triples['tail_neg'],
                valid_triples['head_type_idx'].reshape(-1, 1),
                valid_triples['tail_type_idx'].reshape(-1, 1)
            ], axis=1),
            'test': np.concatenate([
                test_triples['head_neg'],
                test_triples['tail_neg'],
                test_triples['head_type_idx'].reshape(-1, 1),
                test_triples['tail_type_idx'].reshape(-1, 1)
            ], axis=1)
        }


    n_relations = int(max(train_triples['relation'])) + 1
    if name == 'ogbl-biokg':
        n_entities = sum(dataset[0]['num_nodes_dict'].values())
        assert train_triples['head'].max() <= n_entities
  
    print(f"{n_entities} entities and {n_relations} relations")

    train_array = np.concatenate([
        train_triples['head'].reshape(-1, 1),
        train_triples['relation'].reshape(-1, 1),
        train_triples['tail'].reshape(-1, 1)
    ], axis=1).astype(np.int64, copy=True)
    if other_data['train'] is not None:
        train_array = np.concatenate([train_array, other_data['train']], axis=1).astype(np.int64, copy=True)
    valid_array = np.concatenate([
        valid_triples['head'].reshape(-1, 1),
        valid_triples['relation'].reshape(-1, 1),
        valid_triples['tail'].reshape(-1, 1),
        other_data['valid']
    ], axis=1).astype(np.int64, copy=True)
    test_array = np.concatenate([
        test_triples['head'].reshape(-1, 1),
        test_triples['relation'].reshape(-1, 1),
        test_triples['tail'].reshape(-1, 1),
        other_data['test']
    ], axis=1).astype(np.int64, copy=True)

    triples = {'train': train_array, 'valid': valid_array, 'test': test_array}

 

    return triples, n_entities, n_relations




# class OGBBenchmarkCreator(KGEBenchmarkCreator):
#     def __init__(self, dataset_name: str, dataset_path: str, processed_path: str):
#         """
#         Initialize benchmark creator for OGB datasets.
#         Args:
#             dataset_name: Name of the OGB dataset (e.g., 'ogbl-biokg')
#             dataset_path: Path to store the raw OGB dataset
#             processed_path: Path to store the processed files
#         """
#         # First preprocess the OGB dataset
#         triples, n_entities, n_relations = preprocess_ogbl_dataset(
#             dataset_name, 
#             dataset_path, 
#             processed_path
#         )
        
#         self.train_triples = [tuple(t) for t in triples['train'][:, :3]]
#         self.valid_triples = [tuple(t) for t in triples['valid'][:, :3]]
#         self.test_triples = [tuple(t) for t in triples['test'][:, :3]]
        
#         # Initialize with empty 1D arrays instead of zero-dimensional arrays
#         self.candidates: DefaultDict[Tuple[int, int], npt.NDArray[np.int64]] = \
#             defaultdict(lambda: np.array([], dtype=np.int64))
#         self.seen_triples_mask: DefaultDict[Tuple[int, int], npt.NDArray[np.int64]] = \
#             defaultdict(lambda: np.array([], dtype=np.int64))
        
#         # Update how candidates are stored for test triples    
#         for i, (h, r, t) in enumerate(self.test_triples):
#             # Store head negatives directly as numpy array
#             self.candidates[(h, r)] = triples['test'][i, 3:503]
#             # Store tail negatives    
#             self.candidates[(t, r + n_relations)] = triples['test'][i, 503:1003]

#         # Initialize rest of the structures
#         self.n_entities = n_entities
#         self.n_relations = n_relations
#         self.filtered_forward = 0
#         self.filtered_inverse = 0
#         self.known_triples = defaultdict(set)
#         self.benchmark = defaultdict(lambda: np.array([], dtype=np.int64))
#         self.labels = defaultdict(lambda: np.array([], dtype=np.int64))
#         self.test_triples_dict = defaultdict(set)
        
#         # Fill known triples from train and valid
#         for h, r, t in self.train_triples + self.valid_triples:
#             self.known_triples[(h, r)].add(t)
#             self.known_triples[(t, r + self.n_relations)].add(h)
#         # Fill test triples dict
#         for h, r, t in self.test_triples:
#             self.test_triples_dict[(h, r)].add(t)
#             self.test_triples_dict[(t, r + self.n_relations)].add(h)
#         # After filling test_triples_dict
#         total_stored = sum(len(targets) for targets in self.test_triples_dict.values())
#         print(f"Total stored in test_triples_dict: {total_stored}")
#         print(f"Sample entries: {list(self.test_triples_dict.items())[:3]}")

#     def _load_triples(self, file_path: str, return_neg_samples: bool = False) -> Union[List[Tuple[int, int, int]], Tuple[List[Tuple[int, int, int]], Dict[Tuple[int, int], np.ndarray]]]:
#         """Load triples and optionally negative samples for OGB format."""
#         triples = []
#         neg_candidates = {}
        
#         # Load pickle file
#         with open(file_path, 'rb') as f:
#             data = pickle.load(f)
            
#         # Convert to triples
#         for triple in data:
#             # Access tuple elements directly
#             h_idx = triple[0]
#             r_idx = triple[1]
#             t_idx = triple[2]
#             triples.append((h_idx, r_idx, t_idx))
            
#             # Process negative samples if requested and available
#             if return_neg_samples:
#                 print(triple)
#                 # Get all 1000 negative samples (indices 3 through 1002)
#                 neg_samples = np.array(triple[3:1003])
#                 head_candidates = neg_samples[:500]
#                 tail_candidates = neg_samples[500:]
                
#                 # Store candidates for both directions
#                 neg_candidates[(h_idx, r_idx)] = tail_candidates
#                 neg_candidates[(t_idx, r_idx + self.n_relations)] = head_candidates
        
#         if return_neg_samples:
#             return triples, neg_candidates
#         return triples


#     def _get_valid_and_unseen_candidates(self, e: int, r: int) -> np.ndarray:
#         """Filter out candidates we've already processed."""
#         return np.setdiff1d(self.candidates[(e, r)], self.seen_triples_mask[(e, r)])

#     def _update_seen_mask(self, key: Tuple[int, int], values: Union[np.ndarray, int]) -> None:
#         """Helper to update seen mask with new values.
        
#         Args:
#             key: Tuple of (entity, relation)
#             values: Array of values or single value to add
#         """
#         if isinstance(values, (int, np.integer)):
#             values = np.array([values])
#         self.seen_triples_mask[key] = np.concatenate([
#             self.seen_triples_mask[key],
#             values
#         ])

#     def _add_candidates_to_seen(self, e: int, r: int, candidates: np.ndarray, direction: str) -> None:
#         """Mark candidates as seen for both directions."""
#         if direction == 'tail':
#             # Store forward direction
#             self._update_seen_mask((e, r), candidates)
#             # Store inverse direction
#             for c in candidates:
#                 self._update_seen_mask((c, r + self.n_relations), e)
#         else:
#             # Store inverse direction
#             self._update_seen_mask((e, r), candidates)
#             # Store forward direction
#             for c in candidates:
#                 self._update_seen_mask((c, r - self.n_relations), e)





# if __name__ == "__main__":
#     # Example usage:
#     """

#    d
#     """
#     creator = OGBBenchmarkCreator(
#         dataset_name='ogbl-biokg',
#         dataset_path='/home/yaaani85/Documents/projects/work/art_submission/data/ogbl_biokg',
#         processed_path='/home/yaaani85/Documents/projects/work/art_submission/data/ogbl_biokg/processed'
#     )
#     creator.save_benchmark('./ogbl_biokg/')