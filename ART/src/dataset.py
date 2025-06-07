import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchkge.data_structures import KnowledgeGraph
from abc import ABC, abstractmethod


from src.utils import preprocess_ogbl_dataset, get_dict_of_tails_and_heads, load_dataset, get_dict_of_tails_and_heads
from src.globals import NBF_TYPE_COUNTS, THIS_TYPE_COUNTS


class BaseDataset(Dataset, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_dataloader(self, batch_size: int, split: str = 'train', num_workers: int = 0, persistent_workers: bool = False, sample_size: int = None):
        pass

    @property
    @abstractmethod
    def n_relations(self):
        pass

    @property
    @abstractmethod
    def dict_of_tails_train_val(self):
        pass

    @property
    @abstractmethod
    def dict_of_heads_train_val(self):
        pass

    @property
    @abstractmethod
    def dict_of_tails_test(self):
        pass

    @property
    @abstractmethod
    def dict_of_heads_test(self):
        pass

    @property
    @abstractmethod
    def n_entities(self):
        pass

class KGEDataset(BaseDataset):
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.reciprocal = config.get('reciprocal', False)
        self.kg_train, self.kg_val, self.kg_test = load_dataset(config)
        self._dict_of_tails_train_val, self._dict_of_heads_train_val = get_dict_of_tails_and_heads([self.kg_train, self.kg_val])
        self._dict_of_tails_test, self._dict_of_heads_test = get_dict_of_tails_and_heads([self.kg_test])
        self._dict_of_tails_val, self._dict_of_heads_val = get_dict_of_tails_and_heads([self.kg_val])

    def get_dataloader(self, batch_size: int, split: str = 'train', num_workers: int = 0, persistent_workers: bool = False, sample_size: int = None):
        if split == 'train':
            kg = self.kg_train
        elif split == 'valid':
            kg = self.kg_val
        elif split == 'test':
            kg = self.kg_test
        else:
            raise ValueError(f"Unknown split: {split}")

        return DataLoader(
            torch.utils.data.TensorDataset(kg.head_idx, kg.relations, kg.tail_idx),
            batch_size=batch_size, shuffle=(split == 'train'),
            num_workers=num_workers, persistent_workers=persistent_workers
        )

    @property
    def dict_of_tails_train_val(self):
        return self._dict_of_tails_train_val

    @property
    def dict_of_heads_train_val(self):
        return self._dict_of_heads_train_val

    @property
    def n_entities(self):
        return self.kg_train.n_ent

    @property
    def dict_of_tails_val(self):
        return self._dict_of_tails_val

    @property
    def dict_of_heads_val(self):
        return self._dict_of_heads_val
    
    @property
    def dict_of_tails_test(self):
        return self._dict_of_tails_test

    @property
    def dict_of_heads_test(self):
        return self._dict_of_heads_test

    @property
    def n_relations(self):
        return self.kg_train.n_rel 

    # for some reason, torchkge KG_train object has dict of tails and heads from train+val+test
    @property
    def dict_of_tails(self):
        return self.kg_train.dict_of_tails

    @property
    def dict_of_heads(self):
        return self.kg_train.dict_of_heads


    
class OgbKGEDataset(BaseDataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(config['device'])
        self.reciprocal = config.get('reciprocal', False)
        self.class_name = config['dataset']['class']
        self.root = config['dataset']['path']

        if self.class_name != 'OGBLBioKG':
            raise ValueError(f"Unsupported dataset class: {self.class_name}. This class is designed for 'OGBLBioKG'.")

        self.processed_dir = os.path.join(self.root, 'ogbl_biokg', 'processed')
        os.makedirs(self.processed_dir, exist_ok=True)
        print("Preprocessing data...")
        self._preprocess_data()

        print("Dataset created")

    def _preprocess_data(self):
        if self._check_preprocessed_files_exist():
            print("Preprocessed files exist. Loading...")
            self._load_preprocessed_files()
        else:
            print("Preprocessed files do not exist. Processing and saving...")
            self._process_and_save_data()

    def _check_preprocessed_files_exist(self):
        files_to_check = [
            'kg_train.pkl', 'kg_val.pkl', 'kg_test.pkl',
          'train_val_dicts.pkl'
        ]
        return all(os.path.exists(os.path.join(self.processed_dir, f)) for f in files_to_check)

    def _load_preprocessed_files(self):
        print("Loading preprocessed files...")
        with open(os.path.join(self.processed_dir, 'kg_train.pkl'), 'rb') as f:
            self.kg_train = pickle.load(f)
        with open(os.path.join(self.processed_dir, 'kg_val.pkl'), 'rb') as f:
            self.kg_val, self.val_neg_samples = pickle.load(f)
        with open(os.path.join(self.processed_dir, 'kg_test.pkl'), 'rb') as f:
            self.kg_test, self.test_neg_samples = pickle.load(f)
        
        with open(os.path.join(self.processed_dir, 'train_val_dicts.pkl'), 'rb') as f:
            self._dict_of_tails_train_val, self._dict_of_heads_train_val = pickle.load(f)

        self._n_entities = self.kg_train.n_ent
        self._n_relations = self.kg_train.n_rel

    def _process_and_save_data(self):
        print("Processing and saving data...")
        triples, self._n_entities, self._n_relations = preprocess_ogbl_dataset('ogbl-biokg', self.root, self.processed_dir)
 
        self.kg_train = self._create_kg(triples['train'])
        self.kg_val, self.val_neg_samples = self._create_kg(triples['valid'], return_neg_samples=True)
        self.kg_test, self.test_neg_samples = self._create_kg(triples['test'], return_neg_samples=True)
        
   
        self._dict_of_tails_train_val, self._dict_of_heads_train_val = get_dict_of_tails_and_heads([self.kg_train, self.kg_val])

        with open(os.path.join(self.processed_dir, 'kg_train.pkl'), 'wb') as f:
            pickle.dump(self.kg_train, f)
        with open(os.path.join(self.processed_dir, 'kg_val.pkl'), 'wb') as f:
            pickle.dump((self.kg_val, self.val_neg_samples), f)
        with open(os.path.join(self.processed_dir, 'kg_test.pkl'), 'wb') as f:
            pickle.dump((self.kg_test, self.test_neg_samples), f)
       
        with open(os.path.join(self.processed_dir, 'train_val_dicts.pkl'), 'wb') as f:
            pickle.dump((self._dict_of_tails_train_val, self._dict_of_heads_train_val), f)

    def _create_kg(self, array, return_neg_samples=False):
        head_idx = torch.LongTensor(array[:, 0])
        tail_idx = torch.LongTensor(array[:, 2])
        relations = torch.LongTensor(array[:, 1])
        neg_samples = torch.LongTensor(array[:, 3:1003]) if array.shape[1] > 3 else None

        kg_dict = {
            'heads': head_idx,
            'tails': tail_idx,
            'relations': relations,
        }
        entity_dict = {i: i for i in range(self.n_entities)}
        relation_dict = {i: i for i in range(self.n_relations)}
        kg = KnowledgeGraph(kg=kg_dict, ent2ix=entity_dict, rel2ix=relation_dict)
        
        if return_neg_samples:
            return kg, neg_samples
        return kg

    def filter_invalid_triples(self, scores: torch.Tensor, true_entities: torch.Tensor, mode: str, split: str, start_idx: int, end_idx: int) -> torch.Tensor:
        # Select the appropriate negative samples
        if split == 'valid':
            neg_samples = self.val_neg_samples[start_idx:end_idx]
        elif split == 'test':
            neg_samples = self.test_neg_samples[start_idx:end_idx]
        else:
            raise ValueError(f"Invalid split: {split}. Expected 'valid' or 'test'.")
        
        # Move neg_samples to the same device as scores if needed
        if neg_samples.device != scores.device:
            neg_samples = neg_samples.to(scores.device)
        
        valid_mask = torch.zeros_like(scores, dtype=torch.bool)
        
        if mode == 'tail':
            neg_samples_batch = neg_samples[:, 500:]
        else:  # head
            neg_samples_batch = neg_samples[:, :500]
        
        # Create valid mask in a vectorized manner
        valid_mask.scatter_(1, neg_samples_batch, True)
        valid_mask.scatter_(1, true_entities.unsqueeze(1), True)
        


        filtered_scores = scores.clone()
        filtered_scores[~valid_mask] = -float('Inf')
        
        return filtered_scores

    def get_dataloader(self, batch_size: int, split: str = 'train', num_workers: int = 0, persistent_workers: bool = False, sample_size: int = None):
        if split == 'train':
            kg = self.kg_train
            dataset = torch.utils.data.TensorDataset(kg.head_idx, kg.relations, kg.tail_idx)
        elif split == 'valid':
            kg = self.kg_val
            dataset = torch.utils.data.TensorDataset(kg.head_idx, kg.relations, kg.tail_idx, self.val_neg_samples)
        elif split == 'test':
            kg = self.kg_test
            dataset = torch.utils.data.TensorDataset(kg.head_idx, kg.relations, kg.tail_idx, self.test_neg_samples)
        else:
            raise ValueError(f"Unknown split: {split}")

        if split in ['train'] and sample_size is not None and sample_size < len(dataset):
            # Create a sampler that will be different for each epoch
            sampler = torch.utils.data.RandomSampler(dataset, num_samples=sample_size, replacement=True)
            shuffle = False  # We're using a sampler, so we don't need to shuffle
        else:
            sampler = None
            shuffle = (split == 'train')  # Only shuffle for training set if not using sampler

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            sampler=sampler
        )

    @property
    def n_relations(self):
        return self._n_relations

    @property
    def dict_of_tails_train_val(self):
        return self._dict_of_tails_train_val

    @property
    def dict_of_heads_train_val(self):
        return self._dict_of_heads_train_val

    @property
    def dict_of_tails_test(self):
        return self.kg_test.dict_of_tails

    @property
    def dict_of_heads_test(self):
        return self.kg_test.dict_of_heads

    @property
    def n_entities(self):
        return self._n_entities



