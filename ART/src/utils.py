from __future__ import annotations
import numpy as np
from collections import defaultdict
from torchkge.utils.datasets import load_fb15k237, load_wn18rr
from torchkge.data_structures import KnowledgeGraph
import torch.nn.functional as F
import torch
import pandas as pd
from torch.utils.data import DataLoader
from ogb.linkproppred import LinkPropPredDataset
from collections import defaultdict
from typing import Optional, List, Union
from torchkge.data_structures import KnowledgeGraph
import numpy as np
from scipy.stats import entropy
from collections import Counter
import matplotlib.pyplot as plt
import os
import json
from easydict import EasyDict
import jinja2
import yaml
import pickle
from torch.utils.data import DataLoader, TensorDataset


def get_relation_frequencies(kg_train: KnowledgeGraph) -> torch.Tensor:

    relation_frequencies = torch.zeros(kg_train.n_rel, dtype=torch.int)

    for j in range(kg_train.n_facts):
        relation = int(kg_train.relations[j].item())
        relation_frequencies[relation] += 1

    return relation_frequencies

def get_prior_frequencies(kg_train: KnowledgeGraph) -> torch.Tensor:
    # Check if cached frequencies exist
    
    print("Computing prior frequencies...")
    prior_frequencies = torch.zeros(kg_train.n_ent, dtype=torch.int)

    for j in range(kg_train.n_facts):
        subject = int(kg_train.head_idx[j].item())
        prior_frequencies[subject] += 1
        object = int(kg_train.tail_idx[j].item())
        prior_frequencies[object] += 1

    # Cache the frequencies
    
    return prior_frequencies
def get_avg_indegree_per_relation(kg_train: KnowledgeGraph) -> dict[int, float]:

    relation_indegree: dict[int, tuple[int, set]] = {}

    for j in range(kg_train.n_facts):
        relation = kg_train.relations[j].item()

        if relation not in relation_indegree:
            relation_indegree[relation] = (0, set())

        current_count = relation_indegree[relation][0]
        current_nodes = relation_indegree[relation][1]
        current_count = current_count + 1
        current_nodes.add(relation)

        relation_indegree[relation] = (current_count, current_nodes)

    avg_indegree_per_relation = {}
    for i, (k, v) in enumerate(relation_indegree.items()):
        avg_indegree_per_relation[i] = k / len(v)

    return avg_indegree_per_relation


def get_avg_indegree_per_node(kg_train: KnowledgeGraph) -> torch.Tensor:

    object_frequencies = torch.zeros(kg_train.n_ent, dtype=torch.int)
    for j in range(kg_train.n_facts):
        object_id = int(kg_train.tail_idx[j].item())

        object_frequencies[object_id] += 1

    return torch.sum(object_frequencies)/kg_train.n_ent


def get_dict_of_tails_and_heads(kg_list: List[KnowledgeGraph], 
                                existing_tails: Optional[dict] = None, 
                                existing_heads: Optional[dict] = None) -> tuple[dict, dict]:
    dict_of_tails = defaultdict(set) if existing_tails is None else existing_tails
    dict_of_heads = defaultdict(set) if existing_heads is None else existing_heads

    for kg in kg_list:
        for i in range(kg.n_facts):
            head = kg.head_idx[i].item()
            relation = kg.relations[i].item()
            tail = kg.tail_idx[i].item()

            dict_of_tails[(head, relation)].add(tail)
            dict_of_heads[(tail, relation)].add(head)

    return dict_of_tails, dict_of_heads

def get_true_targets_batch(scores, dictionary, key1, key2):

    b_size = scores.shape[0]
    labels = torch.zeros_like(scores)

    for i in range(b_size):
        true_targets = dictionary[key1[i].item(), key2[i].item()].copy()
        if not true_targets:
            continue
        true_targets = torch.tensor(list(true_targets)).long()
        labels[i][true_targets] = True

    return labels


def subset_to_dataframe(subset):
    data_loader = DataLoader(subset)
    heads, tails, relations = [], [], []

    for batch in data_loader:
        batch = batch.squeeze(0)
        heads.append(batch[0].item())
        tails.append(batch[1].item())
        relations.append(batch[2].item())

    return pd.DataFrame({
        'from': heads,
        'rel': relations,
        'to': tails
    })
def load_nbf_mapping(dataset_name, data_path):
    from torchdrug import datasets
    """Load FB15k dataset. See `here
    <https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data>`__
    for paper by Bordes et al. originally presenting the dataset.

    Parameters
    ----------
    data_home: str, optional
        Path to the `torchkge_data` directory (containing data folders). If
        files are not present on disk in this directory, they are downloaded
        and then placed in the right place.

    Returns
    -------
    kg_train: torchkge.data_structures.KnowledgeGraph
    kg_val: torchkge.data_structures.KnowledgeGraph
    kg_test: torchkge.data_structures.KnowledgeGraph

    """
    if dataset_name == "fb15k237":
        dataset = datasets.FB15k237(data_path)
    elif dataset_name == "wn18rr":
        dataset = datasets.WN18RR(data_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    df1, df2, df3 = dataset.split()
    df1 = subset_to_dataframe(df1)
    df2 = subset_to_dataframe(df2)
    df3 = subset_to_dataframe(df3)
    df = pd.concat([df1, df2, df3])
    entity_dict = {string: int(i) for i, string in enumerate(dataset.entity_vocab)}
    relation_dict = {string: int(i) for i, string in enumerate(dataset.relation_vocab)}
    kg_nbf = {}
    kg_nbf['heads'] = torch.tensor(df['from'].values)
    kg_nbf['relations'] = torch.tensor(df['rel'].values)
    kg_nbf['tails'] = torch.tensor(df['to'].values)
    kg = KnowledgeGraph(kg=kg_nbf, ent2ix=entity_dict, rel2ix=relation_dict)

    return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))

def load_dataset(config) -> tuple[KnowledgeGraph, KnowledgeGraph, KnowledgeGraph]:

    def assert_three(result: tuple[KnowledgeGraph, KnowledgeGraph, KnowledgeGraph] | tuple[KnowledgeGraph, KnowledgeGraph]
                     ) -> tuple[KnowledgeGraph, KnowledgeGraph, KnowledgeGraph]:
        assert len(result) == 3
        return result

    dataset_class = config['dataset']['class'].lower()

    if dataset_class == "fb15k237":
        if config['model_type'] == 'nbf':
            return assert_three(load_nbf_mapping("fb15k237", config['dataset']['path']))

        return assert_three(load_fb15k237(data_home=config['dataset']['path']))

    elif dataset_class == "wn18rr":
        if config['model_type'] == 'nbf':
            return assert_three(load_nbf_mapping("wn18rr", config['dataset']['path']))
        return assert_three(load_wn18rr(data_home=config['dataset']['path']))

    else:
        raise Exception(f"Dataset unknown: {dataset_class}")

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

def convert_benchmark_to_nbf_mappings(config, benchmark_dict, benchmark_ent2idx=None, benchmark_rel2idx=None):
    """Convert benchmark dictionary from benchmark IDs (sorted) to NBF IDs (vocab order)"""
    if config['dataset']['class'].lower() == "ogblbiokg":
        pass
        # Create offset dictionaries for both mappings
        # nbf_offsets = create_offset_dict(NBF_TYPE_COUNTS)
        # this_offsets = create_offset_dict(THIS_TYPE_COUNTS)
        
        # def convert_id(idx):
        #     for entity_type, offset in this_offsets.items():
        #         if idx >= offset and idx < offset + THIS_TYPE_COUNTS[entity_type]:
        #             return idx - this_offsets[entity_type] + nbf_offsets[entity_type]
        #     raise ValueError(f"Index {idx} is out of range")
        
        # converted_dict = {}
        # for (h, r, t), (head_data, tail_data) in benchmark_dict.items():
        #     new_h = convert_id(h)
        #     new_r = r  # Relations are 1:1 mapped
        #     new_t = convert_id(t)
            
        #     head_candidates, head_labels = head_data
        #     tail_candidates, tail_labels = tail_data
            
        #     # Convert while maintaining numpy array format
        #     new_head_candidates = np.array([convert_id(c) for c in head_candidates], dtype=np.int64)
        #     new_tail_candidates = np.array([convert_id(c) for c in tail_candidates], dtype=np.int64)
            
        #     converted_dict[(new_h, new_r, new_t)] = (
        #         (new_head_candidates, head_labels),
        #         (new_tail_candidates, tail_labels)
        #     )
        
        # return converted_dict
    else:
        pass
        # Original code for other datasets
        # be
        # from torchdrug import datasets  
        # with open(os.path.join(benchmark_dir, 'ent2idx.pkl'), 'rb') as f:
        #     benchmark_ent2idx = pickle.load(f)
        # with open(os.path.join(benchmark_dir, 'rel2idx.pkl'), 'rb') as f:
        #     benchmark_rel2idx = pickle.load(f)
        # # For other datasets, create mapping from benchmark IDs to NBF IDs
        # # We need to go: benchmark_id -> entity_name -> nbf_id
        #     entity_dict = {
        #         benchmark_id: self.nbf_entity_dict[entity_name] 
        #         for entity_name, benchmark_id in self.benchmark_ent2idx.items()
        #     }
        #     relation_dict = {
        #         benchmark_id: self.nbf_relation_dict[relation_name]
        #         for relation_name, benchmark_id in self.benchmark_rel2idx.items()
        #     }
        # if config['dataset']['class'].lower() == "fb15k237":
        #     dataset = datasets.FB15k237(config['dataset']['path'])
        # else:
        #     dataset = datasets.WN18RR(config['dataset']['path'])
            
        #     # Get both mappings
        #     # NBF mapping: name -> nbf_id (vocab order)
        #     nbf_entity_dict = {string: int(i) for i, string in enumerate(dataset.entity_vocab)}
        #     nbf_relation_dict = {string: int(i) for i, string in enumerate(dataset.relation_vocab)}
            
        #     # Benchmark mapping: name -> bench_id (sorted order)
        #     # We already have this in self.benchmark_ent2idx and self.benchmark_rel2idx
            
        #     # Create mapping from benchmark IDs to NBF IDs
        #     bench_to_nbf_ent = {}
        #     for name in dataset.entity_vocab:
        #         bench_id = benchmark_ent2idx[name]
        #         nbf_id = nbf_entity_dict[name]
        #         bench_to_nbf_ent[bench_id] = nbf_id
            
        #     bench_to_nbf_rel = {}
        #     for name in dataset.relation_vocab:
        #         bench_id = benchmark_rel2idx[name]
        #         nbf_id = nbf_relation_dict[name]
        #         bench_to_nbf_rel[bench_id] = nbf_id
            
        #     # Print first few mappings for debugging
        #     print("\nFirst few entity mappings (bench_id -> nbf_id):")
        #     print(dict(list(bench_to_nbf_ent.items())[:5]))
        #     print("\nFirst few relation mappings (bench_id -> nbf_id):")
        #     print(dict(list(bench_to_nbf_rel.items())[:5]))
            
        #     # Convert using these mappings
        #     converted_dict = {}
        #     for (h, r, t), (head_data, tail_data) in benchmark_dict.items():
        #         new_h = bench_to_nbf_ent[h]
        #         new_r = bench_to_nbf_rel[r]
        #         new_t = bench_to_nbf_ent[t]
                
        #         head_candidates, head_labels = head_data
        #         tail_candidates, tail_labels = tail_data
                
        #         new_head_candidates = [bench_to_nbf_ent[c] for c in head_candidates]
        #         new_tail_candidates = [bench_to_nbf_ent[c] for c in tail_candidates]
                
        #         converted_dict[(new_h, new_r, new_t)] = (
        #             (new_head_candidates, head_labels),
        #             (new_tail_candidates, tail_labels)
        #         )
            
        #     return converted_dict

def get_ir_dataloader(config, dataset, split):
    benchmark_path = os.path.join(config['dataset']['path'], 'kbc_benchmarks', config['dataset']['class'].lower())
        
    # Load benchmark dictionary
    with open(os.path.join(benchmark_path, f'{split}_benchmark.pkl'), 'rb') as f:
        benchmark_dict = pickle.load(f)
    print("Loading benchmark dict...")
    print(f"benchmark_dict = {benchmark_dict}")  # Check what's in it
    print("converting benchmark dict") 
        # Convert mappings for NBF model
    if config['model_type'] == 'nbf':
        benchmark_dict = convert_benchmark_to_nbf_mappings(config, benchmark_dict)
        
    print(f"Number of queries in benchmark: {len(benchmark_dict)}")
        
        # Create dataset from all queries
    test_queries = []
    tail_masks = []
    head_masks = []
    tail_scores_list = []
    head_scores_list = []
    tail_labels = 0
    head_labels = 0
    tail_labels_list = []
    head_labels_list = []
    
    for i, (query, (head_data, tail_data)) in enumerate(benchmark_dict.items()):
        head_candidates, head_labels = head_data
        tail_candidates, tail_labels = tail_data
        tail_mask = torch.zeros(config['dataset']['n_entities'], dtype=torch.bool)
        head_mask = torch.zeros(config['dataset']['n_entities'], dtype=torch.bool)
        # Handle tail candidates
        if len(tail_candidates) > 0:
            # Convert to numpy array if not already
            tail_candidates = np.asarray(tail_candidates, dtype=np.int64)
            tail_sort_idx = np.argsort(tail_candidates)
            tail_labels_list.extend(tail_labels[tail_sort_idx])
            
            # Create masks (ensure proper type conversion)
            tail_mask.scatter_(0, torch.tensor(tail_candidates[tail_sort_idx], dtype=torch.int64), True)
        tail_masks.append(tail_mask)
        
        # Similar changes for head candidates...
        if len(head_candidates) > 0:
            head_candidates = np.asarray(head_candidates, dtype=np.int64)
            head_sort_idx = np.argsort(head_candidates)
            head_labels_list.extend(head_labels[head_sort_idx])
            
            head_mask = torch.zeros(config['dataset']['n_entities'], dtype=torch.bool)
            head_mask.scatter_(0, torch.tensor(head_candidates[head_sort_idx], dtype=torch.int64), True)

        head_masks.append(head_mask)
        
        test_queries.append(list(query))

    # Convert to numpy arrays once
    print("True forward triples", sum(tail_labels_list))
    print("True inverse triples", sum(head_labels_list))
    labels = np.array(tail_labels_list + head_labels_list)
    test_queries_tensor = torch.tensor(test_queries).to(config['device'])

    dataset = TensorDataset(
        test_queries_tensor,
        torch.stack(tail_masks),
        torch.stack(head_masks),
        torch.arange(len(benchmark_dict))
    )
        
    test_loader = DataLoader(
        dataset,
        batch_size=config['test_batch_size'],
        shuffle=False
    )

    return test_loader, labels

def calculate_distribution(subjects):
    count = Counter(subjects)
    total = sum(count.values())
    return {k: v / total for k, v in count.items()}

def kl_divergence(p, q):
    # Get the union of all keys
    all_keys = set(p.keys()) | set(q.keys())
    
    # Create arrays with zeros for missing keys
    p_values = np.array([p.get(k, 0) for k in all_keys])
    q_values = np.array([q.get(k, 1e-10) for k in all_keys])  # Use a small value instead of 0 for q
    
    # Normalize the arrays
    p_values = p_values / np.sum(p_values)
    q_values = q_values / np.sum(q_values)
    
    return entropy(p_values, qk=q_values)

def plot_distributions(train_dist, test_dist, save_path):
    all_keys = set(train_dist.keys()) | set(test_dist.keys())
    
    # Calculate the difference between train and test distributions
    diff = {k: abs(train_dist.get(k, 0) - test_dist.get(k, 0)) for k in all_keys}
    
    # Sort keys by the difference, in descending order
    sorted_keys = sorted(diff, key=diff.get, reverse=True)[:100]
    
    train_values = [train_dist.get(k, 0) for k in sorted_keys]
    test_values = [test_dist.get(k, 0) for k in sorted_keys]
    
    plt.figure(figsize=(15, 8))
    x = range(len(sorted_keys))
    plt.bar([i-0.2 for i in x], train_values, width=0.4, alpha=0.6, label='Train')
    plt.bar([i+0.2 for i in x], test_values, width=0.4, alpha=0.6, label='Test')
    plt.xlabel('Top 100 Subject Entity IDs with Largest Differences')
    plt.ylabel('Probability')
    plt.title('Distribution of Top 100 Subjects with Largest Differences in Train vs Test')
    plt.legend()
    plt.xticks(x, sorted_keys, rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_optimization_metric(metrics, metric_name=None):
    """Get the metric to optimize (convert all to maximization problems)"""
    return {
        None: -metrics['validation_loss'],  # default
        'validation_loss': -metrics['validation_loss'],
        'train_loss': -metrics['train_loss'],
        'train_and_validation_loss': -(metrics['validation_loss'] + metrics['train_loss']),
        'mrr': metrics['filtered_mrr'],
    }[metric_name]



def save_checkpoint(model, save_path, metric_name, metric_value):
    """Save model checkpoint and print status"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"\nSaving model checkpoint:")
    print(f"Metric: {metric_name} = {metric_value:.6f}")
    print(f"Directory: {os.path.dirname(save_path)}")
    print(f"Full path: {save_path}")
    
    torch.save(model.state_dict(), save_path)
    print(f"Successfully saved model!\n")

def get_results_path(model_dir: str, result_name: str) -> str:
    """Convert model directory path to results path, maintaining the same structure"""
    # Simply replace 'saved_models' with 'results' in the path
    results_path = os.path.join(
        'results',
        os.path.relpath(model_dir, 'saved_models'),
        f"{result_name}.json"
    )
    
    return results_path


def save_results(results: dict, model_dir: str, result_name: str):
    """Save experiment results to JSON file"""
    results_path = get_results_path(model_dir, result_name)
    
    print(f"\nSaving results:")
    print(f"Result: {result_name}")
    print(f"Full path: {results_path}")
    
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Successfully saved results!\n")

def add_defaults_and_override_from_cli(config, args, default_args, cli_to_config):
    """
    Add default values to config and override with CLI arguments.
    
    Args:
        config (EasyDict): The base configuration from YAML
        args (Namespace): The parsed CLI arguments
        default_args (Namespace): The default CLI arguments
        cli_to_config (dict): Mapping of CLI args to nested config paths
    
    Returns:
        EasyDict: Updated configuration
    """
    # First, add all default values that aren't in config
    for arg in vars(default_args):
        if arg not in config and getattr(default_args, arg) is not None:
            config[arg] = getattr(default_args, arg)

    # Then override with explicitly set CLI args
    for arg in vars(args):
        value = getattr(args, arg)
        default_value = getattr(default_args, arg)
        # Only override if value is not None and different from default
        if value is not None and value != default_value:
            if arg in cli_to_config:
                # Handle nested configs
                section, key = cli_to_config[arg]
                if section not in config:
                    config[section] = EasyDict({})
                config[section][key] = value
                print(f"Overriding {section}.{key} with value: {value}")
            else:
                # Handle flat configs
                config[arg] = value
                print(f"Overriding {arg} with value: {value}")
    
    return config
def get_model_path_suffix(config):
    """Generate model path suffix based on all important model settings"""
    parts = []
    
    # Common model parameters
    if config.get('version'):
        parts.append(f"v_{config['version']}")
    parts.append(f"lr{config['lr']}")
    parts.append(f"drop{config['dropout']}")
    
    # Model specific parameters
    if config['model_type'] == 'arm_transformer':
        parts.extend([
            f"blocks{config['num_blocks']}",
            f"heads{config['num_heads']}",
            f"neurons{config['num_neurons']}"
        ])
    elif config['model_type'] == 'arm_convolution':
        parts.extend([
            f"kernel{config['kernel_size']}",
            f"hidden{config['m']}"
        ])
    
    # Prior settings
    prior_parts = []
    if config['prior'].get('alpha'):
        prior_parts.append(f"alpha_{config['prior']['alpha']}")
    if config['prior'].get('init_weights'):
        prior_parts.append(f"init_{config['prior']['init_weights']}")
    if config['prior'].get('optimize_temperature'):
        prior_parts.append("opt_temp")
    if prior_parts:
        parts.append('prior_' + '_'.join(prior_parts))
    
    # Training settings
    parts.append(f"opt_{config['optimize']}")
    parts.append(f"batch{config['batch_size']}")
    
    return "_".join(parts)

def load_config_yaml(config_path):
    """
    Load and parse a YAML configuration file with Jinja2 templating.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        EasyDict: Parsed configuration as an EasyDict object
    """
    with open(config_path, "r") as f:
        raw = f.read()
        template = jinja2.Template(raw)
        config_yaml = template.render()
        config = yaml.safe_load(config_yaml)
        return EasyDict(config)