from external_models.wrappers.gekcs import GEKCS
from typing import Union
import numpy as np
from src.models import ARM

def count_parameters(model, detailed: bool = False) -> Union[float, dict]:
    """Count model parameters, optionally breaking down by parameter type.
    
    Args:
        model: The model to analyze
        detailed: If True, return breakdown of embedding vs scoring parameters
        
    Returns:
        Either total parameter count or dict with parameter breakdown
    """
    if not detailed:
        return sum(p.numel() for p in model.parameters())
    
    embedding_params = 0
    scoring_params = 0
    
    if isinstance(model, GEKCS):
        # For ComplEx models, all parameters are embeddings
        embedding_params = sum(p.numel() for p in model.parameters())
    else:
        # For transformer and other models
        for name, module in model.named_modules():
            params = sum(p.numel() for p in module.parameters())
            if 'emb' in name.lower():
                embedding_params += params
            else:
                scoring_params += params
    
    return {
        "total_params": "{:,.2f}M".format((embedding_params + scoring_params) / 1000000),
        "embedding_params": "{:,.2f}M".format(embedding_params / 1000000),
        "scoring_params": "{:,.2f}M".format(scoring_params / 1000000)
    }



def get_total_probability_mass(model, true_scores):
    total_mass = 0
    per_triple_scores = []
        
    if isinstance(model, GEKCS):
        # Average head and tail scores pairwise
        for head_score, tail_score in true_scores:
            triple_score = (head_score + tail_score) / 2
            per_triple_scores.append(triple_score)
            total_mass += triple_score
    elif isinstance(model, ARM):
        # Add head and tail scores pairwise
        for head_score, tail_score in true_scores:
            triple_score = head_score + tail_score
            per_triple_scores.append(triple_score)
            total_mass += triple_score
    else:
        raise ValueError("Only Probabilistic models supported")

    # Calculate statistics
    per_triple_stats = {
        "total_probability_mass": f"{total_mass:.2e}",
        "mean_score_per_triple": f"{np.mean(per_triple_scores):.2e}",
        "std_score_per_triple": f"{np.std(per_triple_scores):.2e}",
        "median_score_per_triple": f"{np.median(per_triple_scores):.2e}",
        "min_score": f"{np.min(per_triple_scores):.2e}",
        "max_score": f"{np.max(per_triple_scores):.2e}",
        "25th_percentile": f"{np.percentile(per_triple_scores, 25):.2e}",
        "75th_percentile": f"{np.percentile(per_triple_scores, 75):.2e}",
        "num_triples": len(per_triple_scores)
    }

    return per_triple_stats


def subject_distribution_shift(config, dataset):
    """Analyze distribution shift between train and test subjects.
    
    Returns:
        dict: Statistics about subject distribution shift
    """
    train_subjects = dataset.kg_train.head_idx.numpy()
    test_subjects = dataset.kg_test.head_idx.numpy()
    
    # Get unique test subjects only
    unique_test_subjects = np.unique(test_subjects)
        
    # Calculate relative frequencies
    train_counts = np.bincount(train_subjects)
    test_counts = np.bincount(test_subjects)
        
    # Ensure arrays are long enough for indexing
    max_id = max(len(train_counts), len(test_counts), max(unique_test_subjects) + 1)
    if len(train_counts) < max_id:
        train_counts = np.pad(train_counts, (0, max_id - len(train_counts)))
    
    # Count zero-shot cases (subjects that never appear in training)
    zero_shot_subjects = [s for s in unique_test_subjects if train_counts[s] == 0]
    
    # Calculate relative frequencies
    train_total = len(train_subjects)
    test_total = len(test_subjects)
    
    # Prepare results
    subject_frequencies = []
    for subject in unique_test_subjects:
        train_freq = train_counts[subject] / train_total
        test_freq = test_counts[subject] / test_total
        subject_frequencies.append({
            'subject': subject,
            'train_freq': train_freq,
            'test_freq': test_freq
        })
    
    results = {
        'zero_shot_count': len(zero_shot_subjects),
        'total_test_subjects': len(unique_test_subjects),
        'zero_shot_percentage': len(zero_shot_subjects) / len(unique_test_subjects) * 100,
        'subject_frequencies': subject_frequencies
    }
    
    return results

def subject_given_relation_distribution(dataset) -> dict:
    """Analyze unique subject-relation pairs in test set and their overlap with training.
    
    Returns:
        dict: Statistics about unique subject-relation pairs and zero-shot cases
    """
    # Get all triples
    train_subjects = dataset.kg_val.head_idx.numpy()
    train_relations = dataset.kg_val.relations.numpy()
    test_subjects = dataset.kg_test.head_idx.numpy()
    test_relations = dataset.kg_test.relations.numpy()
    
    # Get unique (subject, relation) pairs from test set
    test_pairs = set((s, r) for s, r in zip(test_subjects, test_relations))
    train_pairs = set((s, r) for s, r in zip(train_subjects, train_relations))
    
    # Find pairs that appear only in test set
    novel_pairs = test_pairs - train_pairs
    
    return {
        'total_unique_test_pairs': len(test_pairs),
        'novel_pairs': len(novel_pairs),
        'novel_pairs_percentage': (len(novel_pairs) / len(test_pairs)) * 100,
    }