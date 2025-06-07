# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
Modified for ARM models
"""


from typing import Optional, Union, cast

import torch
from torchkge.exceptions import NotYetEvaluatedError
from torchkge.utils import filter_scores, get_rank 
from tqdm.autonotebook import tqdm
import os
import pickle
from src.dataset import KGEDataset, OgbKGEDataset
from src.criterion import LogLoss
from src.models import ARM

class LinkPredictionEvaluator:
    def __init__(self, config, model, dataset: Union[KGEDataset, OgbKGEDataset], split: str = 'test'):
        self.config = config
        self.device = torch.device(config['device'])
        self.model = model
        self.model.eval()
        self.dataset = dataset
        self.split = split
        self.batch_size = config['test_batch_size']

        self.processed_dir = os.path.join(config['dataset']['path'], 'ogbl_biokg', 'processed')

        if split == 'test':
            self.kg_eval = self.dataset.kg_test
        elif split == 'valid':
            self.kg_eval = self.dataset.kg_val
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'test' or 'valid'.")

        # Get dataloader
        self.dataloader = self.dataset.get_dataloader(batch_size=self.batch_size, split=split)

        # Get number of facts from the dataloader
        self.n_facts = len(self.dataloader.dataset)

        # Initialize tensors with the correct size
        self.rank_true_heads = torch.zeros(size=(self.n_facts,)).long().to(self.device)
        self.rank_true_tails = torch.zeros(size=(self.n_facts,)).long().to(self.device)
        self.filt_rank_true_heads = torch.zeros(size=(self.n_facts,)).long().to(self.device)
        self.filt_rank_true_tails = torch.zeros(size=(self.n_facts,)).long().to(self.device)
        self.evaluated = False
        self.is_typed_dataset = isinstance(dataset, (OgbKGEDataset ))
        self.seen_triples = set()

        # Add criterion initialization
        self.criterion = LogLoss(
            config.get('prediction_smoothing', 0.0),
            config.get('label_smoothing', 0.0),
            dataset.n_entities,
            dataset.n_relations + dataset.n_relations  # Double for inverse relations
        )
        # Initialize total loss
        self.total_loss = 0.0
        self.num_batches = 0

    def evaluate_lp(self, verbose: bool = True, store_true_triples: bool = False, save_k_negatives=None, calculate_validation_loss: bool = False) -> None:
        
        use_cuda = self.device == "cuda"
        print("USE CUDA", use_cuda)

        start_idx = 0
        batch_size = self.batch_size
        self.total_loss = 0.0
        self.num_batches = 0
        self.true_scores = []
        self.negative_scores = []
        # Initialize storage for scores as lists of tuples

        for i, (batch) in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            h_idx, r_idx, t_idx = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
            start_idx = i * batch_size
            end_idx = min((i+1) * batch_size, self.n_facts)

            # # Calculate loss
            if calculate_validation_loss:
                assert isinstance(self.model, ARM), "Only supprted for ARM models"
                batch_validation_loss = get_validation_loss(self.model, (h_idx, r_idx, t_idx), self.dataset.kg_train.n_rel)
           
            # Tail prediction
            y_score_t = self.model.inference_tail_prediction(h_idx, r_idx, t_idx)
            filt_scores_t = self.filter_triples(y_score_t, (h_idx, r_idx, t_idx), 'tail', start_idx, end_idx)

            y_score_h = self.model.inference_head_prediction(h_idx, r_idx, t_idx )
            filt_scores_h = self.filter_triples(y_score_h, (h_idx, r_idx, t_idx), 'head', start_idx, end_idx)
            # Store true scores if requested
            if store_true_triples:
                self.store_true_scores(y_score_t, y_score_h, t_idx, h_idx)

            # Store negative samples
            if save_k_negatives is not None:
                self.store_negative_scores(filt_scores_t, t_idx, save_k_negatives)
                tail_negs = self._sample_negative_scores(filt_scores_t, t_idx, save_k_negatives)
                head_negs = self._sample_negative_scores(filt_scores_h, h_idx, save_k_negatives)
                self.negative_scores.extend(zip(tail_negs, head_negs))

            self.rank_true_tails[start_idx:end_idx] = get_rank(y_score_t, t_idx).detach()
            self.filt_rank_true_tails[start_idx:end_idx] = get_rank(filt_scores_t, t_idx).detach()
            self.rank_true_heads[start_idx:end_idx] = get_rank(y_score_h, h_idx).detach()
            self.filt_rank_true_heads[start_idx:end_idx] = get_rank(filt_scores_h, h_idx).detach()

        self.evaluated = True

        if use_cuda:
            self.rank_true_heads = self.rank_true_heads.cpu()
            self.rank_true_tails = self.rank_true_tails.cpu()
            self.filt_rank_true_heads = self.filt_rank_true_heads.cpu()
            self.filt_rank_true_tails = self.filt_rank_true_tails.cpu()
        
        if store_true_triples:
            dataset_dir = os.path.join('./saved_models', 
                                     self.config["dataset"]["class"].lower(), 
                                     self.config["model_type"])
            os.makedirs(dataset_dir, exist_ok=True)
            
            with open(os.path.join(dataset_dir, f'true_scores_{self.split}.pkl'), 'wb') as f:
                pickle.dump(self.true_scores, f)
                print("saved scores")
            
        if save_k_negatives is not None:
            with open(os.path.join(dataset_dir, f'negative_scores_{self.split}.pkl'), 'wb') as f:
                pickle.dump(self.negative_scores, f)
                print("saved negatives")
        
        return self.true_scores, self.negative_scores

    def _sample_negative_scores(self, scores_tensor: torch.Tensor, true_indices: torch.Tensor, n_samples: int) -> list:
        """Helper function to sample negative scores from a tensor of filtered scores.
        
        Args:
            scores_tensor: Tensor of scores with filtered scores set to -inf
            true_indices: Tensor of indices for true triples to exclude
            n_samples: Number of negative samples to take per triple. If -1, return all valid scores.
            
        Returns:
            list of sampled negative scores
        """
        negative_samples = []
        
        for score_row, true_idx in zip(scores_tensor, true_indices):
            # Get scores that aren't filtered out and exclude the true triple
            valid_mask = (score_row != float('-inf'))
            valid_mask[true_idx] = False  # Exclude the true triple
            valid_scores = score_row[valid_mask]
            
            if n_samples == -1:
                # Return all valid scores
                negative_samples.extend(valid_scores.cpu().tolist())
            else:
                # Randomly sample from valid scores
                if len(valid_scores) > n_samples:
                    perm = torch.randperm(len(valid_scores))[:n_samples]
                    sampled_scores = valid_scores[perm]
                    negative_samples.extend(sampled_scores.cpu().tolist())
                else:
                    # If we have fewer valid scores than requested samples, return all of them
                    negative_samples.extend(valid_scores.cpu().tolist())
        
        return negative_samples
    
    def filter_triples(self, score, triple, direction='tail', start_idx=0, end_idx=None):
        if direction == 'tail':
            if self.is_typed_dataset:
                return self.dataset.filter_invalid_triples(score, triple[2], 'tail', self.split, start_idx, end_idx)
            else:
                return filter_scores(score, self.dataset.dict_of_tails, triple[0], triple[1], triple[2])
        else:
            if self.is_typed_dataset:
                return self.dataset.filter_invalid_triples(score, triple[0], 'head', self.split, start_idx, end_idx)
            else:
                return filter_scores(score, self.dataset.dict_of_heads, triple[2], triple[1], triple[0])
    
    def store_true_triples(self, y_score_t, y_score_h, t_idx, h_idx):
        true_tail_scores = y_score_t.gather(1, t_idx.view(-1, 1)).squeeze().cpu().tolist()
        true_head_scores = y_score_h.gather(1, h_idx.view(-1, 1)).squeeze().cpu().tolist()
        self.true_scores.extend(zip(true_tail_scores, true_head_scores))
    
    def store_negative_scores(self, filt_scores, idx, n_samples):
        tail_negs = self._sample_negative_scores(filt_scores, idx, n_samples)
        head_negs = self._sample_negative_scores(filt_scores, idx, n_samples)
        self.negative_scores.extend(zip(tail_negs, head_negs))

    def mean_rank(self):
        """

        Returns
        -------
        mean_rank: float
            Mean rank of the true entity when replacing alternatively head
            and tail in any fact of the dataset.
        filt_mean_rank: float
            Filtered mean rank of the true entity when replacing
            alternatively head and tail in any fact of the dataset.

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        sum_ = (self.rank_true_heads.float().mean() +
              self.rank_true_tails.float().mean()).item()
        filt_sum = (self.filt_rank_true_heads.float().mean() +
                  self.filt_rank_true_tails.float().mean()).item()
        return sum_ / 2, filt_sum / 2

    def mean_rank_head(self):
        """

        Returns
        -------
        mean_rank: float
            Mean rank of the true entity when replacing alternatively head
            and tail in any fact of the dataset.
        filt_mean_rank: float
            Filtered mean rank of the true entity when replacing
            alternatively head and tail in any fact of the dataset.

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        return self.rank_true_heads.float().mean().item(), self.filt_rank_true_heads.float().mean().item()

    def mean_rank_tail(self):
        """

        Returns
        -------
        mean_rank: float
            Mean rank of the true entity when replacing alternatively head
            and tail in any fact of the dataset.
        filt_mean_rank: float
            Filtered mean rank of the true entity when replacing
            alternatively head and tail in any fact of the dataset.

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        return self.rank_true_tails.float().mean().item(), self.filt_rank_true_tails.float().mean().item()

    def hit_at_k_heads(self, k: int = 10) -> tuple[float, float]:
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        head_hit = (self.rank_true_heads <= k).float().mean()
        filt_head_hit = (self.filt_rank_true_heads <= k).float().mean()

        return head_hit.item(), filt_head_hit.item()

    def hit_at_k_tails(self, k: int = 10) -> tuple[float, float]:
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        tail_hit = (self.rank_true_tails <= k).float().mean()
        filt_tail_hit = (self.filt_rank_true_tails <= k).float().mean()

        return tail_hit.item(), filt_tail_hit.item()

    def hit_at_k(self, k: int = 10):
        """

        Parameters
        ----------
        k: int
            Hit@k is the number of entities that show up in the top k that
            give facts present in the dataset.

        Returns
        -------
        avg_hitatk: float
            Average of hit@k for head and tail replacement.
        filt_avg_hitatk: float
            Filtered average of hit@k for head and tail replacement.

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')

        head_hit, filt_head_hit = self.hit_at_k_heads(k=k)
        tail_hit, filt_tail_hit = self.hit_at_k_tails(k=k)

        return (head_hit + tail_hit) / 2, (filt_head_hit + filt_tail_hit) / 2

    def hit_at_k_tail(self, k: int = 10):
        """

        Parameters
        ----------
        k: int
            Hit@k is the number of entities that show up in the top k that
            give facts present in the dataset.

        Returns
        -------
        avg_hitatk: float
            Average of hit@k for head and tail replacement.
        filt_avg_hitatk: float
            Filtered average of hit@k for head and tail replacement.

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')

        tail_hit, filt_tail_hit = self.hit_at_k_tails(k=k)

        return tail_hit, filt_tail_hit

    def mrr(self) -> tuple[float, float]:
        """

        Returns
        -------
        avg_mrr: float
            Average of mean recovery rank for head and tail replacement.
        filt_avg_mrr: float
            Filtered average of mean recovery rank for head and tail
            replacement.

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        head_mrr = (self.rank_true_heads.float()**(-1)).mean()
        tail_mrr = (self.rank_true_tails.float()**(-1)).mean()
        filt_head_mrr = (self.filt_rank_true_heads.float()**(-1)).mean()
        filt_tail_mrr = (self.filt_rank_true_tails.float()**(-1)).mean()

        return ((head_mrr + tail_mrr).item() / 2,
                (filt_head_mrr + filt_tail_mrr).item() / 2)

    def mrr_head(self) -> tuple[float, float]:
        """


        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        head_mrr = (self.rank_true_heads.float()**(-1)).mean().item()
        filt_head_mrr = (self.filt_rank_true_heads.float()**(-1)).mean().item()

        return head_mrr, filt_head_mrr

    def mrr_tail(self) -> tuple[float, float]:
        """

        Returns
        -------

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        tail_mrr = (self.rank_true_tails.float()**(-1)).mean().item()
        filt_tail_mrr = (self.filt_rank_true_tails.float()**(-1)).mean().item()

        return tail_mrr, filt_tail_mrr

    def print_results(self, k: Optional[Union[int, list[int]]] = 10, n_digits: int = 3) -> None:
        """

        Parameters
        ----------
        k: int or list
            k (or list of k) such that hit@k will be printed.
        n_digits: int
            Number of digits to be printed for hit@k and MRR.
        """

        if isinstance(k, int):
            print('Hit@{} : {} \t\t Filt. Hit@{} : {}'.format(
                k, round(self.hit_at_k(k=k)[0], n_digits),
                k, round(self.hit_at_k(k=k)[1], n_digits)))
        elif isinstance(k, list):
            for i in k:
                print('Hit@{} : {} \t\t Filt. Hit@{} : {}'.format(
                    i, round(self.hit_at_k(k=i)[0], n_digits),
                    i, round(self.hit_at_k(k=i)[1], n_digits)))
        else:
            raise AssertionError(
                f"Paramter k with value{k} is not a integer or list of integers")

        print('Mean Rank : {} \t Filt. Mean Rank : {}'.format(
            int(self.mean_rank()[0]), int(self.mean_rank()[1])))
        print('MRR : {} \t\t Filt. MRR : {}'.format(
            round(self.mrr()[0], n_digits), round(self.mrr()[1], n_digits)))


    def get_link_prediction_metrics(self):
        self.evaluate_lp()
        
        # Calculate average loss
        avg_loss = self.total_loss / self.num_batches if self.num_batches > 0 else float('inf')
        
        # Get existing metrics
        mrr, filtered_mrr = self.mrr()
        mean_rank, filtered_mean_rank = self.mean_rank()
        hit_k_1, hit_k_1_filtered = self.hit_at_k(k=1)
        hit_k_3, hit_k_3_filtered = self.hit_at_k(k=3)
        hit_k_5, hit_k_5_filtered = self.hit_at_k(k=5)
        hit_k_10, hit_k_10_filtered = self.hit_at_k(k=10)
        mrr_head, mrr_head_filtered = self.mrr_head()
        mrr_tail, mrr_tail_filtered = self.mrr_tail()

        return {
            'validation_loss': avg_loss,  # Add loss to metrics
            'mrr': mrr,
            'mean_rank': mean_rank,
            'mrr_head': mrr_head,
            'mrr_tail': mrr_tail,
            'filtered_mean_rank': filtered_mean_rank,
            'filtered_mrr': filtered_mrr,
            'hit_k_1': hit_k_1,
            'hit_k_1_filtered': hit_k_1_filtered,
            'hit_k_3': hit_k_3,
            'hit_k_3_filtered': hit_k_3_filtered,
            'hit_k_5': hit_k_5,
            'hit_k_5_filtered': hit_k_5_filtered,
            'hit_k_10': hit_k_10,
            'hit_k_10_filtered': hit_k_10_filtered,
            'mrr_tail_filtered': mrr_tail_filtered,
            'mrr_head_filtered': mrr_head_filtered,
        }
