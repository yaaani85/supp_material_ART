from __future__ import annotations
from torch import Tensor
from torch.nn.init import xavier_normal_
from torch.distributions import Dirichlet
import torch.nn.functional as F
import torch
from torch.nn import Module, Parameter, Embedding
from typing import List

from src.autoregressive_models.arm import AutoRegressiveModel
from external_models.gekcs.gekc_models import TractableKBCModel
from external_models.gekcs.models import KBCModel
from src.priors import Prior

class ARM(Module):
    def __init__(self,
                 arm_model: AutoRegressiveModel,
                 embedding_dim: int,
                 n_entities: int,
                 n_relations: int,
                 raw_frequencies: Tensor,
                 prior_config: dict,  # Replace prior_type with prior_config dict
                 ) -> None:
        super(ARM, self).__init__()
        self.embedding_dimension = embedding_dim
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.ent_emb = self._init_embedding(self.n_entities, self.embedding_dimension)
        self.rel_emb = self._init_embedding(self.n_relations, self.embedding_dimension)
        
        # Initialize prior with new unified class
        self.prior = Prior(
            n_entities=n_entities,
            raw_frequencies=raw_frequencies,
            init_weights=prior_config.get('init_weights'),
            optimize_temperature=prior_config.get('optimize_temperature', False),
            alpha=prior_config.get('alpha', 1.0)
        )
        
        # Tail prediction
        self.arm_model = arm_model

    @property
    def is_arm_model(self) -> bool:
        return isinstance(self.arm_model, AutoRegressiveModel)

    def _init_embedding(self, n_emb: int, emb_dim: int) -> Embedding:

        # test if works
        embedding = Embedding(n_emb, emb_dim)
        t = Dirichlet(torch.tensor([0.01] * emb_dim)
                      ).sample([embedding.weight.shape[0]])

        embedding.weight = Parameter(t)
        return embedding

    def scoring_function(self, h_idx: Tensor, r_idx: Tensor, t_idx: Tensor):
        """Compute the scoring function for the triplets given as argument:
        by applying convolutions to the concatenation of the embeddings. See
        referenced paper for more details on the score. See
        torchkge.models.interfaces.Models for more details on the API.

        """
        head = self.ent_emb(h_idx).unsqueeze(1)
        relation = self.rel_emb(r_idx).unsqueeze(1)
        relation_predictions, entity_predictions = self.arm_model((head, relation))
        prior_predictions = self.prior(h_idx.shape[0])

        return prior_predictions, relation_predictions, entity_predictions

    def forward(self, triple: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        heads, relations, tails = triple
        pos = self.scoring_function(heads, relations, tails)
        return pos


    def _compute_joint_probability(self, entity_idx, relation_idx, all_relation_predictions, all_entity_predictions):
        batch_size = entity_idx.shape[0]
        selected_relation_probs = all_relation_predictions.gather(1, relation_idx.unsqueeze(1))
        selected_entity_priors = self.prior(batch_size).gather(1, entity_idx.unsqueeze(1))
        return (selected_entity_priors * selected_relation_probs * all_entity_predictions).squeeze(1)

    def _get_predictions(self, entity_emb, relation_emb):
        entity_emb = entity_emb.unsqueeze(1)
        relation_emb = relation_emb.unsqueeze(1)
        return self.arm_model.forward((entity_emb, relation_emb))

    def inference_tail_prediction(self, h_idx: Tensor, r_idx: Tensor, t_idx: Tensor, candidates_mask=None) -> Tensor:
        h_emb, r_emb = self.inference_get_embeddings(h_idx, r_idx)
        all_relation_predictions, all_tail_predictions = self._get_predictions(h_emb, r_emb)
        scores = self._compute_joint_probability(h_idx, r_idx, all_relation_predictions, all_tail_predictions)
        
        # Filter scores by setting non-candidates to -inf
        if candidates_mask is not None:
            filtered_scores = scores.clone()
            filtered_scores[~candidates_mask] = float('-inf')
            return filtered_scores
        
        return scores

    def inference_head_prediction(self, h_idx: Tensor, r_idx: Tensor, t_idx: Tensor, candidates_mask=None) -> Tensor:
        inverse_r_idx = r_idx + int(self.n_relations / 2)
        t_emb, r_inv_emb = self.inference_get_embeddings(t_idx, inverse_r_idx)
        all_relation_predictions, all_head_predictions = self._get_predictions(t_emb, r_inv_emb)
        scores = self._compute_joint_probability(t_idx, inverse_r_idx, all_relation_predictions, all_head_predictions)
        
        # Filter scores by setting non-candidates to -inf
        if candidates_mask is not None:
            filtered_scores = scores.clone()
            filtered_scores[~candidates_mask] = float('-inf')
            return filtered_scores
        
        return scores

    def inference_get_embeddings(self, entity: torch.Tensor, relation: torch.Tensor):
        """Link prediction evaluation helper function. Get entities embeddings
        and relations embeddings. The output will be fed to the
        `inference_scoring_function` method. See torchkge.models.interfaces.Models for
        more details on the API.

        """

        t = self.ent_emb(entity)
        r_inv = self.rel_emb(relation)
        return t, r_inv



    def get_validation_loss(self, triple, n_relations):
        h_idx, r_idx, t_idx = triple
        r_inv = r_idx + n_relations
        predictions = self(triple)
        loss = self.criterion.log_loss(predictions=predictions, labels=triple)
        
        # Calculate inverse loss
        inverse_triple = (t_idx, r_inv, h_idx)
        inverse_predictions = self(inverse_triple)
        inverse_loss = self.criterion.log_loss(predictions=inverse_predictions, labels=inverse_triple)
        
        total_loss = (loss + inverse_loss).item()
        return total_loss

