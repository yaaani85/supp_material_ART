from torch.nn import Module
from external_models.gekcs.gekc_models import TractableKBCModel
from external_models.gekcs.models import KBCModel
import torch
from src.dataset import KGEDataset, OgbKGEDataset
class GEKCS(Module):

    def __init__(self, config) -> None:
        super(GEKCS, self).__init__()
        # Load model
        from external_models.gekcs.models import ComplEx
        from external_models.gekcs.gekc_models import SquaredComplEx
        self.device = config['device']
        model_class = SquaredComplEx if config['model_type'] == 'complex2' else ComplEx
        base_model = model_class(
            (config['dataset']['n_entities'], config['dataset']['n_relations'], config['dataset']['n_entities']),
            rank=config['embedding_dimension']
        ).to(self.device)
        base_model.load_state_dict(torch.load(config['model_path'], map_location=self.device)["weights"])
        
        # Set properties
        config['prediction_smoothing'] = 0.0
        config['label_smoothing'] = 0.0
        self.emb_dim = config['embedding_dimension'] * 2
        self.kbc_model = base_model
        self.number_of_entities = config['dataset']['n_entities']
        self.n_ent = config['dataset']['n_entities']
        if config['model_type'] == 'complex2':
            print("getting Z")
            self.Z = self.kbc_model.partition_function()[0]
        # Load dataset
        self.config = config
        self.dataset = self._load_dataset()

    @property
    def is_kbc_model(self) -> bool:
        return isinstance(self.kbc_model, KBCModel)

    def inference_tail_prediction(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, candidates_mask: torch.Tensor=None) -> torch.Tensor:
        queries = torch.cat((h.unsqueeze(1), r.unsqueeze(1), t.unsqueeze(1)), dim=1)
        with torch.no_grad():
            cs = self.kbc_model.get_candidates(0, self.number_of_entities, target="rhs")
            qs = self.kbc_model.get_queries(queries, target="rhs")

            if isinstance(self.kbc_model, TractableKBCModel):
                scores = self.kbc_model.eval_circuit_all(qs, cs)
                scores = scores / self.Z
            else:
                scores = qs @ cs

        if candidates_mask is not None:
            scores[~candidates_mask] = float('-inf')
        return scores

    def inference_head_prediction(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, candidates_mask: torch.Tensor=None) -> torch.Tensor:
        queries = torch.cat((h.unsqueeze(1), r.unsqueeze(1), t.unsqueeze(1)), dim=1)
        with torch.no_grad():
            cs = self.kbc_model.get_candidates(0, self.number_of_entities, target="lhs")
            qs = self.kbc_model.get_queries(queries, target="lhs")

            if isinstance(self.kbc_model, TractableKBCModel):
                scores = self.kbc_model.eval_circuit_all(qs, cs)
                scores = scores / self.Z
            else:
                scores = qs @ cs
        if candidates_mask is not None:
            scores[~candidates_mask] = float('-inf')
        return scores


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