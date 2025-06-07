import torch
import torch.nn.functional as F

class LogLoss:
    def __init__(self,
                 prediction_smoothing: float,
                 label_smoothing: float,
                 number_of_entities: int,
                 number_of_relations: int,
                 alpha: int = 1):

        self.prediction_smoothing = prediction_smoothing
        self.label_smoothing = label_smoothing
        self.number_of_entities = number_of_entities
        self.number_of_relations = number_of_relations
        self.use_head = alpha

    def log_loss(self,
                 predictions: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                 labels: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:

        subject, relation, object_ = labels
        subject_pred, relation_pred, object_pred = predictions

        # p(s, r, o) = p(s) * p(r|s) * p(o|r,s)
        log_probs = []
        
        # p(s): probability of subject (optional based on alpha)
        if self.use_head:
            log_p_s = self.log_categorical(subject, subject_pred, self.number_of_entities)
            log_probs.append(log_p_s)
        
        # p(r|s): probability of relation given subject
        log_p_r = self.log_categorical(relation, relation_pred, self.number_of_relations)
        log_probs.append(log_p_r)
        
        # p(o|r,s): probability of object given relation and subject
        log_p_o = self.log_categorical(object_, object_pred, self.number_of_entities)
        log_probs.append(log_p_o)
        
        # Sum log probabilities to get joint probability
        log_prob_triple = torch.stack([p.unsqueeze(-1) for p in log_probs], dim=1).sum(dim=1)
        batch_loss = (-log_prob_triple).sum()

        return batch_loss

    def log_categorical(self, x: torch.Tensor, p: torch.Tensor, num_classes) -> torch.Tensor:
        '''Function written by J. Tomczak (DeepGenerativeModelling) '''
        x_one_hot: torch.Tensor = F.one_hot(x.long(), num_classes=num_classes)

        x_one_hot = ((1 - self.label_smoothing) * x_one_hot) + (1 / x_one_hot.shape[1])
        eps = self.prediction_smoothing

        log_p: torch.Tensor = x_one_hot * torch.log(torch.clamp(p, eps, 1. - eps))

        return torch.sum(log_p, 1)
