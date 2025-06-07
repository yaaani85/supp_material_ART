import os
import pickle
import numpy as np
from tqdm import tqdm
from src.lp_evaluation import LinkPredictionEvaluator
from sklearn.metrics import precision_recall_curve

from src.utils import get_ir_dataloader
from external_models.wrappers.nbf import NBF

def ensure_evaluated(func):
    """Decorator to ensure evaluation is done before calculating metrics"""
    def wrapper(self, *args, **kwargs):
        if self.scores is None:
            print("Scores is None, running evaluate()...")  # Debug print
            self.evaluate()
        if self.scores is None:  # Double check
            raise ValueError("Scores is still None after evaluation!")
        return func(self, *args, **kwargs)
    return wrapper

class InformationRetrievalEvaluator(LinkPredictionEvaluator):
    """Evaluator for Knowledge Base Completion tasks using pre-computed benchmarks"""
    
    def __init__(self, config, model, dataset, split):
        super().__init__(config, model, dataset, split)
        self.n_relations = config['dataset']['n_relations']
        self.n_entities = config['dataset']['n_entities']
        self.batch_size = config['test_batch_size']
        self.device = config['device']
        self.model = model
        self.model.eval()
        self.scores = None

        # Load benchmark file
        self.test_loader, self.labels = get_ir_dataloader(config, dataset, split)
        exit()

    def evaluate(self):
        """Evaluate model on benchmark dataset using DataLoader"""
        tail_scores_list = []
        head_scores_list = []
         
        for batch_idx, (triples, tail_mask, head_mask, query_idx) in tqdm(
            enumerate(self.test_loader), 
            total=len(self.test_loader),
            desc="Evaluating",
            unit="batch",
            leave=True
        ):
            s, r, o = triples.unbind(1)
            
            # Get predictions
            tail_scores = self.model.inference_tail_prediction(s, r, o, candidates_mask=tail_mask.to(self.device)).detach().cpu().numpy()
            head_scores = self.model.inference_head_prediction(s, r, o, candidates_mask=head_mask.to(self.device)).detach().cpu().numpy()
            
            # Process entire batch at once
            tail_valid_mask = tail_scores != float('-inf')
            head_valid_mask = head_scores != float('-inf')
            
            # Add all valid scores from batch
            tail_scores_list.extend(tail_scores[tail_valid_mask])
            head_scores_list.extend(head_scores[head_valid_mask])
        # Combine scores and use pre-computed labels
        self.scores = np.array(tail_scores_list + head_scores_list)

        # NBF takes very long to run, save the results to a file
        if isinstance(self.model, NBF):
            with open(self.model.cache_file, 'wb') as f:
                pickle.dump({'scores': self.scores, 'labels': self.labels}, f)
        # Labels were already combined at init time

    @ensure_evaluated
    def precision_at_k(self):
        return self._calculate_at_k("precision")

    @ensure_evaluated
    def recall_at_k(self):
        return self._calculate_at_k("recall")

    @ensure_evaluated
    def mean_average_precision(self):
        return self._calculate_mean_average_precision()

    @ensure_evaluated
    def metrics_at_max_f1(self):
        return self._calculate_max_f1()

    @ensure_evaluated
    def metrics_at_max_f1_optimistic(self):
        return self._calculate_max_f1_optimistic()

    @ensure_evaluated
    def mean_average_precision_optimistic(self):
        return self._calculate_mean_average_precision_optimistic()
    
    @ensure_evaluated
    def precision_recall_curve(self):
        precision, recall, thresholds = precision_recall_curve(self.labels, self.scores)
        return precision, recall, thresholds

    def _calculate_max_f1(self):
        """Calculate optimal threshold using F1 score and save PR curve data"""
        precision, recall, thresholds = precision_recall_curve(self.labels, self.scores)
        
        # Save PR curve data
        pr_curve_data = {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'scores': self.scores,
            'labels': self.labels
        }
        
        # Save to model directory (assuming it's accessible through config)
        save_path = os.path.join(f"./saved_models/{self.config['dataset']['class'].lower()}/{self.config['model_type']}", 'pr_curve_data.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(pr_curve_data, f)
        
        # Calculate F1 scores
        f1_scores = np.zeros_like(precision)
        for i in range(len(precision)):
            if precision[i] + recall[i] > 0:
                f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            else:
                f1_scores[i] = 0.0
        
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx]
        # TEMP
        predicted_labels = (self.scores >= best_threshold).astype(int)
        true_positives = np.sum((predicted_labels == 1) & (self.labels == 1))
        false_positives = np.sum((predicted_labels == 1) & (self.labels == 0))
        false_negatives = np.sum((predicted_labels == 0) & (self.labels == 1))

        # Calculate metrics
        total_true_facts = np.sum(self.labels)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / total_true_facts
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results = {
            "threshold": float(best_threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "true_positives": int(true_positives),
            "false_positives": int(false_positives),
            "false_negatives": int(false_negatives),
            "total_predictions": int(true_positives + false_positives),
            "total_true_facts": int(total_true_facts),
            "accuracy": float(true_positives / total_true_facts)
        }
        print(results)
        return results
    
    def _calculate_max_f1_optimistic(self):
        # Calculate initial optimal threshold
        precision, recall, thresholds = precision_recall_curve(self.labels, self.scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold_idx = np.argmax(f1_scores)
        initial_best_threshold = thresholds[best_threshold_idx]
        initial_best_f1 = f1_scores[best_threshold_idx]

        # Sort scores and labels
        sorted_indices = np.argsort(-self.scores)
        sorted_scores = self.scores[sorted_indices]
        sorted_labels = self.labels[sorted_indices]

        # Relabel facts above threshold as true
        adjusted_labels = sorted_labels.copy()
        facts_relabeled = 0
        for i in range(len(sorted_scores)):
            if sorted_scores[i] >= initial_best_threshold:
                if adjusted_labels[i] == 0:
                    adjusted_labels[i] = 1
                    facts_relabeled += 1
            else:
                break  # Stop when we reach scores below the threshold

        # Recalculate metrics using adjusted labels
        new_precision, new_recall, new_thresholds = precision_recall_curve(adjusted_labels, sorted_scores)
        new_f1_scores = 2 * (new_precision * new_recall) / (new_precision + new_recall + 1e-8)
        new_best_threshold_idx = np.argmax(new_f1_scores)
        new_best_threshold = new_thresholds[new_best_threshold_idx]
        new_best_f1 = new_f1_scores[new_best_threshold_idx]

        results = [
            f"Initial Optimal Threshold: {initial_best_threshold:.4f}",
            f"Initial Best F1 Score: {initial_best_f1:.4f}",
            f"Number of facts relabeled as true: {facts_relabeled}",
            f"New Optimal Threshold after adjustment: {new_best_threshold:.4f}",
            f"New Best F1 Score after adjustment: {new_best_f1:.4f}",
            f"New Precision at optimal threshold: {new_precision[new_best_threshold_idx]:.4f}",
            f"New Recall at optimal threshold: {new_recall[new_best_threshold_idx]:.4f}",
        ]

        return results 

        return adjusted_labels, sorted_scores
    def _calculate_at_k(self, metric_type):
        # Move from ExperimentRunner
        total_positives = np.sum(self.labels)
        top_k = [int(total_positives/8), int(total_positives/4), int(total_positives/2), int(total_positives)]

        results = []
        for k in top_k:
            indices = np.argpartition(self.scores, -k)[-k:]
            topk_labels = self.labels[indices]
            topk_true_positives = np.sum(topk_labels)
      
            if metric_type == "precision":
                metric_value = topk_true_positives / k
                print(topk_true_positives, k)
                result = f"{metric_type.upper()} for top {k}: {metric_value:.4f}"
            else:  # recall
                metric_value = topk_true_positives / total_positives
                result = f"{metric_type.capitalize()}@{k}: {metric_value:.4f}"

            results.append(result)

        return "\n".join(results)

    def _calculate_mean_average_precision(self):
        # Move from ExperimentRunner
        sorted_indices = np.argsort(-self.scores)
        sorted_labels = self.labels[sorted_indices]

        precisions = []
        relevant_count = 0

        for k, label in enumerate(sorted_labels):
            if label == 1:
                relevant_count += 1
                precision_at_k = relevant_count / (k + 1)
                precisions.append(precision_at_k)

        average_precision = np.mean(precisions) if relevant_count > 0 else 0.0
        return f"Mean Average Precision: {average_precision:.4f}"

    def _calculate_mean_average_precision_optimistic(self):
        """Calculate MAP after relabeling facts above optimal F1 threshold as true"""
        # Calculate initial optimal threshold
        precision, recall, thresholds = precision_recall_curve(self.labels, self.scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold_idx = np.argmax(f1_scores)
        initial_best_threshold = thresholds[best_threshold_idx]

        # Sort scores and labels
        sorted_indices = np.argsort(-self.scores)
        sorted_scores = self.scores[sorted_indices]
        sorted_labels = self.labels[sorted_indices]

        # Relabel facts above threshold as true
        adjusted_labels = sorted_labels.copy()
        facts_relabeled = 0
        for i in range(len(sorted_scores)):
            if sorted_scores[i] >= initial_best_threshold:
                if adjusted_labels[i] == 0:
                    adjusted_labels[i] = 1
                    facts_relabeled += 1
            else:
                break

        # Calculate MAP with adjusted labels
        precisions = []
        relevant_count = 0

        for k, label in enumerate(adjusted_labels):
            if label == 1:
                relevant_count += 1
                precision_at_k = relevant_count / (k + 1)
                precisions.append(precision_at_k)

        average_precision = np.mean(precisions) if relevant_count > 0 else 0.0
        
        results = [
            f"Original MAP: {self._calculate_mean_average_precision().split(': ')[1]}",
            f"Facts relabeled as true: {facts_relabeled}",
            f"Optimistic MAP: {average_precision:.4f}"
        ]
        
        return "\n".join(results)

