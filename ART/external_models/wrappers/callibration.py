import torch
from torch.nn import Module
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import pickle
import argparse
import os

class CalibrationModel(Module):
    def __init__(self, base_model=None, config=None) -> None:
        """Initialize with optional base_model and config for inference"""
        super(CalibrationModel, self).__init__()
        self.device = config['device'] if config else 'cpu'
        self.config = config
        self.base_model = base_model
        self.load_calibration_model()

    @classmethod
    def train_calibration(cls, positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> 'CalibrationWrapper':
        """Train calibration model from scores.
        
        Args:
            positive_scores: Tensor of scores for positive examples
            negative_scores: Tensor of scores for negative examples
            
        Returns:
            Trained CalibrationWrapper instance
        """
        instance = cls(base_model=None, config=None)
        
        P = len(positive_scores)
        N = len(negative_scores)
        alpha = P / (P + N)
        
        # Compute weights for balanced sampling
        w_positive = 1.0
        w_negative = 1.0 / alpha - 1.0
        
        # Combine scores and create labels/weights
        all_scores = torch.cat([positive_scores, negative_scores])
        all_labels = torch.cat([torch.ones(P), torch.zeros(N)])
        all_weights = torch.cat([
            torch.full((P,), w_positive),
            torch.full((N,), w_negative)
        ])
        
        # Fit calibration model
        X = all_scores.numpy().reshape(-1, 1)
        y = all_labels.numpy()
        weights = all_weights.numpy()
        
        instance.platt_scaler = LogisticRegression(solver='lbfgs')
        instance.platt_scaler.fit(X, y, sample_weight=weights)
        
        return instance

    def calibrate_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply calibration to raw scores."""
        if self.platt_scaler is None:
            raise ValueError("Calibration model not trained or loaded!")
            
        X = scores.cpu().numpy().reshape(-1, 1)
        calibrated_probs = self.platt_scaler.predict_proba(X)[:, 1]
        return torch.tensor(calibrated_probs, device=self.device, dtype=torch.float32)  # Explicitly set dtype to float32
    
    def inference_tail_prediction(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, 
                                candidates_mask: torch.Tensor = None) -> torch.Tensor:
        """Get calibrated tail prediction scores."""
        # Get raw scores from base model
        raw_scores = self.base_model.inference_tail_prediction(h, r, t, candidates_mask)
        
        # Calibrate scores (keeping -inf values for filtered entities)
        valid_mask = raw_scores != float('-inf')
        calibrated_scores = torch.full_like(raw_scores, float('-inf'), dtype=torch.float32)
        calibrated_scores[valid_mask] = self.calibrate_scores(raw_scores[valid_mask])
        
        return calibrated_scores
    
    def inference_head_prediction(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor,
                                candidates_mask: torch.Tensor = None) -> torch.Tensor:
        """Get calibrated head prediction scores."""
        # Get raw scores from base model
        raw_scores = self.base_model.inference_head_prediction(h, r, t, candidates_mask)
        
        # Calibrate scores (keeping -inf values for filtered entities)
        valid_mask = raw_scores != float('-inf')
        calibrated_scores = torch.full_like(raw_scores, float('-inf'))
        calibrated_scores[valid_mask] = self.calibrate_scores(raw_scores[valid_mask])
        
        return calibrated_scores
    
    def save_calibration_model(self):
        """Save calibration model to disk"""
        dataset_dir = os.path.join('./saved_models', self.config['dataset']['class'].lower(), "complex*")
        os.makedirs(dataset_dir, exist_ok=True)
        model_path = os.path.join(dataset_dir, 'calibration_model.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.platt_scaler, f)
        print(f"Saved calibration model to {model_path}")
    
    def load_calibration_model(self):
        """Load calibration model from disk"""
        dataset_dir = os.path.join('./saved_models', self.config['dataset']['class'].lower(), "complex*")
        path = os.path.join(dataset_dir, 'calibration_model.pkl')
        
        with open(path, 'rb') as f:
            self.platt_scaler = pickle.load(f)
        print(f"Loaded calibration model from {path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train calibration model')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., FB15K237)')
    
    args = parser.parse_args()
    
    # Construct paths
    dataset_dir = os.path.join('./saved_models', args.dataset.lower(), "complex")
    pos_scores_path = os.path.join(dataset_dir, 'true_scores_valid.pkl')
    neg_scores_path = os.path.join(dataset_dir, 'negative_scores_valid.pkl')
    
    # Load scores
    with open(pos_scores_path, 'rb') as f:
        true_scores = pickle.load(f)  # List of (tail_score, head_score) tuples
    with open(neg_scores_path, 'rb') as f:
        negative_scores = pickle.load(f)  # List of (tail_negs, head_negs) tuples
        
    # Convert to tensors - flatten all scores into a single list
    positive_scores = torch.tensor([score for pair in true_scores for score in pair])
    negative_scores = torch.tensor([score for pair in negative_scores for score in pair])
    
    print(f"Number of positive scores: {len(positive_scores)}")
    print(f"Number of negative scores: {len(negative_scores)}")
    
    # Train calibration model with renamed method
    calibration_model = CalibrationModel.train_calibration(positive_scores, negative_scores)
    
    # Print some model info
    print(f"Calibration model coefficients: {calibration_model.platt_scaler.coef_}")
    print(f"Calibration model intercept: {calibration_model.platt_scaler.intercept_}")
    
    # Save using existing method
    calibration_model.config = {'dataset': {'class': args.dataset}}  # Set minimal config needed for saving
    calibration_model.save_calibration_model()
