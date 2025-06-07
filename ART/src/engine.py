from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import torch
from tqdm import tqdm
import os

from src.utils import get_true_targets_batch, get_optimization_metric
from src.dataset import KGEDataset, OgbKGEDataset  
from src.criterion import LogLoss
from src.lp_evaluation import LinkPredictionEvaluator
from src.dataset import KGEDataset, OgbKGEDataset
from src.utils import save_checkpoint, get_prior_frequencies
from src.autoregressive_models.arm import get_arm_model
from src.models import ARM
class Engine:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        if self.config['model_type'] not in ['arm_transformer', 'arm_convolution']:
            raise ValueError("Only ARM models are currently supported for this engine")
        # Load dataset first since model needs it
        self.dataset = self._load_dataset()
        
        # Setup will handle model loading and other initializations
        self.setup()


    def _get_model(self, number_of_entities, with_inverse_relations):
        # Get frequencies from training data

        raw_frequencies = get_prior_frequencies(self.dataset.kg_train)
        
        # Get the autoregressive model based on config
        arm_model = get_arm_model(self.config, number_of_entities, with_inverse_relations)
        
        model = ARM(
            arm_model=arm_model,
            embedding_dim=self.config['embedding_dimension'],
            n_entities=number_of_entities,
            n_relations=with_inverse_relations,
            raw_frequencies=raw_frequencies,
            prior_config=self.config['prior']
        )
        
        # Load pretrained model if specified
        if self.config.get('use_pretrained_model'):
            model_path = self.config.get('model_path')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Pretrained model not found at: {model_path}")
            
            try:
                model.load_state_dict(torch.load(model_path), strict=False)
                print(f"Successfully loaded pretrained model from: {model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load model from {model_path}. Error: {str(e)}")
        
        model.to(self.device)
        return model

    def setup(self):
        number_of_entities = self.dataset.n_entities
        number_of_relations = self.dataset.n_relations
        number_of_inverse_relations = number_of_relations
        with_inverse_relations = number_of_relations + number_of_inverse_relations

        # Only set up ARM models
        if self.config['model_type'] in ['arm_transformer', 'arm_convolution']:
            self.model = self._get_model(number_of_entities, with_inverse_relations)
            self.criterion = LogLoss(
                self.config['prediction_smoothing'], 
                self.config['label_smoothing'],
                number_of_entities, 
                with_inverse_relations,
                self.config['prior']['alpha']
            )
            self.optimizer = Adam(self.model.parameters(), lr=self.config['lr'])
            
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                factor=self.config['factor'], 
                min_lr=1e-6, 
                patience=self.config['lr_patience'], 
                mode="max"  # Always maximizing now
            )
        else:
            raise ValueError(f"Engine only supports ARM models. Got: {self.config['model_type']}")

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0

        # Create a new dataloader for this epoch
        train_dataloader = self.dataset.get_dataloader(
            batch_size=self.config['batch_size'],
            split='train',
        )

        for batch in tqdm(train_dataloader):
            batch = [b.to(self.device) for b in batch]
            triple = batch[0], batch[1], batch[2]
            self.optimizer.zero_grad()
            predictions = self.model(triple)
            loss = self.criterion.log_loss(predictions=predictions, labels=triple)
            inverse_triple = batch[2], batch[1] + self.dataset.kg_train.n_rel, batch[0]
            inverse_predictions = self.model(inverse_triple)
            inverse_loss = self.criterion.log_loss(predictions=inverse_predictions, labels=inverse_triple)
            loss = loss + inverse_loss
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss

    def _step_scheduler(self, metrics):
        """Step scheduler based on configured metric"""
        step_value = get_optimization_metric(metrics, self.config['optimize'])
        self.scheduler.step(step_value)

    def train(self):
        if self.config['wandb']:
            wandb.init()

        current_patience = 0
        best_metric_value = float('-inf')  # Always maximizing
        best_validation_loss = float('inf')
        best_filtered_mrr = float('-inf')
        
        # Create model subdirectory only if we're going to save
        if self.config['save_model']:
            model_dir = os.path.dirname(self.config['model_dir'])
            os.makedirs(model_dir, exist_ok=True)
            
            # Define save paths relative to model_path directory
            self.config['save_path_mle'] = os.path.join(model_dir, "best_validation_loss.pt")
            self.config['save_path_mrr'] = os.path.join(model_dir, "best_filtered_mrr.pt")
        
        for epoch in range(self.config['epochs']):
            epoch_loss = self.train_epoch()
            self.model.eval()
            with torch.no_grad():
                if epoch % 1 == 0:
                    evaluator = LinkPredictionEvaluator(self.config, self.device, self.model, self.dataset, self.config['test_batch_size'], split='valid')
                    metrics = evaluator.get_link_prediction_metrics()
                    metrics['train_loss'] = epoch_loss
                    metrics['epoch'] = epoch
                    
                    self._step_scheduler(metrics)
                    metrics['lr'] = self.optimizer.param_groups[0]['lr']
                    
                    if self.config['wandb']:
                        wandb.log(metrics)
                    else:
                        print(metrics)
                    
                    current_metric = get_optimization_metric(metrics, self.config['optimize'])
                    if current_metric > best_metric_value:
                        best_metric_value = current_metric
                        current_patience = 0
                    else:
                        current_patience += 1

                    # Track both metrics independently
                    if metrics['validation_loss'] < best_validation_loss:
                        best_validation_loss = metrics['validation_loss']
                        print("LOWEST VALIDATION LOSS TILL NOW", best_validation_loss)
                        if self.config['save_model']:
                            save_checkpoint(self.model, self.config['save_path_mle'], 'VALIDATION LOSS', best_validation_loss)
                    
                    if metrics['filtered_mrr'] > best_filtered_mrr:
                        best_filtered_mrr = metrics['filtered_mrr']
                        print("HIGHEST FILTERED MRR TILL NOW", best_filtered_mrr)
                        if self.config['save_model']:
                            save_checkpoint(self.model, self.config['save_path_mrr'], 'FILTERED MRR', best_filtered_mrr)
                    
                    if current_patience > self.config['max_patience']:
                        print("STOPPED EARLY")
                        if self.config['wandb']:
                            wandb.finish()
                        return
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