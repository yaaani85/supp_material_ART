import json
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import timeit
import torch
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler

from src.config import get_model_path_suffix
from experiments.model_registry import ModelRegistry
from .descriptive_statistics import count_parameters, get_total_probability_mass, subject_distribution_shift, subject_given_relation_distribution
from src.utils import calculate_distribution, kl_divergence, plot_distributions,  get_results_path
from src.engine import Engine
from src.lp_evaluation import LinkPredictionEvaluator
from src.ir_evaluation import InformationRetrievalEvaluator
class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        model_output = ModelRegistry.create_model(config)
        
        self.model = model_output.model
        self.dataset = model_output.dataset
        self.engine = model_output.engine
        
        self.evaluator = InformationRetrievalEvaluator(
            config, self.model, self.dataset, "test"
        )

        print("kbc_evaluator init")
        

    def number_of_parameters(self):
        parameters = count_parameters(self.model)
        self._write_and_print("number_of_parameters", parameters)

    def link_prediction(self):
        metrics = self.evaluator.get_link_prediction_metrics()
        self._write_and_print("link_prediction_metrics", json.dumps(metrics, indent=2))

    def precision_at_k(self):
        results = self.evaluator.precision_at_k()
        self._write_and_print("precision_at_k", results)

    def recall_at_k(self):
        results = self.evaluator.recall_at_k()
        self._write_and_print("recall_at_k", results)

    def mean_average_precision(self):
        results = self.evaluator.mean_average_precision()
        self._write_and_print("mean_average_precision", results)

    def mean_average_precision_optimistic(self):
        results = self.evaluator.mean_average_precision_optimistic()
        self._write_and_print("mean_average_precision_optimistic", results)

    def metrics_at_max_f1_optimistic(self):
        results = self.evaluator.metrics_at_max_f1_optimistic()
        self._write_and_print("metrics_at_max_f1_optimistic", results)

    def metrics_at_max_f1(self):
        results = self.evaluator.metrics_at_max_f1()
        self._write_and_print("metrics_at_max_f1", results)

    def get_total_probability_mass(self):
        true_scores = self.evaluator.evaluate_lp()
        tpm = get_total_probability_mass(self.model, true_scores)
        self._write_and_print("probability_mass_analysis", tpm)

    def subject_distribution_shift(self):
        sdj = subject_distribution_shift(self.dataset)
        self._write_and_print("subject_distribution_shift", sdj)

    def subject_given_relation(self):
        sgr = subject_given_relation_distribution(self.dataset)
        print(sgr)
        self._write_and_print("subject_given_relation", sgr)

    def run_experiment(self):
        if self.config['experiment'] == "all":
            self.run_all_experiments()
        else:
            self.run_single_experiment(self.config['experiment'])

    def run_all_experiments(self):
        self.link_prediction()
        self.mean_average_precision()
        self.mean_average_precision_optimistic()
        self.metrics_at_max_f1_optimistic()
        self.metrics_at_max_f1()

    def run_single_experiment(self, experiment):
        match experiment:
            case "link-prediction":
                self.link_prediction()
            case "number-of-parameters":
                self.number_of_parameters()
            case "precision-at-k":
                self.precision_at_k()
            case "recall-at-k":
                self.recall_at_k()
            case "mean-average-precision":
                self.mean_average_precision()
            case "mean-average-precision-optimistic":
                self.mean_average_precision_optimistic()
            case "metrics-at-max-f1-optimistic":
                self.metrics_at_max_f1_optimistic()
            case "metrics-at-max-f1":
                self.metrics_at_max_f1()
            case "subject-distribution-shift":
                self.subject_distribution_shift()
            case "subject-given-relation":
                self.subject_given_relation()
            case "total-probability-mass":
                self.get_total_probability_mass()
            case "score-distribution-comparison":
                self.compare_score_distributions()
            case _:
                raise ValueError("Experiment Unknown")


    def _write_and_print(self, result_name: str, results: dict):
        """Write results to file and print them"""
        # Get the path for saving results
        results_path = get_results_path(self.config['model_path'], result_name)
        
        print(f"\nSaving results:")
        print(f"Result: {result_name}")
        print(f"Full path: {results_path}")
        
        # Create directory and save results
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Successfully saved results!\n")
        
        # Print results to console
        print("Results:")
        print(json.dumps(results, indent=2))


    @torch.no_grad()
    def compare_score_distributions(self):
        true_scores, false_scores = self.evaluator.evaluate_lp(save_negatives=-1)
        # Process scores based on model type
        true_processed = []
        false_processed = []
        
        if self.config['model_type'].lower() == 'complex2':
            for head_score, tail_score in true_scores:
                true_processed.append((head_score + tail_score) / 2)
            for head_score, tail_score in false_scores:
                false_processed.append((head_score + tail_score) / 2)
        elif self.config['model_type'].lower() == 'arm_transformer':
            for head_score, tail_score in true_scores:
                true_processed.append(head_score + tail_score)
            for head_score, tail_score in false_scores:
                false_processed.append(head_score + tail_score)
                
        # Calculate distribution metrics
        results = {
            "wasserstein_distance": float(wasserstein_distance(true_processed, false_processed)),
            "kl_divergence": float(kl_divergence(true_processed, false_processed)),
            "true_mean": f"{np.mean(true_processed):.2e}",
            "false_mean": f"{np.mean(false_processed):.2e}",
            "true_std": f"{np.std(true_processed):.2e}",
            "false_std": f"{np.std(false_processed):.2e}",
            "separation_ratio": f"{np.mean(true_processed) / np.mean(false_processed):.2e}"
        }
        
        self._write_and_print("score_distribution_comparison", results)
