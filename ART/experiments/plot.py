from experiments.experiment_runner import ExperimentRunner
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from scipy.stats import wasserstein_distance, ttest_ind
from src.utils import calculate_distribution, plot_distributions
import json
import pickle

class PlotRunner:
    def __init__(self, config):
        self.config = config
        self.plot_dir = f"./plots/{self.config['dataset']['class']}"
        os.makedirs(self.plot_dir, exist_ok=True)
        self.ALL_MODELS = ["complex", "arm_transformer", "complex2", "nbf"]
        self.MODEL_LABELS = {
            "complex": "ComplEx",
            "arm_transformer": "ART",
            "complex2": "ComplEx²",
            "nbf": "NBF"
        }
        self.experiment_runner = None
        
    def plot_precision_recall_curves(self):
        """Plot precision-recall curves for all models"""
        plt.figure(figsize=(10, 6))
        
        for model_type in self.ALL_MODELS:
            # Construct path to model's PR curve data
            model_dir = f"./experiments/plot_data/{self.config['dataset']['class'].lower()}/{model_type}"
            print(model_dir)
            pr_data_path = os.path.join(model_dir, 'pr_curve_data.pkl')
            
            if not os.path.exists(pr_data_path):
                print(f"Skipping {model_type} - no PR curve data found")
                continue
                
            # Load and plot PR curve with proper model label
            with open(pr_data_path, 'rb') as f:
                data = pickle.load(f)
            
            plt.plot(data['recall'], data['precision'], linewidth=2, label=self.MODEL_LABELS[model_type])
            
            # Optionally mark max F1 points
            f1_scores = 2 * (data['precision'] * data['recall']) / (data['precision'] + data['recall'] + 1e-8)
            max_f1_idx = np.argmax(f1_scores)
            plt.scatter(data['recall'][max_f1_idx], data['precision'][max_f1_idx], 
                       marker='x', zorder=5)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(self.plot_dir, 'precision_recall_curves_comparison.pdf')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_score_distribution(self, scores, labels):
        total_positives = np.sum(labels)
        k = int(total_positives / 2)  # This gives us the top half of positive samples

        # Get indices of top k scores
        top_k_indices = np.argpartition(scores, -k)[-k:]
        
        # Split the top k into two equal parts
        mid_point = k // 2
        top_25_indices = top_k_indices[-mid_point:]
        next_25_indices = top_k_indices[:-mid_point]

        # Get the scores for each group
        top_25_scores = scores[top_25_indices]
        next_25_scores = scores[next_25_indices]

        plt.figure(figsize=(10, 6))

        # Create boxplot with outliers turned off
        bp = plt.boxplot([next_25_scores, top_25_scores], 
                         labels=['50th to 75th percentile', 'Top 25%'],
                         showfliers=False)  # This line turns off outlier points

        plt.xlabel('Score Group')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)

        # Save the plot
        plot_path = os.path.join(self.plot_dir, 'score_distribution_analysis.pdf')
        plt.savefig(plot_path)
        plt.close()

        # Calculate statistics
        top_25_mean = np.mean(top_25_scores)
        next_25_mean = np.mean(next_25_scores)
        top_half_mean = np.mean(scores[top_k_indices])
        overall_mean = np.mean(scores)

        # Perform t-test
        t_statistic, p_value = ttest_ind(top_25_scores, next_25_scores)

        results = {
            "Top 25% Mean": float(top_25_mean),
            "50th-75th Percentile Mean": float(next_25_mean),
            "Top Half Mean": float(top_half_mean),
            "Overall Mean": float(overall_mean),
            "T-statistic": float(t_statistic),
            "P-value": float(p_value)
        }

        # Write results to file
        results_path = os.path.join(self.plot_dir, 'score_distribution_analysis_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def plot_kde(self, scores, labels):
        print("KDE")
        mask = scores != -float('Inf')
        filtered_scores = scores[mask]
        filtered_labels = labels[mask]

        true_scores = filtered_scores[filtered_labels == True]
        false_scores = filtered_scores[filtered_labels == False]

        plt.figure(figsize=(10, 6))
        sns.kdeplot(true_scores, shade=True, label='True', color='blue')
        sns.kdeplot(false_scores, shade=True, label='False', color='orange')

        plt.xlabel('Scores')
        plt.ylabel('Density')
        plt.legend(loc='upper right')
        plt.title(f'Density Plot of {self.config["model_type"]} Scores by Labels (True/False)')
        plt.show()
        plot_path = os.path.join(self.plot_dir, f'kde_{self.config["model_type"]}.pdf')
        plt.savefig(plot_path, format='pdf')
        plt.close()

    def plot_subject_distribution_shift(self):
        train_subjects = self.experiment_runner.dataset.kg_train.head_idx.numpy()
        test_subjects = self.experiment_runner.dataset.kg_test.head_idx.numpy()

        train_dist = calculate_distribution(train_subjects)
        test_dist = calculate_distribution(test_subjects)

        w_distance = wasserstein_distance(list(train_dist.keys()), list(test_dist.keys()),
                                          list(train_dist.values()), list(test_dist.values()))

        plot_path = os.path.join(self.plot_dir, 'subject_distribution_shift.png')
        plot_distributions(train_dist, test_dist, plot_path)

        return w_distance

    def run_plots(self):
        # scores, labels = self.experiment_runner.evaluate()
        scores, labels = None, None      
        if self.config['plot'] == "all":
            self.run_all_plots(scores, labels)
        else:
            self.run_single_plot(self.config['plot'], scores, labels)

    def run_all_plots(self, scores, labels):
        self.plot_precision_recall_curves()
        self.plot_score_distribution(scores, labels)
        self.plot_global_ranking(scores, labels)

    def run_single_plot(self, plot_type, scores, labels):
        match plot_type:
            case "precision-recall-curves":
                self.plot_precision_recall_curves()
            case "score-distribution":
                self.plot_score_distribution(scores, labels)
            case "kde":
                self.plot_kde(scores, labels)
            case "subject-distribution-shift":
                self.plot_subject_distribution_shift()
            case "global-ranking":
                self.plot_global_ranking(scores, labels)
            case _:
                raise ValueError("Plot type unknown")

def load_plot_data(model_type, dataset_class):
    data_path = f"./plots/{dataset_class}/{model_type}/score_distribution_analysis_data.json"
    with open(data_path, 'r') as f:
        return json.load(f)

def compare_models(model1_type, model2_type, dataset_class):
    data1 = load_plot_data(model1_type, dataset_class)
    data2 = load_plot_data(model2_type, dataset_class)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot for Model 1
    bp1 = ax1.boxplot([data1['bottom_half'], data1['top_half']], 
                      labels=['Bottom Half', 'Top Half'], patch_artist=True)
    ax1.set_title(f"{model1_type} Score Distribution")
    ax1.axhline(y=data1['bottom_mean'], color='blue', linestyle='--', label='Bottom Half Mean')
    ax1.axhline(y=data1['top_mean'], color='green', linestyle='--', label='Top Half Mean')
    ax1.axhline(y=data1['overall_mean'], color='red', linestyle='--', label='Overall Mean')

    # Plot for Model 2
    bp2 = ax2.boxplot([data2['bottom_half'], data2['top_half']], 
                      labels=['Bottom Half', 'Top Half'], patch_artist=True)
    ax2.set_title(f"{model2_type} Score Distribution")
    ax2.axhline(y=data2['bottom_mean'], color='blue', linestyle='--', label='Bottom Half Mean')
    ax2.axhline(y=data2['top_mean'], color='green', linestyle='--', label='Top Half Mean')
    ax2.axhline(y=data2['overall_mean'], color='red', linestyle='--', label='Overall Mean')

    # Color the boxes
    colors = ['lightblue', 'lightgreen']
    for bplot in (bp1, bp2):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    # Set common labels
    fig.text(0.5, 0.04, 'Score Distribution', ha='center', va='center')
    fig.text(0.06, 0.5, 'Score', ha='center', va='center', rotation='vertical')

    # Add a common legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)

    plt.tight_layout()
    plt.savefig(f'./plots/{dataset_class}/model_comparison_{model1_type}_vs_{model2_type}.pdf')
    plt.close()

if __name__ == "__main__":
    model1_type = "arm_transformer"  
    model2_type = "nbf"  
    dataset_class = "OGBLBioKG"  
    compare_models(model1_type, model2_type, dataset_class)