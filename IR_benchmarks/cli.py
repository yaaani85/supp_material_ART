import click
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from query_evaluation.engine import QueryEvaluationEngine
from query_evaluation.dataset import QEDataset
from query_evaluation.custom_types import RankerType
from query_evaluation.factory import get_ranker
from query_evaluation.utils import save_results_to_file

@click.group(context_settings={'show_default': True})
def main() -> None:
    """The main entry point."""
    # Load environment variables first
    load_dotenv()
    
    # Configure logging based on environment variable
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(  # File output
                filename=Path('query_evaluation.log'),
                mode='a'
            )
        ]
    )
    
    # Log initial configuration
    logging.info(f"Starting query evaluation with log level: {log_level}")

@main.command(help="Evaluate a query answering dataset")
@click.option(
    "--ranker",
    type=click.Choice([r.value for r in RankerType]),
    help="The ranker(s) to use for evaluation. If not specified, all rankers will be used.",
    multiple=True,
    default=[r.value for r in RankerType],
    show_default=False,
)
@click.option(
    "--dataset",
    type=str,
    required=True,
    help="Path to the datasets directory",
)
@click.option(
    "--query-type",
    type=click.Choice(['1hop', '2hop', '3hop', '2i', '3i', '2i-1hop', '1hop-2i', 'all']),
    default='all',
    help="Specific query type to evaluate. Will look in 0qual directory.",
    show_default=True,
)
@click.option(
    "--output-file-path",
    type=str,
    default="results.json",
    help="Path to save the evaluation results",
    show_default=True,
)
@click.option(
    "--max-queries",
    type=int,
    default=None,
    help="Maximum number of queries to evaluate. Default is all queries.",
    show_default=True,
)
@click.option(
    "--write-per-query-metrics",
    is_flag=True,
    help="Write per-query metrics to files for each ranker",
    show_default=True,
)
def evaluate(dataset: str, ranker: tuple[str, ...], query_type: str, output_file_path: str, max_queries: int, write_per_query_metrics: bool) -> None:
    """Evaluate a query answering dataset"""
    repository_id = dataset
    
    # Split the file path into name and extension
    name, ext = os.path.splitext(output_file_path)
    
    # Modify output file path based on selections
    parts = [name]
    if set(ranker) != set(r.value for r in RankerType):
        parts.append('_'.join(ranker))
    if query_type != 'all':
        parts.append(query_type)
    parts.append(repository_id)
    output_file_path = f"{'_'.join(parts)}{ext}"
    
    logging.info(f"Evaluating dataset: {dataset}")
    qe_dataset: QEDataset = QEDataset(dataset, query_type=query_type)
    
    # Initialize all requested rankers
    rankers = [get_ranker(RankerType(r), qe_dataset, repository_id) for r in ranker]
    logging.info(f"Initialized rankers: {[r.__class__.__name__ for r in rankers]}")
    
    # Create and run evaluation engine
    engine = QueryEvaluationEngine(qe_dataset, rankers, max_queries=max_queries, write_to_file=write_per_query_metrics)
    metrics = engine.evaluate()
    
    # save_results_to_file(metrics, output_file_path)
    # logging.info(f"Results saved to {output_file_path}")