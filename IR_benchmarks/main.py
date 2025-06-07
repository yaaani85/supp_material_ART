import os
from create_kbc_benchmark import KGEBenchmarkCreator

def create_benchmark_creator(dataset: str, data_dir: str, is_ogb: bool) -> KGEBenchmarkCreator:
    """Factory function to create the appropriate benchmark creator."""
    if is_ogb:
        dataset_name = f"ogbl-{dataset}" if not dataset.startswith('ogbl-') else dataset
        dataset_path = os.path.join(data_dir, dataset_name.replace('-', '_'))
        return KGEBenchmarkCreator.from_ogb(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            processed_path=os.path.join(dataset_path, 'processed')
        )
    
    return KGEBenchmarkCreator(
        train_path=os.path.join(data_dir, dataset, 'train.txt'),
        valid_path=os.path.join(data_dir, dataset, 'valid.txt'),
        test_path=os.path.join(data_dir, dataset, 'test.txt')
    )

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create KBC Benchmark')
    parser.add_argument('--dataset', type=str, help='Dataset name (e.g., FB14k-237, biokg)')
    parser.add_argument('--ogb', action='store_true', help='Use OGB dataset format')
    parser.add_argument('--data_dir', type=str, default='../data', 
                       help='Base directory containing the datasets')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for the benchmark files')
    
    args = parser.parse_args()
    
    # If no output directory is specified, use current directory + dataset name
    if args.output_dir is None:
        args.output_dir = os.path.join('.', args.dataset)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create and save benchmark
    creator = create_benchmark_creator(args.dataset, args.data_dir, args.ogb)
    output_path = os.path.join(args.output_dir, args.dataset)
    # creator.save_benchmark(output_path, split="validation")
    creator.save_benchmark(output_path, split="test")
    print(f"Benchmark saved to {output_path}")

if __name__ == "__main__":
    main()