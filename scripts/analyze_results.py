
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

def load_multiple_results(results_dir):
    """Load results from multiple experiments"""
    results_files = glob(os.path.join(results_dir, '**/results.json'), recursive=True)
    all_results = []

    for result_file in results_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                data['result_path'] = os.path.dirname(result_file)
                data['experiment_name'] = os.path.basename(data['result_path'])
                all_results.append(data)
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
            continue

    return all_results


def filter_for_controlled_comparison(results_list, vary_param, fix_params=None):
    """
    Filter experiments for controlled comparison.

    Args:
        results_list: All loaded results
        vary_param: The parameter to vary (e.g., 'backbone', 'alpha', 'nclt')
        fix_params: Dict of parameters that should be fixed (e.g., {'seed': 1, 'alpha': 0.3})

    Returns:
        Filtered results where only vary_param changes
    """
    if fix_params is None:
        fix_params = {}

    filtered = []

    for result in results_list:
        hp = result.get('hyperparameters', {})

        # Check if all fixed params match
        match = True
        for key, value in fix_params.items():
            if key == 'alpha':
                result_value = hp.get('alpha', result.get('init_alpha', None))
            elif key == 'nclt':
                result_value = hp.get('nclt', hp.get('n_clt', None))
            elif key == 'seed':
                result_value = result.get('seed', None)
            elif key == 'backbone':
                result_value = result.get('backbone', None)
            else:
                result_value = hp.get(key, None)

            if result_value != value:
                match = False
                break

        if match:
            filtered.append(result)

    return filtered


def create_controlled_comparison_plot(results_list, vary_param, output_dir, param_name):
    """Create clean comparison plot for a single varying parameter"""
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    experiments = []
    for result in results_list:
        hp = result['hyperparameters']
        perf = result['performance']

        # Get the varying parameter value
        if vary_param == 'alpha':
            param_value = hp.get('alpha', result.get('init_alpha', 0.3))
        elif vary_param == 'nclt':
            param_value = hp.get('nclt', hp.get('n_clt', 16))
        elif vary_param == 'backbone':
            param_value = result.get('backbone', 'RN101')
        elif vary_param == 'seed':
            param_value = result.get('seed', 1)
        else:
            param_value = hp.get(vary_param, 'unknown')

        experiments.append({
            'Parameter': param_value,
            'Accuracy': perf['best_accuracy'],
            'Precision': perf['best_precision'],
            'Recall': perf['best_recall'],
            'F1-Score': perf['best_f1_score'],
            'Epoch': perf['best_epoch']
        })

    df = pd.DataFrame(experiments)
    df = df.sort_values('Parameter')

    # Save CSV
    df.to_csv(os.path.join(output_dir, f'{param_name}_comparison.csv'), index=False)

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{param_name} Ablation Study', fontsize=16, fontweight='bold')

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx // 2, idx % 2]

        x = range(len(df))
        y = df[metric].values

        ax.bar(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(df['Parameter'].values, rotation=45 if vary_param == 'backbone' else 0)
        ax.set_ylabel(f'{metric} (%)', fontsize=12)
        ax.set_title(f'{metric} vs {param_name}', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, val in enumerate(y):
            ax.text(i, val + 0.5, f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{param_name}_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Generate summary report
    with open(os.path.join(output_dir, f'{param_name}_summary.txt'), 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"{param_name.upper()} ABLATION STUDY - CONTROLLED COMPARISON\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Total experiments: {len(df)}\n\n")

        f.write(f"Results:\n")
        f.write(f"{'-'*80}\n")
        f.write(df.to_string(index=False))
        f.write(f"\n\n{'-'*80}\n")

        best_idx = df['Accuracy'].idxmax()
        f.write(f"\nBest Configuration:\n")
        f.write(f"  {vary_param}: {df.loc[best_idx, 'Parameter']}\n")
        f.write(f"  Accuracy: {df.loc[best_idx, 'Accuracy']:.2f}%\n")
        f.write(f"  Precision: {df.loc[best_idx, 'Precision']:.2f}%\n")
        f.write(f"  Recall: {df.loc[best_idx, 'Recall']:.2f}%\n")
        f.write(f"  F1-Score: {df.loc[best_idx, 'F1-Score']:.2f}%\n")
        f.write(f"  Best Epoch: {df.loc[best_idx, 'Epoch']}\n")

        f.write(f"\n{'='*80}\n")

    print(f"✅ {param_name} comparison complete: {len(df)} experiments analyzed")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--analysis_type', type=str, required=True, 
                       choices=['alpha', 'nclt', 'backbone', 'seed', 'dataset'])
    parser.add_argument('--fix_seed', type=int, default=1)
    parser.add_argument('--fix_alpha', type=float, default=0.3)
    parser.add_argument('--fix_nclt', type=int, default=16)
    parser.add_argument('--fix_backbone', type=str, default='RN101')
    args = parser.parse_args()

    # Load all results
    print(f"Loading results from {args.results_dir}...")
    all_results = load_multiple_results(args.results_dir)
    print(f"Found {len(all_results)} total experiments\n")

    # Define what to fix based on analysis type
    if args.analysis_type == 'alpha':
        fix_params = {'seed': args.fix_seed, 'nclt': args.fix_nclt, 'backbone': args.fix_backbone}
        vary_param = 'alpha'
        param_name = 'Alpha'

    elif args.analysis_type == 'nclt':
        fix_params = {'seed': args.fix_seed, 'alpha': args.fix_alpha, 'backbone': args.fix_backbone}
        vary_param = 'nclt'
        param_name = 'NCLT'

    elif args.analysis_type == 'backbone':
        fix_params = {'seed': args.fix_seed, 'alpha': args.fix_alpha, 'nclt': args.fix_nclt}
        vary_param = 'backbone'
        param_name = 'Backbone'

    elif args.analysis_type == 'seed':
        fix_params = {'alpha': args.fix_alpha, 'nclt': args.fix_nclt, 'backbone': args.fix_backbone}
        vary_param = 'seed'
        param_name = 'Seed'

    elif args.analysis_type == 'dataset':
        fix_params = {'seed': args.fix_seed, 'alpha': args.fix_alpha, 'nclt': args.fix_nclt, 'backbone': args.fix_backbone}
        vary_param = 'dataset'
        param_name = 'Dataset'

    # Filter for controlled comparison
    print(f"Filtering for {param_name} ablation...")
    print(f"Fixed parameters: {fix_params}\n")

    filtered_results = filter_for_controlled_comparison(all_results, vary_param, fix_params)

    if len(filtered_results) < 2:
        print(f"❌ ERROR: Need at least 2 experiments for {param_name} comparison, found {len(filtered_results)}")
        print(f"\nMake sure you have experiments with:"  )
        for key, val in fix_params.items():
            print(f"  {key} = {val}")
        return

    # Create comparison
    create_controlled_comparison_plot(filtered_results, vary_param, args.output_dir, param_name)


if __name__ == '__main__':
    main()
