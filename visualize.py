import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os


def load_results(filename=None):
    """Load results from JSON file."""
    if filename is None:
        # Find the most recent results file
        result_files = glob.glob('results_*.json')
        if not result_files:
            raise FileNotFoundError("No results files found. Run train.py first.")
        filename = max(result_files, key=os.path.getctime)
        print(f"Loading results from: {filename}")
    
    with open(filename, 'r') as f:
        results = json.load(f)
    
    return results


def plot_training_curves(results, save_path='training_curves.png'):
    """Plot training curves for all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Curves Comparison', fontsize=16, fontweight='bold')
    
    epochs = None
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for exp_name, exp_data in results.items():
        history = exp_data['history']
        if epochs is None:
            epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], label=exp_name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Training Accuracy
    ax = axes[0, 1]
    for exp_name, exp_data in results.items():
        history = exp_data['history']
        ax.plot(epochs, history['train_acc'], label=exp_name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy (%)')
    ax.set_title('Training Accuracy')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Validation Accuracy
    ax = axes[1, 0]
    for exp_name, exp_data in results.items():
        history = exp_data['history']
        ax.plot(epochs, history['val_acc'], label=exp_name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Validation Accuracy')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Test Accuracy
    ax = axes[1, 1]
    for exp_name, exp_data in results.items():
        history = exp_data['history']
        ax.plot(epochs, history['test_acc'], label=exp_name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_comparison_bar(results, save_path='comparison_bar.png'):
    """Plot bar chart comparing final test accuracies."""
    exp_names = []
    test_accs = []
    
    for exp_name, exp_data in results.items():
        exp_names.append(exp_name)
        test_accs.append(exp_data['final_test_acc'])
    
    # Sort by accuracy
    sorted_data = sorted(zip(exp_names, test_accs), key=lambda x: x[1], reverse=True)
    exp_names, test_accs = zip(*sorted_data)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(range(len(exp_names)), test_accs, color='steelblue', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, test_accs)):
        ax.text(acc + 0.1, i, f'{acc:.2f}%', va='center', fontweight='bold')
    
    ax.set_yticks(range(len(exp_names)))
    ax.set_yticklabels(exp_names, fontsize=10)
    ax.set_xlabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Final Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison bar chart saved to {save_path}")
    plt.close()


def plot_grouped_comparison(results, save_path='grouped_comparison.png'):
    """Plot grouped comparison by experiment type."""
    # Group experiments
    baseline = []
    flip_all_late = []
    flip_all_early = []
    flip_any_late = []
    flip_any_early = []
    
    for exp_name, exp_data in results.items():
        acc = exp_data['final_test_acc']
        if 'Baseline' in exp_name:
            baseline.append(('no aug' if 'no aug' in exp_name else 'aug', acc))
        elif 'Flip-All Late' in exp_name:
            flip_all_late.append(('no aug' if 'no aug' in exp_name else 'aug', acc))
        elif 'Flip-All Early' in exp_name:
            flip_all_early.append(('no aug' if 'no aug' in exp_name else 'aug', acc))
        elif 'Flip-Any Late' in exp_name:
            flip_any_late.append(('no aug' if 'no aug' in exp_name else 'aug', acc))
        elif 'Flip-Any Early' in exp_name:
            flip_any_early.append(('no aug' if 'no aug' in exp_name else 'aug', acc))
    
    # Prepare data for grouped bar chart
    categories = ['Baseline', 'Flip-All\nLate Fusion', 'Flip-All\nEarly Fusion', 
                  'Flip-Any\nLate Fusion', 'Flip-Any\nEarly Fusion']
    no_aug_accs = []
    aug_accs = []
    
    for group in [baseline, flip_all_late, flip_all_early, flip_any_late, flip_any_early]:
        no_aug = next((acc for mode, acc in group if mode == 'no aug'), 0)
        aug = next((acc for mode, acc in group if mode == 'aug'), 0)
        no_aug_accs.append(no_aug)
        aug_accs.append(aug)
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - width/2, no_aug_accs, width, label='No Augmentation', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, aug_accs, width, label='With Augmentation', alpha=0.8, color='coral')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Experiment Type', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy Comparison by Experiment Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Grouped comparison saved to {save_path}")
    plt.close()


def print_summary_table(results):
    """Print a summary table of results."""
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(f"{'Experiment Name':<40} {'Best Val Acc':<15} {'Best Test Acc':<15} {'Final Test Acc':<15}")
    print("-"*80)
    
    for exp_name, exp_data in results.items():
        print(f"{exp_name:<40} {exp_data['best_val_acc']:>13.2f}%  {exp_data['best_test_acc']:>13.2f}%  {exp_data['final_test_acc']:>13.2f}%")
    
    print("="*80)


if __name__ == '__main__':
    # Load results
    try:
        results = load_results()
    except FileNotFoundError as e:
        print(e)
        exit(1)
    
    # Print summary
    print_summary_table(results)
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_training_curves(results)
    plot_comparison_bar(results)
    plot_grouped_comparison(results)
    
    print("\nAll visualizations generated successfully!")




