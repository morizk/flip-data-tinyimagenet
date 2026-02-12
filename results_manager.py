"""
Results Manager for structured experiment tracking and paper-ready exports.
"""
import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional, Any
import numpy as np


class ResultsManager:
    """Manages structured saving and loading of experiment results."""
    
    def __init__(self, results_dir='results'):
        """
        Args:
            results_dir: Root directory for results
        """
        self.results_dir = Path(results_dir)
        self.experiments_dir = self.results_dir / 'experiments'
        self.aggregated_dir = self.results_dir / 'aggregated'
        
        # Create directories
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.aggregated_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_experiment_id(self, architecture, dataset, flip_mode, fusion_type, use_augmentation):
        """
        Generate experiment ID: {arch}_{dataset}_{flipmode}_{fusion}_{aug}
        
        Args:
            architecture: Model architecture (e.g., 'resnet18')
            dataset: Dataset name (e.g., 'tinyimagenet')
            flip_mode: 'none', 'all', or 'inverted'
            fusion_type: 'early', 'late', or 'baseline'
            use_augmentation: Boolean
        
        Returns:
            Experiment ID string
        """
        aug_str = 'aug' if use_augmentation else 'noaug'
        fusion_str = fusion_type if fusion_type != 'baseline' else 'baseline'
        flip_str = flip_mode if flip_mode != 'none' else 'none'
        
        exp_id = f"{architecture}_{dataset}_{flip_str}_{fusion_str}_{aug_str}"
        return exp_id
    
    def save_experiment(self, experiment_id, config, metrics, history, model_path=None):
        """
        Save experiment results to structured directory.
        
        Args:
            experiment_id: Unique experiment identifier
            config: Experiment configuration dict
            metrics: Final metrics dict
            history: Per-epoch history dict
            model_path: Path to model checkpoint (optional)
        """
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config.json
        config_path = exp_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save metrics.json
        metrics_path = exp_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save history.json
        history_path = exp_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Copy model if provided
        if model_path and os.path.exists(model_path):
            import shutil
            dest_model_path = exp_dir / 'model.pth'
            shutil.copy2(model_path, dest_model_path)
        
        # Generate summary.txt
        summary_path = exp_dir / 'summary.txt'
        self._generate_summary(exp_dir, config, metrics, summary_path)
    
    def _generate_summary(self, exp_dir, config, metrics, summary_path):
        """Generate human-readable summary."""
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"EXPERIMENT SUMMARY: {exp_dir.name}\n")
            f.write("="*80 + "\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write("-"*80 + "\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\nMETRICS:\n")
            f.write("-"*80 + "\n")
            for key, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            
            f.write("\n" + "="*80 + "\n")
    
    def load_experiment(self, experiment_id):
        """
        Load experiment results.
        
        Args:
            experiment_id: Experiment identifier
        
        Returns:
            dict with 'config', 'metrics', 'history'
        """
        exp_dir = self.experiments_dir / experiment_id
        
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment {experiment_id} not found")
        
        config_path = exp_dir / 'config.json'
        metrics_path = exp_dir / 'metrics.json'
        history_path = exp_dir / 'history.json'
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        return {
            'config': config,
            'metrics': metrics,
            'history': history
        }
    
    def aggregate_results(self):
        """
        Aggregate all experiment results into single files.
        
        Returns:
            dict with all results
        """
        all_results = {}
        
        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir():
                try:
                    exp_data = self.load_experiment(exp_dir.name)
                    all_results[exp_dir.name] = {
                        'config': exp_data['config'],
                        'metrics': exp_data['metrics'],
                        'history': exp_data['history']
                    }
                except Exception as e:
                    print(f"Warning: Failed to load {exp_dir.name}: {e}")
        
        # Save aggregated results
        aggregated_path = self.aggregated_dir / 'all_results.json'
        with open(aggregated_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return all_results
    
    def export_to_csv(self, output_path=None):
        """
        Export results to CSV for easy analysis.
        
        Args:
            output_path: Optional output path (default: aggregated/comparison_table.csv)
        """
        if output_path is None:
            output_path = self.aggregated_dir / 'comparison_table.csv'
        
        all_results = self.aggregate_results()
        
        rows = []
        for exp_id, exp_data in all_results.items():
            config = exp_data['config']
            metrics = exp_data['metrics']
            
            row = {
                'experiment_id': exp_id,
                'architecture': config.get('architecture', ''),
                'dataset': config.get('dataset', ''),
                'fusion_type': config.get('fusion_type', ''),
                'flip_mode': config.get('flip_mode', ''),
                'use_augmentation': config.get('use_augmentation', False),
                'best_val_acc': metrics.get('best_val_acc', 0.0),
                'best_test_acc': metrics.get('best_test_acc', 0.0),
                'final_test_acc': metrics.get('final_test_acc', 0.0),
                'best_epoch': metrics.get('best_epoch', 0),
                'total_training_time_seconds': metrics.get('total_training_time_seconds', 0.0),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        return df
    
    def generate_paper_tables(self, output_path=None):
        """
        Generate paper-ready results in LaTeX format.
        
        Args:
            output_path: Optional output path (default: aggregated/paper_ready_results.json)
        """
        if output_path is None:
            output_path = self.aggregated_dir / 'paper_ready_results.json'
        
        all_results = self.aggregate_results()
        
        # Organize by architecture
        paper_results = {
            'main_table': [],
            'ablation_fusion': [],
            'ablation_flip_mode': [],
            'ablation_augmentation': []
        }
        
        for exp_id, exp_data in all_results.items():
            config = exp_data['config']
            metrics = exp_data['metrics']
            
            # Main table entry
            entry = {
                'architecture': config.get('architecture', ''),
                'fusion_type': config.get('fusion_type', 'baseline'),
                'flip_mode': config.get('flip_mode', 'none'),
                'use_augmentation': config.get('use_augmentation', False),
                'test_accuracy': metrics.get('best_test_acc', 0.0),
                'val_accuracy': metrics.get('best_val_acc', 0.0),
            }
            paper_results['main_table'].append(entry)
        
        # Save paper-ready results
        with open(output_path, 'w') as f:
            json.dump(paper_results, f, indent=2)
        
        return paper_results
    
    def generate_latex_tables(self, output_path=None):
        """
        Generate LaTeX table code for paper.
        
        Args:
            output_path: Optional output path for .tex file
        """
        df = self.export_to_csv()
        
        # Group by architecture
        latex_lines = []
        latex_lines.append("\\begin{table}[h]")
        latex_lines.append("\\centering")
        latex_lines.append("\\begin{tabular}{lcccc}")
        latex_lines.append("\\hline")
        latex_lines.append("Architecture & Fusion & Flip Mode & Aug & Test Acc \\\\")
        latex_lines.append("\\hline")
        
        for _, row in df.iterrows():
            arch = row['architecture']
            fusion = row['fusion_type']
            flip = row['flip_mode']
            aug = 'Yes' if row['use_augmentation'] else 'No'
            acc = f"{row['best_test_acc']:.2f}"
            
            latex_lines.append(f"{arch} & {fusion} & {flip} & {aug} & {acc} \\\\")
        
        latex_lines.append("\\hline")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\caption{Experimental Results}")
        latex_lines.append("\\label{tab:results}")
        latex_lines.append("\\end{table}")
        
        latex_code = "\n".join(latex_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(latex_code)
        
        return latex_code
    
    def list_experiments(self):
        """List all completed experiments."""
        experiments = []
        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir():
                experiments.append(exp_dir.name)
        return sorted(experiments)









