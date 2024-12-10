import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class HyperParamResult:
    steps: int
    learning_rate: float
    memory_depth: int
    cutoff_dim: int
    mae: float
    rmse: float
    r2: float
    corr: float

def parse_hyperparameter_results(content: str) -> List[HyperParamResult]:
    results = []
    current_params = None
    
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('Model: qnn_hp_'):
            # If we have a complete set of parameters, add it to results
            if current_params and all(k in current_params for k in ['steps', 'learning_rate', 'memory_depth', 'cutoff_dim', 'mae', 'rmse', 'r2', 'corr']):
                results.append(HyperParamResult(**current_params))
            
            # Parse model name to get parameters
            parts = line.split('qnn_hp_s')[1].split('_')
            steps = int(parts[0])
            lr = float(parts[1].split('lr')[1])
            md = int(parts[2].split('md')[1])
            cd = int(parts[3].split('cd')[1])
            
            current_params = {
                'steps': steps,
                'learning_rate': lr,
                'memory_depth': md,
                'cutoff_dim': cd
            }
        elif current_params is not None:
            if line.startswith('mae:'):
                current_params['mae'] = float(line.split(': ')[1])
            elif line.startswith('rmse:'):
                current_params['rmse'] = float(line.split(': ')[1])
            elif line.startswith('r2:'):
                current_params['r2'] = float(line.split(': ')[1])
            elif line.startswith('corr:'):
                current_params['corr'] = float(line.split(': ')[1])
                # After getting correlation, this set is complete
                if all(k in current_params for k in ['steps', 'learning_rate', 'memory_depth', 'cutoff_dim', 'mae', 'rmse', 'r2', 'corr']):
                    results.append(HyperParamResult(**current_params))
                    current_params = None
    
    # Add the last set of parameters if it's complete
    if current_params and all(k in current_params for k in ['steps', 'learning_rate', 'memory_depth', 'cutoff_dim', 'mae', 'rmse', 'r2', 'corr']):
        results.append(HyperParamResult(**current_params))
    
    return results

def parse_model_comparison_results(content: str) -> pd.DataFrame:
    models = {}
    current_model = None
    metrics = {}
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('Model:'):
            if current_model and metrics:
                models[current_model] = metrics.copy()
            current_model = line.split('Model: ')[1]
            metrics = {}
        elif line.startswith('Metrics:'):
            continue
        elif line.startswith('accuracy:'):
            continue
        elif any(line.startswith(metric) for metric in ['mae:', 'rmse:', 'r2:', 'corr:', 'marpd:', 'mdae:']):
            key, value = line.split(': ')
            metrics[key] = float(value)
            
    # Add the last model
    if current_model and metrics:
        models[current_model] = metrics.copy()
        
    return pd.DataFrame.from_dict(models, orient='index')

def plot_hyperparameter_heatmaps(df: pd.DataFrame):
    params = ['learning_rate', 'memory_depth', 'cutoff_dim', 'steps']
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    plot_idx = 0
    for p1, p2 in [(params[i], params[j]) for i in range(len(params)) 
                   for j in range(i+1, len(params))]:
        if plot_idx < 6:
            pivot = df.pivot_table(values='rmse', 
                                 index=p1, 
                                 columns=p2, 
                                 aggfunc='mean')
            
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis_r', 
                       ax=axes[plot_idx])
            axes[plot_idx].set_title(f'RMSE: {p1} vs {p2}')
            
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('hyperparameter_heatmaps.png')
    plt.close()

def plot_model_comparison(df: pd.DataFrame):
    # Plot RMSE comparison
    plt.figure(figsize=(15, 6))
    df['rmse'].plot(kind='bar')
    plt.title('RMSE Comparison Across Models')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('model_rmse_comparison.png')
    plt.close()
    
    # Plot R² comparison
    plt.figure(figsize=(15, 6))
    df['r2'].plot(kind='bar')
    plt.title('R² Comparison Across Models')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('R²')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('model_r2_comparison.png')
    plt.close()

def analyze_models(df: pd.DataFrame):
    print("\n=== Model Comparison Analysis ===")
    
    # Analyze MLP models
    mlp_models = df[df.index.str.startswith('mlp')]
    print("\nMLP Models Performance:")
    print(mlp_models[['rmse', 'r2']].to_string())
    
    # Analyze QNN models with uncertainty quantification
    qnn_uq = df[df.index.str.startswith('qnn_uq')]
    print("\nQNN UQ Models Performance:")
    print(qnn_uq[['rmse', 'r2']].to_string())
    
    # Analyze Selective QNN models
    selective_models = df[df.index.str.startswith('qnn_selective')]
    print("\nSelective QNN Models Performance:")
    print(selective_models[['rmse', 'r2']].to_string())
    
    return mlp_models, qnn_uq, selective_models

def plot_selective_qnn_analysis(df: pd.DataFrame):
    # Filter selective QNN models
    selective_models = df[df.index.str.startswith('qnn_selective')].copy()
    
    # Extract threshold and samples from model names
    selective_models['threshold'] = selective_models.index.str.extract(r't(0\.\d)')[0].astype(float)
    selective_models['samples'] = selective_models.index.str.extract(r's(\d+)')[0].astype(int)
    
    plt.figure(figsize=(12, 6))
    for samples in sorted(selective_models['samples'].unique()):
        mask = selective_models['samples'] == samples
        plt.plot(selective_models[mask]['threshold'], 
                selective_models[mask]['rmse'], 
                marker='o', 
                label=f'{samples} samples')
    
    plt.xlabel('Threshold')
    plt.ylabel('RMSE')
    plt.title('Selective QNN Performance vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('selective_qnn_analysis.png')
    plt.close()

def analyze_hyperparameter_impact(results: List[HyperParamResult]):
    df = pd.DataFrame([vars(r) for r in results])
    
    best_rmse = df.loc[df['rmse'].idxmin()]
    
    print("=== Hyperparameter Analysis ===")
    print("\nBest Configuration:")
    print(f"Steps: {best_rmse['steps']}")
    print(f"Learning Rate: {best_rmse['learning_rate']}")
    print(f"Memory Depth: {best_rmse['memory_depth']}")
    print(f"Cutoff Dimension: {best_rmse['cutoff_dim']}")
    print(f"RMSE: {best_rmse['rmse']:.6f}")
    print(f"R²: {best_rmse['r2']:.6f}")
    
    # Create parameter impact plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, param in enumerate(['steps', 'learning_rate', 'memory_depth', 'cutoff_dim']):
        means = df.groupby(param)['rmse'].mean()
        stds = df.groupby(param)['rmse'].std()
        
        axes[idx].errorbar(means.index, means.values, yerr=stds.values, 
                          marker='o', linestyle='-')
        axes[idx].set_title(f'{param.replace("_", " ").title()} vs RMSE')
        axes[idx].grid(True)
        
    plt.tight_layout()
    plt.savefig('parameter_impact.png')
    plt.close()
    
    return df

# Read and analyze files
print("Starting analysis...")

with open('hyperparameter_results_20241128_175629.txt', 'r') as f:
    hp_content = f.read()
    
with open('model_comparison_20241128_182307.txt', 'r') as f:
    mc_content = f.read()

# Perform analysis and create plots
hp_results = parse_hyperparameter_results(hp_content)
df_hyperparams = analyze_hyperparameter_impact(hp_results)
plot_hyperparameter_heatmaps(df_hyperparams)

# Model comparison analysis
df_models = parse_model_comparison_results(mc_content)
mlp_models, qnn_uq, selective_models = analyze_models(df_models)
plot_model_comparison(df_models)
plot_selective_qnn_analysis(df_models)

print("\nAnalysis complete. The following plots have been saved:")
print("1. hyperparameter_heatmaps.png - Heatmaps showing interactions between hyperparameters")
print("2. parameter_impact.png - Individual parameter impact on RMSE")
print("3. model_rmse_comparison.png - RMSE comparison across all models")
print("4. model_r2_comparison.png - R² comparison across all models")
print("5. selective_qnn_analysis.png - Analysis of selective QNN performance")