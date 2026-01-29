import json
import os
import matplotlib.pyplot as plt
import numpy as np
from reproduce.train_all import QUERY_TYPE_MAPPING

datasets = ['FB15k', 'FB15k-237', 'NELL']
model_map = {
    'FB15k': 'FB15k-model-rank-1000-epoch-100-1602520745',
    'FB15k-237': 'FB15k-237-model-rank-1000-epoch-100-1602508358',
    'NELL': 'NELL-model-rank-1000-epoch-100-1602499096'
}

def load_json_results(dataset, method='co', metric_name='MRRm_new', debug=False):
    """Load JSON results for all query types for a dataset."""
    results = []
    model_name = model_map[dataset]
    for col, qtype in QUERY_TYPE_MAPPING.items():
        co_folder = 'co_debug' if debug else 'co'
        if method == 'co':
            fname = f"results/reproduce/{co_folder}/cont_n={model_name}_t={qtype}_r=None_m=test_lr=0.1_opt=adam_ms=1000.json"
        else:  # beam is always in debug since the dataset is too big for combinatorial search
            topk = 8
            if dataset == 'NELL':
                topk = 8 if qtype in ['1_2', '2_2', '2_2_disj', '2_3', '4_3_disj'] else 5
            fname = f"results/reproduce/beam/topk_d={dataset}_t=min_e={qtype}_rank=1000_k={topk}_sn=0.json"
            if qtype == '1_1':
                fname = f"results/reproduce/{co_folder}/cont_n={model_name}_t=1_1_r=None_m=test_lr=0.1_opt=adam_ms=1000.json"
        
        if os.path.exists(fname):
            with open(fname) as f:
                results.append(json.load(f).get(metric_name, 0))
        else:
            results.append(0)
    return results

def print_table_with_diff(datasets, methods, metrics):
    """Print single table with all metrics and delta rows."""
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"{dataset}")
        print(f"{'='*70}")
        print(f"{'Metric/Method':<20} {'Avg':>7} " + " ".join([f"{q:>7}" for q in QUERY_TYPE_MAPPING.keys()]))
        print("-" * 70)
        
        for metric_name in metrics:
            all_results = {}
            for method_name, method_key in methods:
                results = load_json_results(dataset, method_key, metric_name=metric_name, debug=True)
                all_results[method_name] = results
                avg = sum(results) / len(results) if results else 0
                label = f"{metric_name[:3]}-{method_name.split('-')[1]}"
                # print(f"{label:<20} {avg:>7.3f} " + " ".join([f"{r:>7.3f}" for r in results]))
            
            # Add difference row
            if len(all_results) == 2:
                method_names = list(all_results.keys())
                if 'time' in metric_name.lower():
                    diff = [all_results[method_names[0]][i] / all_results[method_names[1]][i] if all_results[method_names[1]][i] != 0 else 0 for i in range(len(all_results[method_names[0]]))]
                    avg_diff = sum(diff) / len(diff) if diff else 0

                    # Invert to get speedup
                    avg_diff = 1 / avg_diff if avg_diff != 0 else 0
                    diff = [1 / d if d != 0 else 0 for d in diff]
                    print(f"{metric_name[:3] + '-x':<20} {avg_diff:>7.3f} " + " ".join([f"{d:>7.3f}" for d in diff]))
                else:
                    diff = [all_results[method_names[0]][i] - all_results[method_names[1]][i] for i in range(len(all_results[method_names[0]]))]
                    avg_diff = sum(diff) / len(diff) if diff else 0
                    print(f"{metric_name[:3] + '-Δ':<20} {avg_diff:>+7.3f} " + " ".join([f"{d:>+7.3f}" for d in diff]))
            # print()

def print_table(datasets, methods, metric_name='MRRm_new'):
    """Print table of results."""
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"{dataset}")
        print(f"{'='*70}")
        print(f"{'Method':<15} {'Avg':>7} " + " ".join([f"{q:>7}" for q in QUERY_TYPE_MAPPING.keys()]))
        print("-" * 70)
        
        for method_name, method_key in methods:
            results = load_json_results(dataset, method_key, metric_name=metric_name)
            avg = sum(results) / len(results) if results else 0
            print(f"{method_name:<15} {avg:>7.3f} " + " ".join([f"{r:>7.3f}" for r in results]))

def plot_histogram(datasets, method='co'):
    """Plot bar histogram with queries grouped by dataset."""
    data = {ds: load_json_results(ds, method) for ds in datasets}
    
    x = np.arange(len(QUERY_TYPE_MAPPING))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, ds in enumerate(datasets):
        ax.bar(x + i * width, data[ds], width, label=ds)
    
    ax.set_xlabel('Query Type')
    ax.set_ylabel('MRRm_new')
    ax.set_title(f'Results by Query Type and Dataset ({"CQD-CO" if method == "co" else "CQD-Beam"})')
    ax.set_xticks(x + width)
    ax.set_xticklabels(QUERY_TYPE_MAPPING.keys())
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Print combined table
print_table_with_diff(datasets, [('CQD-CO', 'co'), ('CQD-Beam', 'beam')], metrics=['MRRm_new', 'time_taken'])

# Print table
# print_table(datasets, [('CQD-CO', 'co'), ('CQD-Beam', 'beam')], metric_name='MRRm_new')

# Plot histogram
# plot_histogram(datasets, method='co')
