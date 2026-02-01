#!/usr/bin/env python3
import subprocess
import sys
import json
import os
import os.path as osp
from pathlib import Path

HYPERPARAM_GRID = {
	'learning_rate': [0.01, 0.1],
	'density_reg': [1e-4, 1e-5],
}

QUERY_TYPES = {
	'1p': '1_1',
	'2p': '1_2',
	'3p': '1_3',
	'2i': '2_2',
	'3i': '2_3',
	'pi': '3_3',
	'ip': '4_3',
	'2u': '2_2_disj',
	'up': '4_3_disj'
}

EXPERIMENTS = {
	# 'FB15k': {
	# 	'model': 'models/FB15k-model-rank-1000-epoch-100-1602520745.pt',
	# 	'data': 'data/FB15k',
	# },
	'FB15k-237': {
		'model': 'models/FB15k-237-model-rank-1000-epoch-100-1602508358.pt',
		'data': 'data/FB15k-237',
	},
	# 'NELL': {
	# 	'model': 'models/NELL-model-rank-1000-epoch-100-1602499096.pt',
	# 	'data': 'data/NELL',
	# },
}

def run_hyperparam_search(dataset, conditional):
	if dataset not in EXPERIMENTS:
		print(f"Dataset {dataset} not found. Available: {list(EXPERIMENTS.keys())}")
		sys.exit(1)
	
	config = EXPERIMENTS[dataset]
	cond_str = 'cond' if conditional else 'uncond'
	results_dir = Path(f'results/hyperparam_search_{dataset}_{cond_str}')
	results_dir.mkdir(parents=True, exist_ok=True)
	
	max_steps = 100
	optimal_params = {}
	
	for query_name, chain_type in QUERY_TYPES.items():
		print(f"\n=== {dataset} ({cond_str}) - Query: {query_name} ===")
		query_results = {}
		
		for lr in HYPERPARAM_GRID['learning_rate']:
			for reg in HYPERPARAM_GRID['density_reg']:
				cmd = [
					'python', 'src/kbc/cqd_co.py',
					'--path', config['data'],
					'--model_path', config['model'],
					'--mode', 'valid',
					'--chain_type', chain_type,
					'--lr', str(lr),
					'--reg', str(reg),
					'--results_dir', str(results_dir),
					'--max-steps', str(max_steps),
					'--conditional', str(conditional),
					'--debug',
				]
				print(f"{query_name} - lr={lr}, reg={reg}")
				subprocess.run(cmd)
				
				# Read stats
				model_name = osp.splitext(osp.basename(config['model']))[0]
				reg_str = f'{reg}' if reg is not None else 'None'
				stats_file = results_dir / f'cont_n={model_name}_t={chain_type}_r={reg_str}_m=valid_lr={lr}_opt=adam_ms={max_steps}.json'
				
				if stats_file.exists():
					with open(stats_file) as f:
						stats = json.load(f)
					hp_key = f"lr={lr},reg={reg}"
					query_results[hp_key] = stats.get('MRRm_new', 0)
		
		# Find optimal hyperparams for this query
		if query_results:
			best_hp = max(query_results.items(), key=lambda x: x[1])
			optimal_params[query_name] = {
				'hyperparams': best_hp[0],
				'MRR': best_hp[1]
			}
	
	# Save optimal hyperparams per query type
	hp_dir = Path('results/hyperparam')
	hp_dir.mkdir(parents=True, exist_ok=True)
	
	with open(hp_dir / f'{dataset}_optimal_{cond_str}.json', 'w') as f:
		json.dump(optimal_params, f, indent=2)
	
	print(f"\n=== Optimal hyperparams for {dataset} ({cond_str}) ===")
	for query, params in optimal_params.items():
		print(f"{query}: {params['hyperparams']} (MRR={params['MRR']:.4f})")

if __name__ == "__main__":
	for dataset in EXPERIMENTS.keys():
		run_hyperparam_search(dataset, conditional=False)
		run_hyperparam_search(dataset, conditional=True)
