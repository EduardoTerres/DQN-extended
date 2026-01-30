#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

HYPERPARAM_GRID = {
	'learning_rate': [0.001, 0.01, 0.1],
	'batch_size': [32, 64, 128],
	'embedding_dim': [100, 200, 500],
}

EXPERIMENTS = {
	'FB15k': {
		'model': 'models/FB15k-model-rank-1000-epoch-100-1602520745.pt',
		'data': 'data/FB15k',
	},
	'FB15k-237': {
		'model': 'models/FB15k-237-model-rank-1000-epoch-100-1602508358.pt',
		'data': 'data/FB15k-237',
	},
	'NELL': {
		'model': 'models/NELL-model-rank-1000-epoch-100-1602499096.pt',
		'data': 'data/NELL',
	},
}

def run_hyperparam_search(dataset):
	if dataset not in EXPERIMENTS:
		print(f"Dataset {dataset} not found. Available: {list(EXPERIMENTS.keys())}")
		sys.exit(1)
	
	config = EXPERIMENTS[dataset]
	results_dir = Path(f'results/hyperparam_search_{dataset}')
	results_dir.mkdir(parents=True, exist_ok=True)
	
	run_id = 0
	for lr in HYPERPARAM_GRID['learning_rate']:
		for bs in HYPERPARAM_GRID['batch_size']:
			for emb_dim in HYPERPARAM_GRID['embedding_dim']:
				run_id += 1
				cmd = [
					'python', 'src/kbc/cqd_co.py',
					'--path', config['data'],
					'--model_path', config['model'],
					'--mode', 'val',
					'--learning_rate', str(lr),
					'--batch_size', str(bs),
					'--embedding_dim', str(emb_dim),
					'--results_dir', str(results_dir),
					'--debug',
				]
				print(f"[{run_id}] {dataset} - lr={lr}, bs={bs}, dim={emb_dim}")
				subprocess.run(cmd)

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print(f"Usage: {sys.argv[0]} <dataset>")
		print(f"Available datasets: {list(EXPERIMENTS.keys())}")
		sys.exit(1)
	
	run_hyperparam_search(sys.argv[1])
