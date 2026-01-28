#!/usr/bin/env python3
import subprocess
from pathlib import Path

# Parameter configurations for each dataset and method
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

QUERY_TYPES = [
	'1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2u', 'up'
]

# Mapping from shorthand notation to QuerDAG enum values
QUERY_TYPE_MAPPING = {
	'1p': '1_1',
	'2p': '1_2',
	'3p': '1_3',
	'2i': '2_2',
	'3i': '2_3',
	'ip': '4_3',
	'pi': '3_3',
	'2u': '2_2_disj',
	'up': '4_3_disj'
}

METHODS = ['beam'] #, 'co']

DEBUG = True

def run_experiments():
	results_dir = Path('results/reproduce')
	results_dir.mkdir(parents=True, exist_ok=True)
	
	for dataset, config in EXPERIMENTS.items():
		for method in METHODS:
			if method == 'beam':
				cmd = [
					'python', 'src/kbc/cqd_beam.py',
					config['data'],
					'--model_path', config['model'],
					'--mode', 'test',
					'--results_dir', 'results/reproduce/beam/',
					'--candidates', '8',
					*(['--debug'] if DEBUG else []),
				]
				print(f"Running {dataset} - {method.upper()} - All Query Types")
				subprocess.run(cmd)
			else:  # co
				for query_type in QUERY_TYPES:
					# Map the query type to the QuerDAG enum value
					chain_type = QUERY_TYPE_MAPPING[query_type]
					cmd = [
						'python', 'src/kbc/cqd_co.py',
						'--path', config['data'],
						'--model_path', config['model'],
						'--mode', 'test',
						'--chain_type', chain_type,
						'--results_dir', 'results/reproduce/co/',
						*(['--debug'] if DEBUG else []),
					]
				
					print(f"Running {dataset} - {method.upper()} - {query_type}")
					subprocess.run(cmd)

if __name__ == "__main__":
	run_experiments()
