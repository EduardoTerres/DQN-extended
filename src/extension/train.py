#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

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

QUERY_TYPE_MAPPING = {
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

MODE = 'test'
MAX_STEPS = 100


def parse_hparams(hparams):
	parts = dict(p.split('=') for p in hparams.split(','))
	lr = float(parts['lr'])
	reg = None if parts['reg'] == 'None' else float(parts['reg'])
	return lr, reg


def run_for(conditional):
	cond_str = 'cond' if conditional else 'uncond'
	hp_dir = Path('results/hyperparam')
	results_dir = Path(f'results/reproduce/{cond_str}-final')
	results_dir.mkdir(parents=True, exist_ok=True)

	for dataset, config in EXPERIMENTS.items():
		hp_file = hp_dir / f'{dataset}_optimal_{cond_str}.json'
		if not hp_file.exists():
			print(f"Skipping {dataset} ({cond_str}) - missing {hp_file}")
			continue
		opt = json.loads(hp_file.read_text())

		for query_type in QUERY_TYPES:
			chain_type = QUERY_TYPE_MAPPING[query_type]
			lr, reg = parse_hparams(opt[query_type]['hyperparams'])
			cmd = [
				'python', 'src/kbc/cqd_co.py',
				'--path', config['data'],
				'--model_path', config['model'],
				'--mode', MODE,
				'--chain_type', chain_type,
				'--lr', str(lr),
				'--reg', str(reg),
				'--max-steps', str(MAX_STEPS),
				'--conditional', str(conditional),
				'--results_dir', str(results_dir),
			]
			if reg is not None:
				cmd += ['--reg', str(reg)]
			print(f"Running {dataset} - {cond_str.upper()} - {query_type} ({opt[query_type]['hyperparams']})")
			subprocess.run(cmd)


if __name__ == "__main__":
	run_for(conditional=False)
	run_for(conditional=True)
