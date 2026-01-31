#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import os.path as osp
import json
import time

from tqdm import tqdm
import torch

from pathlib import Path

from kbc.utils import QuerDAG
from kbc.utils import preload_env
from kbc.utils import limit_dataset
from kbc.metrics import evaluation

from density.likelihood import (
	load_model,
	compute_likelihood,
)

from density.sum import get_normalization_constant_batch

DATASET = 'FB15k-237'
import matplotlib.pyplot as plt

def plot_likelihoods(likelihoods, args):
	"""Plot likelihoods evolution during optimization."""
	fontsize = 20
	plt.figure(figsize=(10, 6))
	plt.plot(likelihoods)
	plt.xlabel('Step', fontsize=fontsize)
	plt.ylabel('Likelihood', fontsize=fontsize)
	plt.title(f'Likelihood Evolution - {args.chain_type}', fontsize=fontsize)
	plt.grid(True)
	plt.tight_layout()
	
	model_name = osp.splitext(osp.basename(args.model_path))[0]
	reg_str = f'{args.reg}' if args.reg is not None else 'None'
	plt.savefig(f'likelihoods_n={model_name}_t={args.chain_type}_r={reg_str}_m={args.mode}.png')
	plt.close()


def get_likelihood_fn(model_type: str):
	"""Get likelihood function based on dataset.
	
	Model is either 'flow' or 'vae'.
	"""
	repo_root = Path(__file__).resolve().parents[2]
	model_path = repo_root / "results" / f"{model_type}_model_{DATASET}" / f"{model_type}_model_final.pt"

	model, _ = load_model(model_path=model_path, model_type=model_type, device='mps')
	likelihood_fn = (
		lambda embeddings:
		compute_likelihood(embeddings, model, model_type=model_type, device="mps", show_progress=False)
	)
	return likelihood_fn

def score_queries(args):
	mode = args.mode

	dataset = osp.basename(args.path)

	data_hard_path = osp.join(args.path, f'{dataset}_{mode}_hard.pkl')
	data_complete_path = osp.join(args.path, f'{dataset}_{mode}_complete.pkl')

	data_hard = pickle.load(open(data_hard_path, 'rb'))
	data_complete = pickle.load(open(data_complete_path, 'rb'))

	# Limit dataset to first 100 samples in DEBUG mode
	if args.debug:
		data_hard = limit_dataset(data_hard, max_samples=100)
		data_complete = limit_dataset(data_complete, max_samples=100)
		print(f"DEBUG MODE: Using only first 500 samples per chain type")

	# Instantiate singleton KBC object
	preload_env(args.model_path, data_hard, args.chain_type, mode='hard')
	env = preload_env(args.model_path, data_complete, args.chain_type,
					  mode='complete')

	queries = env.keys_hard
	test_ans_hard = env.target_ids_hard
	test_ans = env.target_ids_complete
	chains = env.chains
	kbc = env.kbc

	if args.reg is not None:
		env.kbc.regularizer.weight = args.reg

	disjunctive = args.chain_type in (QuerDAG.TYPE2_2_disj.value,
									  QuerDAG.TYPE4_3_disj.value)

	returns = None
	params = None
	likelihoods = None

	likelihood_fn = get_likelihood_fn(model_type=args.model)

	if args.chain_type == QuerDAG.TYPE1_1.value:
		# scores = kbc.model.link_prediction(chains)

		s_emb = chains[0][0]
		p_emb = chains[0][1]

		scores_lst = []
		nb_queries = s_emb.shape[0]
		for i in tqdm(range(nb_queries)):
			batch_s_emb = s_emb[i, :].view(1, -1)
			batch_p_emb = p_emb[i, :].view(1, -1)
			batch_chains = [(batch_s_emb, batch_p_emb, None)]
			batch_scores = kbc.model.link_prediction(batch_chains)
			scores_lst += [batch_scores]

		scores = torch.cat(scores_lst, 0)

	elif args.chain_type in (QuerDAG.TYPE1_2.value, QuerDAG.TYPE1_3.value):
		returns = kbc.model.optimize_chains(chains, kbc.regularizer,
										   max_steps=args.max_steps,
										   lr=args.lr,
										   optimizer=args.optimizer,
										   norm_type=args.t_norm,
										   likelihood_fn=likelihood_fn,
										#    norm_constant_fn=get_normalization_constant_batch,
										   )

	elif args.chain_type in (QuerDAG.TYPE2_2.value, QuerDAG.TYPE2_2_disj.value,
							 QuerDAG.TYPE2_3.value):
		returns = kbc.model.optimize_intersections(chains, kbc.regularizer,
												  max_steps=args.max_steps,
												  lr=args.lr,
												  optimizer=args.optimizer,
												  norm_type=args.t_norm,
												  disjunctive=disjunctive,
												  likelihood_fn=likelihood_fn,
												  norm_constant_fn=get_normalization_constant_batch,
												  )

	elif args.chain_type == QuerDAG.TYPE3_3.value:
		returns = kbc.model.optimize_3_3(chains, kbc.regularizer,
										max_steps=args.max_steps,
										lr=args.lr,
										optimizer=args.optimizer,
										norm_type=args.t_norm,
										likelihood_fn=likelihood_fn)

	elif args.chain_type in (QuerDAG.TYPE4_3.value,
							 QuerDAG.TYPE4_3_disj.value):
		returns = kbc.model.optimize_4_3(chains, kbc.regularizer,
										max_steps=args.max_steps,
										lr=args.lr,
										optimizer=args.optimizer,
										norm_type=args.t_norm,
										disjunctive=disjunctive,
										likelihood_fn=likelihood_fn)
	else:
		raise ValueError(f'Uknown query type {args.chain_type}')

	if returns is not None:
		scores, params, likelihoods = returns
	
	# Plot and save likelihoods evolution
	if likelihoods:
		plot_likelihoods(likelihoods, args)


	return scores, params, queries, test_ans, test_ans_hard


def main(args):
	print(args)

	start_time = time.time()

	scores, params, queries, test_ans, test_ans_hard = score_queries(args)
	metrics = evaluation(scores, queries, test_ans, test_ans_hard)
	
	end_time = time.time()
	metrics['time_taken'] = end_time - start_time
	
	if params is not None:
		norms = torch.norm(params[0], dim=1)
		print(f"Min norm: {norms.min():.4f}, Max norm: {norms.max():.4f}, Mean norm: {norms.mean():.4f}")

	print(metrics)

	model_name = osp.splitext(osp.basename(args.model_path))[0]
	reg_str = f'{args.reg}' if args.reg is not None else 'None'
	
	# Create results directory if it doesn't exist
	if not osp.exists(args.results_dir):
		os.makedirs(args.results_dir)

	with open(f'{args.results_dir}/cont_n={model_name}_t={args.chain_type}_r={reg_str}_m={args.mode}_lr={args.lr}_opt={args.optimizer}_ms={args.max_steps}.json', 'w') as f:
		json.dump(metrics, f)


if __name__ == "__main__":

	datasets = ['FB15k', 'FB15k-237', 'NELL']
	modes = ['valid', 'test', 'train']
	chain_types = [t.value for t in QuerDAG]

	t_norms = ['min', 'product']

	parser = argparse.ArgumentParser(description="Complex Query Decomposition - Continuous Optimisation")
	parser.add_argument('--path', help='Path to directory containing queries', default=f"data/{DATASET}")
	parser.add_argument(
		'--model_path', help="The path to the KBC model. Can be both relative and full",
		default=f"models/{DATASET}-model-rank-1000-epoch-100-1602508358.pt",
	)
	parser.add_argument('--dataset', choices=datasets, help="Dataset in {}".format(datasets), default=DATASET)
	parser.add_argument('--mode', choices=modes, default='test',
						help="Dataset validation mode in {}".format(modes))
	parser.add_argument('--debug', action='store_true', help='Activate debug mode with reduced dataset size', default=True)

	parser.add_argument('--model', choices=['flow', 'vae'], default='vae', help="Density model for likelihood computation")

	parser.add_argument('--chain_type', choices=chain_types, default=QuerDAG.TYPE2_2_disj.value,
						help="Chain type experimenting for ".format(chain_types))

	parser.add_argument('--t_norm', choices=t_norms, default='prod', help="T-norms available are ".format(t_norms))
	parser.add_argument('--reg', type=float, help='Regularization coefficient', default=None)
	parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
	parser.add_argument('--optimizer', type=str, default='adam',
						choices=['adam', 'adagrad', 'sgd'])
	parser.add_argument('--max-steps', type=int, default=1000)

	parser.add_argument('--results_dir', type=str, default='results/reproduce/', help='Directory to save results')

	main(parser.parse_args())
