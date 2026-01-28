import json
import os
from reproduce.train_all import QUERY_TYPE_MAPPING

datasets = ['FB15k', 'FB15k-237', 'NELL']

model_map = {
    'FB15k': 'FB15k-model-rank-1000-epoch-100-1602520745',
    'FB15k-237': 'FB15k-237-model-rank-1000-epoch-100-1602508358',
    'NELL': 'NELL-model-rank-1000-epoch-100-1602499096'
}

for dataset in datasets:
    print(f"\n{'='*70}")
    print(f"{dataset}")
    print(f"{'='*70}")
    print(f"{'Method':<15} {'Avg':>7} " + " ".join([f"{q:>7}" for q in QUERY_TYPE_MAPPING.keys()]))
    print("-" * 70)
    
    # Continuous (co) results
    model_name = model_map[dataset]
    results_cont = []
    for col, qtype in QUERY_TYPE_MAPPING.items():
        fname = f"results/reproduce/co/cont_n={model_name}_t={qtype}_r=None_m=test_lr=0.1_opt=adam_ms=1000.json"
        if os.path.exists(fname):
            with open(fname) as f:
                data = json.load(f)
                results_cont.append(data.get('MRRm_new', 0))
        else:
            results_cont.append(0)
    avg_cont = sum(results_cont) / len(results_cont) if results_cont else 0
    print(f"{'CQD-CO':<15} {avg_cont:>7.3f} " + " ".join([f"{r:>7.3f}" for r in results_cont]))

    # Topk (beam) results
    results_topk = []
    for col, qtype in QUERY_TYPE_MAPPING.items():
        fname = f"results/reproduce/beam/topk_d={dataset}_t=min_e={qtype}_rank=1000_k=8_sn=0.json"
        if qtype == '1_1':
            fname = f"results/reproduce/co/cont_n={model_map[dataset]}_t=1_1_r=None_m=test_lr=0.1_opt=adam_ms=1000.json"
        if os.path.exists(fname):
            with open(fname) as f:
                data = json.load(f)
                results_topk.append(data.get('MRRm_new', 0))
        else:
            results_topk.append(0)
    avg_topk = sum(results_topk) / len(results_topk) if results_topk else 0
    print(f"{'CQD-Beam':<15} {avg_topk:>7.3f} " + " ".join([f"{r:>7.3f}" for r in results_topk]))

