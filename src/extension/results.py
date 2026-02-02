import json, glob
from reproduce.train_all import QUERY_TYPE_MAPPING

DATASETS = ['FB15k', 'FB15k-237', 'NELL']

def load(folder, dataset, metric='MRRm_new'):
    vals = []
    for q in QUERY_TYPE_MAPPING.values():
        # Get all files for this query type
        all_files = glob.glob(f'{folder}/*_t={q}_*.json')
        v = []
        for f in all_files:
            # Extract dataset name from filename (between 'n=' and '-model')
            if 'n=' in f:
                extracted = f.split('n=')[1].split('-model')[0]
                if extracted == dataset:
                    with open(f) as fh:
                        v.append(json.load(fh).get(metric, 0))
        vals.append(sum(v) / len(v) if v else 0)
    return vals

for dataset in DATASETS:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'Avg':>7} " + " ".join([f"{q:>7}" for q in QUERY_TYPE_MAPPING.values()]))
    for name, folder in [('conditional','results/reproduce/cond-final'), ('unconditional','results/reproduce/uncond-final')]:
        r = load(folder, dataset)
        avg = sum(r) / len(r) if r else 0
        print(f"{name:<15} {avg:>7.3f} " + " ".join([f"{x:>7.3f}" for x in r]))
