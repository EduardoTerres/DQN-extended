import json, glob
from reproduce.train_all import QUERY_TYPE_MAPPING

def load(folder, metric='MRRm_new'):
    vals = []
    for q in QUERY_TYPE_MAPPING.values():
        files = glob.glob(f'{folder}/*_t={q}_*.json')
        v = []
        for f in files:
            with open(f) as fh:
                v.append(json.load(fh).get(metric, 0))
        vals.append(sum(v) / len(v) if v else 0)
    return vals

print(f"{'Method':<15} {'Avg':>7} " + " ".join([f"{q:>7}" for q in QUERY_TYPE_MAPPING.values()]))
for name, folder in [('conditional','results/reproduce/cond-final'), ('unconditional','results/reproduce/uncond-final')]:
    r = load(folder)
    avg = sum(r) / len(r) if r else 0
    print(f"{name:<15} {avg:>7.3f} " + " ".join([f"{x:>7.3f}" for x in r]))
