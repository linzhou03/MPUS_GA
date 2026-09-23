"""Read predeclared primary outputs; never substitute legacy fused scores for dual heads."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from .run_boundary_suite import COUNTS, VARIANTS, SEEDS


def summarize(root):
    rows=[]
    for variant in VARIANTS:
        for direction,n in COUNTS.items():
            seeds=[]
            for seed in SEEDS:
                folds=[]
                for subject in range(1,n+1):
                    p=root/variant/direction/f'seed_{seed}_subject_{subject:02d}.offline.json'
                    if p.exists():folds.append(json.loads(p.read_text()))
                seeds.append(dict(seed=seed,complete=len(folds)==n,folds=len(folds),
                    metrics={k:mean(f['primary'][k] for f in folds) if folds else None
                        for k in ('accuracy','balanced_accuracy','macro_f1')},
                    generator_macro_f1=mean(f['final']['generator']['macro_f1'] for f in folds) if folds else None))
            complete=all(s['complete'] for s in seeds)
            rows.append(dict(variant=variant,direction=direction,planned_folds=n*2,
                completed_folds=sum(s['folds'] for s in seeds),complete=complete,seeds=seeds,
                mean={k:mean(s['metrics'][k] for s in seeds) for k in seeds[0]['metrics']} if complete else None,
                seed_std={k:stdev(s['metrics'][k] for s in seeds) for k in seeds[0]['metrics']} if complete else None))
    return dict(note='Primary: frozen fused for r2/proto_off; mean dual heads for dual_source/dual_mcd. '
        'Only completed two-seed directions receive a combined score. Legacy summary.csv is fused-only.',rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    a=p.parse_args();print(json.dumps(summarize(a.root),indent=2))


if __name__=='__main__':main()
