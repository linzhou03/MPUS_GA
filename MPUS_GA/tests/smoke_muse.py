"""Actual R2 train_step + MUSE on synthetic EEG; no target truth or real-data tuning."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import torch

from MPUS_GA.tests.test_muse_integration import fixture, with_ids, step, batch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--iterations',type=int,choices=(20,350,650),required=True)
    parser.add_argument('--variant',default='starvation')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(43)
    parts=fixture(args.variant)
    source=with_ids(batch(True,18))
    # The synthetic source task is deliberately learnable, with different class noise.
    for x in source['x'].values():
        x.mul_(.03)
        for i,c in enumerate(source['y']):x[i,:,:,int(c)]+=3.
    target=deepcopy(source);del target['y'];target['domain_id'].fill_(1)
    for x in target['x'].values():
        x[1::3]=.5*x[0::3]+.5*x[2::3]
        x.add_(torch.randn_like(x)*.02)
    started=time.monotonic();trace=[];active_hard=active_partial=active_sub=0
    for iteration in range(1,args.iterations+1):
        row=step(parts,source,target,iteration)
        assert torch.isfinite(torch.tensor(row['total']))
        muse=row['muse']
        if iteration<=300:assert muse['added_loss']==0
        if iteration<600:assert muse['L_sub']==0 and not any(muse['alignment_gate'])
        if iteration>300:
            active_hard+=int(muse['L_hard']>0)
            active_partial+=int(muse['L_partial']>0)
        if iteration>=600:active_sub+=int(muse['L_sub']>0)
        if iteration==1 or iteration%50==0 or iteration==args.iterations:
            trace.append(muse)
            print(json.dumps({'iteration':iteration,'hard':muse['hard_count'],'partial':muse['partial_effective_count'],
                              'coverage':muse['coverage'],'starvation':muse['starvation'],'gate':muse['alignment_gate'],
                              'L_hard':muse['L_hard'],'L_partial':muse['L_partial'],'L_sub':muse['L_sub']}),flush=True)
    if args.iterations>300:
        assert active_hard>0 and active_partial>0,(active_hard,active_partial)
        assert max(trace[-1]['starvation'])-min(trace[-1]['starvation'])>1e-4
    if args.variant=='full' and args.iterations==650:
        assert any(trace[-1]['alignment_gate']) and active_sub>0
    report={'status':'PASS','kind':'synthetic_EEG_actual_R2_train_step','iterations':args.iterations,'variant':args.variant,
            'seconds':time.monotonic()-started,'hard_active_steps_after_300':active_hard,
            'partial_active_steps_after_300':active_partial,'trace':trace}
    report['subgroup_active_steps_after_600']=active_sub
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print('PASS',args.iterations,'steps',flush=True)


if __name__=='__main__':main()
